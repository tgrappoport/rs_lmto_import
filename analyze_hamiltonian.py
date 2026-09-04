"""Analyse the RS-LMTO Hamiltonian of a material folder, as stored.

Deliberately standalone: parses the `*_lat.dat`/`*_hr.dat` pair directly
instead of going through rs_lmto_import.build_lattice(), so it runs without
the `kite` package installed and reports on the RAW data -- before any
truncation, threshold, or core-orbital dropping the importer would apply.

    python analyze_hamiltonian.py <material_dir> [orbitals_per_atom]
    python analyze_hamiltonian.py mos2 18

Assumes the same basis convention as rs_lmto_import: contiguous per-atom
blocks in `_lat.dat` position order, spin-major within each atom (all 9 spd
orbitals of spin up, then all 9 of spin down).
"""
import os
import re
import sys
import collections

import numpy as np

SPD = ("s", "px", "py", "pz", "dxy", "dyz", "dzx", "dx2-y2", "d3z2-r2")


def parse_lat(path):
    txt = open(path).read()

    def block(name):
        m = re.search(r"begin\s+%s(.*?)end\s+%s" % (name, name), txt, re.S)
        if m is None:
            raise ValueError(f"{path}: missing '{name}' block")
        return [l.strip() for l in m.group(1).strip().splitlines() if l.strip()]

    vectors = np.array([[float(x) for x in l.split(",")] for l in block("unit_cell")])
    labels, positions = [], []
    for line in block("position"):
        parts = line.split(",")
        labels.append(parts[0].strip())
        positions.append([float(x) for x in parts[1:]])
    return vectors, labels, np.array(positions), float(block("fermi")[0])


def parse_hr(path):
    data = {}
    for line_no, line in enumerate(open(path), 1):
        t = line.split()
        if not t:
            continue
        if len(t) != 7:
            raise ValueError(f"{path}:{line_no}: expected 7 fields, got {len(t)}")
        r = (int(t[0]), int(t[1]), int(t[2]))
        data[(r, int(t[3]), int(t[4]))] = complex(float(t[5]), float(t[6]))
    return data


def reorder_spin_major_to_atom_major(hr, n_atoms, orbitals_per_atom, n_spin=2):
    """File order -> atom-major order.

    The `_hr.dat` is written spin-major over the whole system (all atoms for
    spin up, then all atoms for spin down); this remaps to one contiguous
    2*n_spatial block per atom, which is what the per-atom analysis below
    assumes. Same permutation as rs_lmto_import's function of the same name,
    duplicated here to keep this script runnable without the `kite` package.
    """
    n_spatial = orbitals_per_atom // n_spin
    perm = {}
    for spin in range(n_spin):
        for atom in range(n_atoms):
            for spatial in range(n_spatial):
                k_file = spin * (n_atoms * n_spatial) + atom * n_spatial + spatial
                k_int = atom * orbitals_per_atom + spin * n_spatial + spatial
                perm[k_file + 1] = k_int + 1
    return {(r, perm[i], perm[j]): v for (r, i, j), v in hr.items()}


def _Lz():
    """Lz in the _SPD_BASIS order (s, px, py, pz, dxy, dyz, dzx, dx2-y2,
    d3z2-r2), dimensionless L/hbar."""
    Lz = np.zeros((9, 9), dtype=complex)
    for i, j, v in ((1, 2, -1j), (4, 7, 2j), (5, 6, 1j)):
        Lz[i, j] = v
        Lz[j, i] = np.conj(v)
    return Lz


def section(title):
    print("\n" + title)
    print("-" * len(title))


def analyse(material_dir, orbitals_per_atom=18):
    hr_path = [f for f in os.listdir(material_dir) if f.endswith("_hr.dat")]
    lat_path = [f for f in os.listdir(material_dir) if f.endswith("_lat.dat")]
    if len(hr_path) != 1 or len(lat_path) != 1:
        raise ValueError(f"{material_dir}: need exactly one *_hr.dat and one *_lat.dat")
    vectors, labels, positions, fermi = parse_lat(os.path.join(material_dir, lat_path[0]))
    hr_file = parse_hr(os.path.join(material_dir, hr_path[0]))

    n_atoms = len(labels)
    # The file is spin-major over the whole system; everything below wants
    # one contiguous block per atom. Without this every per-atom quantity
    # (on-site energies, spin splitting, SOC) comes out scrambled.
    hr = reorder_spin_major_to_atom_major(hr_file, n_atoms, orbitals_per_atom)
    n_orb = n_atoms * orbitals_per_atom
    n_spatial = orbitals_per_atom // 2  # spin-major, 2 spins

    section("Basis")
    print(f"atoms                : {n_atoms}  {labels}")
    print(f"orbitals/atom        : {orbitals_per_atom}  (2 spins x {n_spatial} spatial)")
    print(f"total orbitals       : {n_orb}")
    print(f"stored matrix elems  : {len(hr)}")
    print(f"highest index in hr  : {max(i for (_r, i, _j) in hr)}")
    print(f"Fermi level          : {fermi}")

    # ---- geometry of the stored R vectors -----------------------------
    section("Real-space range")
    Rs = sorted({r for (r, _i, _j) in hr})
    axes_used = [ax for ax in range(3) if any(r[ax] for r in Rs)]
    print(f"periodic axes (R != 0 somewhere): {axes_used}")
    dist = {}
    for r in Rs:
        dist[r] = np.linalg.norm(np.dot(r, vectors))
    shells = collections.defaultdict(list)
    for r, d in dist.items():
        shells[round(d, 6)].append(r)
    print(f"{len(Rs)} distinct R vectors in {len(shells)} distance shells:")
    for d in sorted(shells):
        rs = shells[d]
        idx_range = max(max(abs(c) for c in r) for r in rs)
        # weight of the shell: largest |H| anywhere in it
        w = max(abs(v) for (r, _i, _j), v in hr.items() if r in set(rs))
        print(f"  |R| = {d:6.3f}   {len(rs):3d} vectors   max|R_i| = {idx_range}"
              f"   max|H| = {w:.4e}")

    # ---- hermiticity ---------------------------------------------------
    section("Hermiticity  H(R)_ij = conj(H(-R)_ji)")
    missing, worst, worst_key = 0, 0.0, None
    for (r, i, j), v in hr.items():
        w = hr.get((tuple(-c for c in r), j, i))
        if w is None:
            missing += 1
            continue
        d = abs(v - w.conjugate())
        if d > worst:
            worst, worst_key = d, (r, i, j)
    scale = max(abs(v) for v in hr.values())
    print(f"missing conjugate partners : {missing}")
    print(f"max |H(R)_ij - conj(H(-R)_ji)| : {worst:.3e}   at R={worst_key}")
    print(f"largest |H| anywhere           : {scale:.3e}   -> relative {worst/scale:.2e}")

    # ---- onsite block --------------------------------------------------
    section("On-site block  H(R=0)")
    H0 = np.zeros((n_orb, n_orb), dtype=complex)
    for (r, i, j), v in hr.items():
        if r == (0, 0, 0):
            H0[i - 1, j - 1] = v
    diag = np.real(np.diag(H0))
    offdiag = H0 - np.diag(np.diag(H0))
    print(f"max |imag| on the diagonal : {np.abs(np.imag(np.diag(H0))).max():.3e}")
    print(f"max |off-diagonal| at R=0  : {np.abs(offdiag).max():.3e}")
    print("on-site energies per atom (spin up / spin down):")
    for a, lab in enumerate(labels):
        base = a * orbitals_per_atom
        up = diag[base:base + n_spatial]
        dn = diag[base + n_spatial:base + orbitals_per_atom]
        print(f"  {lab}")
        for o in range(n_spatial):
            name = SPD[o] if n_spatial == len(SPD) else f"orb{o}"
            print(f"    {name:>8}  {up[o]:10.5f}  {dn[o]:10.5f}   "
                  f"split = {up[o]-dn[o]:+9.5f}")

    # ---- spin structure / spin-orbit coupling --------------------------
    section("Spin structure and spin-orbit coupling")
    if n_spatial != 9:
        print(f"{n_spatial} spatial orbitals per spin -- the spd analysis below "
              f"assumes 9; skipping")
    else:
        Lz = _Lz()
        print("For each atom: the on-site up-up and down-down blocks are compared.")
        print("A real exchange splitting shows up on the DIAGONAL; spin-orbit")
        print("coupling shows up OFF-diagonal, as H_uu - H_dd = lambda*Lz.")
        print()
        for a, lab in enumerate(labels):
            b = a * orbitals_per_atom
            uu = H0[b:b + n_spatial, b:b + n_spatial]
            dd = H0[b + n_spatial:b + orbitals_per_atom,
                    b + n_spatial:b + orbitals_per_atom]
            ud = H0[b:b + n_spatial, b + n_spatial:b + orbitals_per_atom]
            D = uu - dd
            exch = np.abs(np.real(np.diag(uu)) - np.real(np.diag(dd))).max()
            # lambda per shell: Lz[1,2] = -1j (p), Lz[4,7] = 2j (d)
            lam_p = (D[1, 2] / -1j).real
            lam_d = (D[4, 7] / 2j).real
            lam_d2 = (D[5, 6] / 1j).real   # independent d element, must agree
            model = np.zeros_like(D)
            for (i, j), lam in (((1, 2), lam_p), ((4, 7), lam_d), ((5, 6), lam_d)):
                model[i, j] = lam * Lz[i, j]
                model[j, i] = np.conj(model[i, j])
            resid = np.linalg.norm(D - model) / max(np.linalg.norm(D), 1e-30)
            print(f"  {lab}")
            print(f"    exchange splitting (diagonal) : {exch:.3e}"
                  + ("   -> non-magnetic" if exch < 1e-6 else "   -> SPIN-POLARISED"))
            print(f"    lambda_p                      : {lam_p:+.6f}")
            print(f"    lambda_d                      : {lam_d:+.6f}"
                  f"   (independent element: {lam_d2:+.6f})")
            print(f"    residual of H_uu-H_dd vs lambda*Lz : {resid:.2e}")
            print(f"    max |H| in the up-down block  : {np.abs(ud).max():.4e}")
        print()
        print("A residual at machine precision, two independent d elements giving")
        print("the same lambda_d, and a zero diagonal splitting together confirm the")
        print("orbital ordering (_SPD_BASIS) and the spin-major file layout.")

    # ---- decay with distance -------------------------------------------
    section("Hopping magnitude vs distance")
    by_shell = collections.defaultdict(float)
    count = collections.defaultdict(int)
    for (r, i, j), v in hr.items():
        if r == (0, 0, 0) and i == j:
            continue
        d = round(dist[r], 6)
        by_shell[d] = max(by_shell[d], abs(v))
        if abs(v) > 1e-8:
            count[d] += 1
    for d in sorted(by_shell):
        print(f"  |R| = {d:6.3f}   max|t| = {by_shell[d]:.4e}   "
              f"nonzero elems = {count[d]}")

    # ---- spectrum ------------------------------------------------------
    section("Spectrum")
    Rarr = np.array([k[0] for k in hr])
    Iarr = np.array([k[1] - 1 for k in hr])
    Jarr = np.array([k[2] - 1 for k in hr])
    Varr = np.array(list(hr.values()))
    Rcart = Rarr @ vectors

    def H_of_k(k):
        k3 = np.zeros(3)
        k3[:len(k)] = k
        M = np.zeros((n_orb, n_orb), dtype=complex)
        np.add.at(M, (Iarr, Jarr), Varr * np.exp(1j * (Rcart @ k3)))
        return M

    if len(axes_used) == 2:
        a2 = vectors[np.ix_(axes_used, [0, 1])]
        B = 2 * np.pi * np.linalg.inv(a2).T
        grid = []
        n_k = 12
        for m in range(n_k):
            for n in range(n_k):
                grid.append(m / n_k * B[0] + n / n_k * B[1])
        E = np.array([np.linalg.eigvalsh((lambda M: (M + M.conj().T) / 2)(H_of_k(k)))
                      for k in grid])
        print(f"sampled a {n_k}x{n_k} k-grid")
        print(f"eigenvalue range      : {E.min():.4f} .. {E.max():.4f}")
        print(f"Fermi level           : {fermi}")
        below = (E < fermi).sum() / len(grid)
        print(f"average bands below Ef: {below:.2f} of {n_orb}")
        # is there a gap at Ef?
        occ = E[E < fermi].max()
        emp = E[E >= fermi].min()
        print(f"highest occupied      : {occ:.4f}")
        print(f"lowest unoccupied     : {emp:.4f}")
        print(f"gap across Ef         : {emp - occ:.4f}"
              + ("   (metallic / no gap on this grid)" if emp - occ < 1e-3 else ""))
        # deep states, the ones drop_core_orbitals() would target
        deep = np.sort(np.unique(np.round(E.min(axis=0), 3)))[:6]
        print(f"lowest band minima    : {deep}")
    else:
        print(f"{len(axes_used)} periodic axes -- k-grid analysis only implemented for 2D")


if __name__ == "__main__":
    material = sys.argv[1] if len(sys.argv) > 1 else "mos2"
    n_orb_atom = int(sys.argv[2]) if len(sys.argv) > 2 else 18
    analyse(material, n_orb_atom)
