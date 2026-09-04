"""Band structure coloured by <Sz>, built from the EXPORTED lattice.

This is a check on what `load_material()` actually hands to KITE, not on the
raw file: H(k) is reassembled from the `kite.lattice.Lattice` object's own
sublattices and hoppings, applying the same conjugate mirroring KITE's
`add_one_hopping()` does at export time. So it exercises the half-hopping
registration, the value_threshold cut, the dimensionality reduction and the
sublattice/position assignment -- the places a bug would actually live.

It also rebuilds H(k) directly from the parsed data, truncated identically,
and reports the maximum eigenvalue difference between the two. That number
isolates pipeline errors from truncation: it should be at machine precision.

The physics check: in a 2H-MoS2 monolayer the valence band at K is split by
spin-orbit coupling into two branches of opposite, nearly pure Sz, and the
sign reverses at K'. A wrong orbital ordering or spin block assignment does
not reproduce that.

    python plot_spin_texture.py [material_dir] [orbitals_per_atom] [max_hopping_range]
"""
import os
import sys
import types

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _ensure_kite():
    """Use the real kite package if importable; otherwise install a minimal
    stand-in that records exactly what build_lattice() registers. The
    stand-in stores sublattices/hoppings verbatim, so the reconstruction
    below sees the same data a real export would."""
    try:
        from kite import lattice  # noqa: F401
        return False
    except Exception:
        pass

    class Lattice:
        def __init__(self, *vectors):
            self.vectors = [np.asarray(v, dtype=float) for v in vectors]
            self.sublattices = []
            self.hoppings = []

        def add_sublattices(self, *subs):
            self.sublattices.extend(subs)

        def add_hoppings(self, *hops):
            self.hoppings.extend(hops)

    mod = types.ModuleType("kite")
    lat = types.ModuleType("kite.lattice")
    lat.Lattice = Lattice
    mod.lattice = lat
    sys.modules["kite"] = mod
    sys.modules["kite.lattice"] = lat
    return True


STUBBED = _ensure_kite()

import rs_lmto_import as rli  # noqa: E402
from analyze_hamiltonian import parse_lat, parse_hr, reorder_spin_major_to_atom_major  # noqa: E402


def hamiltonian_from_lattice(lattice, lat_vectors, dims):
    """Reassemble H(k) from the registered sublattices and hoppings, adding
    the conjugate mirror KITE generates itself."""
    names = [s[0] for s in lattice.sublattices]
    onsite = np.array([float(np.real(s[2])) for s in lattice.sublattices])
    index = {name: i for i, name in enumerate(names)}
    n = len(names)

    rows, cols, vals, Rc = [], [], [], []
    for R, ni, nj, value in lattice.hoppings:
        i, j = index[ni], index[nj]
        r_full = np.zeros(3)
        for slot, axis in enumerate(dims):
            r_full[axis] = R[slot]
        rows.append(i); cols.append(j); vals.append(complex(value))
        Rc.append(r_full @ lat_vectors)
    rows = np.array(rows); cols = np.array(cols)
    vals = np.array(vals); Rc = np.array(Rc)

    def H(k):
        k3 = np.zeros(3)
        k3[:len(k)] = k
        phase = np.exp(1j * (Rc @ k3))
        M = np.zeros((n, n), dtype=complex)
        np.add.at(M, (rows, cols), vals * phase)
        M = M + M.conj().T            # the mirror KITE adds itself
        M[np.diag_indices(n)] += onsite
        return M

    return H, names


def sz_from_names(names, spin_labels=("u", "d")):
    """<Sz> weight per sublattice, read off the registered names."""
    up, dn = spin_labels
    sz = np.zeros(len(names))
    for i, name in enumerate(names):
        tag = name.split("_")[-1]
        if tag.startswith(up):
            sz[i] = +0.5
        elif tag.startswith(dn):
            sz[i] = -0.5
        else:
            raise ValueError(f"cannot tell the spin of sublattice {name!r}")
    return sz


def kpath(lat_vectors, dims, n=120):
    a = lat_vectors[np.ix_(dims, [0, 1])]
    B = 2 * np.pi * np.linalg.inv(a).T
    G = np.zeros(2)
    # Which combination of b1/b2 is the BZ corner depends on the cell
    # convention (60 vs 120 degrees between a1 and a2), so don't hard-code
    # one: pick the shortest of the candidates, which is the corner of the
    # first zone, and check it against the expected |K| = |b1|/sqrt(3).
    expected = np.linalg.norm(B[0]) / np.sqrt(3)
    candidates = [(B[0] + B[1]) / 3, (2 * B[0] + B[1]) / 3,
                  (B[0] - B[1]) / 3, (B[0] + 2 * B[1]) / 3,
                  (2 * B[0] - B[1]) / 3, (B[0] - 2 * B[1]) / 3]
    matches = [c for c in candidates
               if np.isclose(np.linalg.norm(c), expected, rtol=1e-6)]
    if not matches:
        raise ValueError(
            f"no (m*b1 + n*b2)/3 candidate has |K| = |b1|/sqrt(3) = "
            f"{expected:.6f} -- this k-path is only valid for a hexagonal "
            f"lattice."
        )
    K = matches[0]
    M = B[0] / 2
    pts = [-K, G, K, M, G]
    labels = ["K'", "$\\Gamma$", "K", "M", "$\\Gamma$"]
    ks, ticks = [], [0]
    for p, q in zip(pts[:-1], pts[1:]):
        for t in np.linspace(0, 1, n, endpoint=False):
            ks.append(p + t * (q - p))
        ticks.append(len(ks))
    ks.append(pts[-1])
    return np.array(ks), ticks, labels


def raw_hamiltonian(material_dir, orbitals_per_atom, max_hopping_range,
                    value_threshold=1e-8):
    """H(k) straight from the parsed file, permuted and truncated the same
    way -- the reference the exported lattice is compared against."""
    hr_name = [f for f in os.listdir(material_dir) if f.endswith("_hr.dat")][0]
    lat_name = [f for f in os.listdir(material_dir) if f.endswith("_lat.dat")][0]
    vectors, labels, _pos, _fermi = parse_lat(os.path.join(material_dir, lat_name))
    hr = parse_hr(os.path.join(material_dir, hr_name))
    hr = reorder_spin_major_to_atom_major(hr, len(labels), orbitals_per_atom)
    if max_hopping_range is not None:
        hr = {k: v for k, v in hr.items()
              if max(abs(c) for c in k[0]) <= max_hopping_range}
    hr = {k: v for k, v in hr.items() if abs(v) >= value_threshold}

    n = len(labels) * orbitals_per_atom
    R = np.array([k[0] for k in hr]); I = np.array([k[1] - 1 for k in hr])
    J = np.array([k[2] - 1 for k in hr]); V = np.array(list(hr.values()))
    Rc = R @ vectors

    def H(k):
        k3 = np.zeros(3)
        k3[:len(k)] = k
        M = np.zeros((n, n), dtype=complex)
        np.add.at(M, (I, J), V * np.exp(1j * (Rc @ k3)))
        return (M + M.conj().T) / 2

    return H


def main(material_dir="mos2", orbitals_per_atom=18, max_hopping_range=2,
         out=None):
    if STUBBED:
        print("note: the `kite` package is not importable; using a minimal "
              "stand-in that records what build_lattice() registers.\n")

    lattice, stats = rli.load_material(
        material_dir, orbitals_per_atom=orbitals_per_atom,
        max_hopping_range=max_hopping_range)
    print("import stats:", stats)

    lat_name = [f for f in os.listdir(material_dir) if f.endswith("_lat.dat")][0]
    vectors, labels, _pos, fermi = parse_lat(os.path.join(material_dir, lat_name))
    dims = stats["periodic_dims"]

    H_exp, names = hamiltonian_from_lattice(lattice, vectors, dims)
    H_raw = raw_hamiltonian(material_dir, orbitals_per_atom, max_hopping_range)
    sz = sz_from_names(names)
    ks, ticks, klabels = kpath(vectors, dims)

    n_orb = len(names)
    E = np.zeros((len(ks), n_orb))
    S = np.zeros((len(ks), n_orb))
    worst = 0.0
    for m, k in enumerate(ks):
        Me = H_exp(k)
        w, v = np.linalg.eigh(Me)
        E[m] = w
        S[m] = np.einsum("in,i,in->n", v.conj(), sz, v).real
        worst = max(worst, np.abs(w - np.linalg.eigvalsh(H_raw(k))).max())
    print(f"\nmax |E(exported lattice) - E(raw data, same truncation)| "
          f"over the path: {worst:.3e} eV")

    x = np.arange(len(ks))
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, window, title in ((axes[0], None, "full spectrum"),
                              (axes[1], (fermi - 2.5, fermi + 3.0), "around $E_F$")):
        for b in range(n_orb):
            sc = ax.scatter(x, E[:, b], c=S[:, b], cmap="coolwarm",
                            vmin=-0.5, vmax=0.5, s=3, linewidths=0)
        ax.axhline(fermi, color="k", ls="--", lw=0.8)
        for t in ticks:
            ax.axvline(t, color="0.85", lw=0.8)
        ax.set_xticks(ticks); ax.set_xticklabels(klabels)
        ax.set_xlim(0, len(ks) - 1)
        if window:
            ax.set_ylim(*window)
        ax.set_ylabel("E (eV)"); ax.set_title(title)
    cb = fig.colorbar(sc, ax=axes, fraction=0.03)
    cb.set_label(r"$\langle S_z \rangle$")
    fig.suptitle(f"{material_dir}: exported lattice, bands coloured by "
                 f"$\\langle S_z\\rangle$ (max_hopping_range={max_hopping_range})")
    out = out or f"{material_dir}_spin_texture.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")

    print()
    for name, idx in (("K", ticks[2]), ("K'", 0)):
        occ = np.where(E[idx] < fermi)[0]
        top = occ[-2:]
        print(f"{name:3s} top valence pair: E = {E[idx][top[0]]:.4f}, "
              f"{E[idx][top[1]]:.4f}   splitting = "
              f"{abs(E[idx][top[1]] - E[idx][top[0]]) * 1000:6.1f} meV   "
              f"<Sz> = {S[idx][top[0]]:+.3f}, {S[idx][top[1]]:+.3f}")
    return out


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "mos2"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 18
    r = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    main(d, n, r)
