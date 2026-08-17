"""Import an RS-LMTO-ASA real-space Hamiltonian into a kite.lattice.Lattice.

RS-LMTO-ASA (real-space linear muffin-tin orbital, atomic sphere
approximation) codes can export a tight-binding-like Hamiltonian as a pair
of plain-text files:

  - a `_lat.dat` file: `begin unit_cell` / `end unit_cell` (3 lattice
    vectors, comma-separated), `begin position` / `end position` (one
    "label, x, y, z" line per atom), and `begin fermi` / `end fermi` (a
    single Fermi-level value);
  - a `_hr.dat` file: a header-less Wannier90-style real-space Hamiltonian,
    one line per nonzero-or-not matrix element:
    `R1 R2 R3 i j Re(H_ij(R)) Im(H_ij(R))`, with i/j 1-indexed orbital
    labels running over ALL orbitals of ALL atoms (not per-atom indices).

This module has no knowledge of "spd" or physical orbital character -- it
only needs to know how many orbitals belong to each atom (assumed the SAME
count for every atom, in contiguous blocks, in the same atom order as the
`_lat.dat` position list) to build one KITE sublattice per orbital, all
orbitals of one atom sharing that atom's position.

Convention check performed once, empirically, before writing this module:
the `_hr.dat` data stores BOTH directions of every hopping explicitly, i.e.
H(R)_ij and H(-R)_ji are both present and satisfy H(R)_ij == conj(H(-R)_ji)
to numerical precision (checked exactly at R=0 across all pairs of a real
72-orbital dataset, and across a random sample of 2000 pairs spanning every
R shell -- 1e-14 agreement for that dataset; a second, 54-orbital dataset
agreed to ~1.5e-5, i.e. still clean but visibly less tightly converged,
worth rechecking per new dataset rather than assumed). KITE's own
`Lattice.add_one_hopping()` already auto-generates that same conjugate
mirror at export time (see its docstring), so this importer must register
only ONE direction per bond -- registering both would silently double
every hopping. build_lattice() below keeps exactly the "positive R" half
(plus the upper triangle i<j at R=(0,0,0); i==j at R=(0,0,0) is the onsite
energy, handled separately, not registered as a hopping at all) and lets
KITE regenerate the other half.

Periodicity is auto-detected, not assumed 3D: a `_lat.dat` always lists 3
unit-cell vectors, but a slab/monolayer DFT calculation typically pads one
direction with vacuum and never actually uses it periodically (e.g. a 2D
material's out-of-plane R-component is identically 0 in every _hr.dat row,
while the corresponding lattice vector is much longer than the in-plane
ones -- a vacuum-spacing artifact of the supercell, not real periodicity).
build_lattice() keeps only the lattice-vector axes for which some _hr.dat
row has a nonzero R-component in that slot, and builds a Lattice with only
that many primitive vectors -- atom positions stay full 3D regardless (a
buckled/out-of-plane offset with no periodic direction of its own is
exactly the situation kite.visualize.hamiltonian_k already supports, e.g.
for phosphorene).
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from kite import lattice as latt

RVec = Tuple[int, ...]

# KITE's compiled ghost-cell width (Src/Generic.hpp: #define NGHOSTS 2) --
# the hard ceiling on how many cells away any hopping can reach in a
# periodic direction, independent of the number of domain divisions used.
_KITE_NGHOSTS = 2


@dataclass
class LatticeFile:
    vectors: np.ndarray       # shape (3, 3): rows are a1, a2, a3
    atom_labels: List[str]
    atom_positions: np.ndarray  # shape (n_atoms, 3)
    fermi_level: float


def parse_lat_file(path: str) -> LatticeFile:
    """Parse an RS-LMTO `_lat.dat` file (see module docstring for format)."""
    with open(path) as f:
        text = f.read()

    def _block(name: str) -> List[str]:
        match = re.search(rf"begin {name}\n(.*?)\nend {name}", text, re.S)
        if not match:
            raise ValueError(f"{path}: missing 'begin {name}' / 'end {name}' block")
        lines = [line.strip() for line in match.group(1).splitlines()]
        return [line for line in lines if line]

    vectors = np.array(
        [[float(x) for x in line.split(",")] for line in _block("unit_cell")]
    )
    if vectors.shape != (3, 3):
        raise ValueError(f"{path}: expected 3 lattice vectors of 3 components each, "
                          f"got shape {vectors.shape}")

    labels: List[str] = []
    positions: List[List[float]] = []
    for line in _block("position"):
        label, *coords = (tok.strip() for tok in line.split(","))
        if len(coords) != 3:
            raise ValueError(f"{path}: position line {line!r} does not have 3 coordinates")
        labels.append(label)
        positions.append([float(x) for x in coords])

    fermi_lines = _block("fermi")
    if len(fermi_lines) != 1:
        raise ValueError(f"{path}: expected exactly one value in the 'fermi' block, "
                          f"got {fermi_lines!r}")

    return LatticeFile(
        vectors=vectors,
        atom_labels=labels,
        atom_positions=np.array(positions),
        fermi_level=float(fermi_lines[0]),
    )


def parse_hr_file(path: str) -> Dict[Tuple[RVec, int, int], complex]:
    """Parse a header-less Wannier90-style `_hr.dat` file.

    Returns {(R, i, j): H_ij(R)}, i/j 1-indexed exactly as stored in the
    file (NOT remapped to per-atom indices -- that happens in
    build_lattice()). R is always the full 3-component (R1, R2, R3) as
    stored; build_lattice() decides which components are actually periodic."""
    data: Dict[Tuple[RVec, int, int], complex] = {}
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 7:
                raise ValueError(f"{path}:{line_no}: expected 7 fields "
                                  f"(R1 R2 R3 i j Re Im), got {len(parts)}: {line!r}")
            r1, r2, r3, i, j = (int(parts[k]) for k in range(5))
            data[((r1, r2, r3), i, j)] = complex(float(parts[5]), float(parts[6]))
    return data


def find_material_files(material_dir: str) -> Tuple[str, str]:
    """Locate the `*_hr.dat`/`*_lat.dat` pair inside a material folder.

    Matches by suffix only (not by requiring the prefix to equal the
    folder's own name), so `rs_lmto/mos2/mos2_hr.dat` and
    `rs_lmto/tini/tini_hr.dat` are both found the same way. Raises
    ValueError (rather than silently picking one) if zero or more than one
    candidate is found for either suffix."""
    def _one(pattern: str) -> str:
        matches = sorted(glob.glob(os.path.join(material_dir, pattern)))
        if len(matches) != 1:
            raise ValueError(
                f"{material_dir}: expected exactly one file matching "
                f"'{pattern}', found {len(matches)}: {matches}"
            )
        return matches[0]

    return _one("*_hr.dat"), _one("*_lat.dat")


def load_material(material_dir: str, orbitals_per_atom: int,
                   spin_labels: Sequence[str] = ("u", "d"),
                   value_threshold: float = 1e-8) -> Tuple[latt.Lattice, dict]:
    """Find, parse, and build the KITE Lattice for a material folder in one
    call -- the folder must contain exactly one `*_hr.dat` and one
    `*_lat.dat` (any prefix). See build_lattice() for the other parameters."""
    hr_path, lat_path = find_material_files(material_dir)
    lat = parse_lat_file(lat_path)
    hr = parse_hr_file(hr_path)
    return build_lattice(lat, hr, orbitals_per_atom=orbitals_per_atom,
                          spin_labels=spin_labels, value_threshold=value_threshold)


def _orbital_names(atom_labels: Sequence[str], orbitals_per_atom: int,
                    spin_labels: Sequence[str]) -> Tuple[List[str], List[int]]:
    """1-indexed _hr.dat orbital position -> (sublattice name, atom index).

    Assumes contiguous per-atom blocks in the same order as atom_labels,
    spin-major within each atom (all of spin_labels[0]'s spatial orbitals,
    then all of spin_labels[1]'s, ...) -- confirmed against this specific
    RS-LMTO output's convention (9 spd orbitals x 2 spins = 18/atom,
    spin-major) before writing this module; verify against your own code's
    convention if it differs."""
    if orbitals_per_atom % len(spin_labels) != 0:
        raise ValueError(
            f"orbitals_per_atom ({orbitals_per_atom}) is not divisible by "
            f"len(spin_labels) ({len(spin_labels)})"
        )
    n_spatial = orbitals_per_atom // len(spin_labels)

    names: List[str] = []
    atom_of: List[int] = []
    for atom_idx, atom_label in enumerate(atom_labels):
        for spin in spin_labels:
            for orb in range(n_spatial):
                names.append(f"{atom_label}_{spin}{orb}")
                atom_of.append(atom_idx)
    return names, atom_of


def _is_positive(r: RVec) -> bool:
    """Lexicographic sign of a (possibly reduced-dimension) lattice vector:
    True for the 'positive' half of each +-R pair (first nonzero component
    > 0), False for the 'negative' half and for an all-zero R."""
    for component in r:
        if component > 0:
            return True
        if component < 0:
            return False
    return False


def _active_dims(hr: Dict[Tuple[RVec, int, int], complex]) -> List[int]:
    """Which of the 3 stored R-axes are ever nonzero across the whole
    dataset -- the genuinely periodic directions. An axis that is 0 in
    every row is either non-periodic slab padding (see module docstring)
    or, in principle, a real periodic direction whose cutoff never reached
    a neighboring cell; this module treats "always 0" as "not periodic" in
    either case, since a direction with no stored R != 0 hopping behaves
    identically to a non-periodic one regardless of the reason."""
    active = [False, False, False]
    for (r, _i, _j) in hr:
        for axis in range(3):
            if r[axis] != 0:
                active[axis] = True
        if all(active):
            break
    return [axis for axis in range(3) if active[axis]]


def build_lattice(lat: LatticeFile, hr: Dict[Tuple[RVec, int, int], complex],
                   orbitals_per_atom: int, spin_labels: Sequence[str] = ("u", "d"),
                   value_threshold: float = 1e-8) -> Tuple[latt.Lattice, dict]:
    """Build a kite.lattice.Lattice from parsed RS-LMTO data.

    orbitals_per_atom must be the same for every atom (contiguous blocks,
    same order as lat.atom_labels) -- mixed per-atom basis sizes are not
    supported by this simple mapping.

    value_threshold: hopping/onsite magnitudes below this are dropped. The
    RS-LMTO real-space cutoff sphere stores many explicit exact (or
    near-numerical-noise) zeros; dropping them keeps the KITE lattice from
    carrying dead weight. Set to 0.0 to keep everything.

    Returns (lattice, stats) where stats is a small dict of counts (kept
    onsite terms, kept hoppings, dropped-by-threshold count, etc.) so the
    caller can sanity-check the import rather than trust it silently.
    """
    n_atoms = len(lat.atom_labels)
    n_orb_total = n_atoms * orbitals_per_atom
    max_index = max(i for (_r, i, _j) in hr)
    if max_index != n_orb_total:
        raise ValueError(
            f"_hr.dat's highest orbital index is {max_index}, but "
            f"{n_atoms} atoms x {orbitals_per_atom} orbitals/atom = {n_orb_total} "
            f"were expected -- orbitals_per_atom is likely wrong."
        )

    names, atom_of = _orbital_names(lat.atom_labels, orbitals_per_atom, spin_labels)

    # KITE's C++ engine hardcodes NGHOSTS=2 (Src/Generic.hpp) -- the ghost-cell
    # width used for periodic wraparound and inter-domain communication in
    # EVERY periodic direction, regardless of how many divisions are used.
    # A hopping reaching further than that is not just slow, it is silently
    # WRONG (the ghost region doesn't hold the needed neighbor-cell data) --
    # nothing else in the pipeline checks this, so fail closed here instead.
    max_range = max((abs(component) for (r, _i, _j) in hr for component in r),
                     default=0)
    if max_range > _KITE_NGHOSTS:
        raise ValueError(
            f"_hr.dat has a hopping reaching {max_range} cells away in some "
            f"direction, but KITE's compiled NGHOSTS={_KITE_NGHOSTS} only "
            f"supports hoppings up to {_KITE_NGHOSTS} cells away in any "
            f"periodic direction -- results would be silently wrong, not just "
            f"slow. Recompile KITE with a larger NGHOSTS (Src/Generic.hpp) "
            f"before using this dataset."
        )

    dims = _active_dims(hr)
    if not dims:
        raise ValueError("_hr.dat has no nonzero R component at all -- "
                          "every stored term is on-site (R=0); nothing periodic to build.")
    periodic_vectors = np.array([lat.vectors[axis] for axis in dims])  # (n_periodic, 3)

    # Drop any Cartesian coordinate that is exactly 0 for EVERY periodic
    # vector (e.g. a slab's out-of-plane z, once the non-periodic axis
    # itself was already dropped above by _active_dims). This is a
    # dimensionality reduction of the LATTICE VECTORS ONLY -- atom
    # positions keep all 3 components regardless, exactly the buckled-
    # monolayer situation kite.visualize.hamiltonian_k already supports
    # (it zero-pads R_cartesian back up to match position dimensionality).
    # Doing this matters beyond tidiness: Lattice.reciprocal_vectors() /
    # brillouin_zone() feed these vectors to scipy's Voronoi, which raises
    # a "coplanar" QhullError on a genuinely 2D point set embedded with a
    # redundant all-zero 3rd coordinate.
    used_coords = [c for c in range(periodic_vectors.shape[1])
                   if not np.allclose(periodic_vectors[:, c], 0.0)]
    lattice_vectors = [list(periodic_vectors[k, used_coords])
                       for k in range(len(dims))]
    lattice = latt.Lattice(*lattice_vectors)

    onsite = [0.0] * n_orb_total
    onsite_kept = 0
    for (r, i, j), value in hr.items():
        if r != (0, 0, 0) or i != j:
            continue
        if abs(value.imag) > value_threshold:
            raise ValueError(
                f"Onsite term (orbital {i}, R=0) has non-negligible imaginary "
                f"part {value.imag!r} -- onsite energies must be real."
            )
        onsite[i - 1] = value.real
        onsite_kept += 1

    sublattices = [
        (names[k], list(lat.atom_positions[atom_of[k]]), onsite[k])
        for k in range(n_orb_total)
    ]
    lattice.add_sublattices(*sublattices)

    hoppings = []
    dropped_small = 0
    for (r, i, j), value in hr.items():
        r_active = tuple(r[axis] for axis in dims)
        if r == (0, 0, 0):
            if i >= j:
                continue  # onsite (i == j) or lower triangle (auto-mirrored from i < j)
        elif not _is_positive(r_active):
            continue  # this is the auto-mirrored conjugate of the +R, (j, i) term
        if abs(value) < value_threshold:
            dropped_small += 1
            continue
        hoppings.append((list(r_active), names[i - 1], names[j - 1], value))

    lattice.add_hoppings(*hoppings)

    stats = {
        "n_orbitals": n_orb_total,
        "periodic_dims": dims,
        "onsite_terms_kept": onsite_kept,
        "hoppings_kept": len(hoppings),
        "dropped_below_threshold": dropped_small,
        "fermi_level": lat.fermi_level,
    }
    return lattice, stats


# ---------------------------------------------------------------------------
# Standard on-site operators: orbital angular momentum L, spin S, and the
# orbital quadrupole moments built from L. Fixed for every material this
# importer handles, since the 9-orbital spd spatial basis and its ordering
# (see _SPD_BASIS below) is the same convention for all of them -- this is
# NOT re-derived per material.
#
# Three consumers share these same embedded operators:
#   1. kite.Calculation.ldos()/ldos_map(operators=[...]) -- site-resolved
#      spin/orbital texture (register_standard_operators() below).
#   2. kite.Calculation.custom_one()/custom_two() -- future spin/orbital
#      Hall conductivity (SHC/OHC), built the same way this session's
#      kane_mele_spin_hall.py/rashba_edelstein_graphene.py examples build
#      their Sz-weighted velocity-velocity vertex, just swapping in the
#      Sz/Lz labels registered here. Not implemented in this module --
#      the registration is the shared piece; the Kubo-formula vertex
#      construction is separate, future work.
#   3. band_projection() below -- eigenvector expectation values along a
#      k-path, i.e. spin/orbital-projected band structure (the same
#      quantity plotted in the reference <Sz>/<Lz>-colored band structures
#      this feature is meant to reproduce with KITE's own tools).
# ---------------------------------------------------------------------------

_SPD_BASIS = ("s", "px", "py", "pz", "dxy", "dyz", "dzx", "dx2-y2", "d3z2-r2")


def _spd_angular_momentum_matrices() -> Dict[str, np.ndarray]:
    """Lx, Ly, Lz (9x9, dimensionless L/hbar) in the _SPD_BASIS order.

    Transcribed from a reference derivation and verified numerically (not
    just trusted) before use: each matrix is Hermitian, and together they
    satisfy the angular momentum algebra [L_a, L_b] = i * epsilon_abc * L_c
    exactly (checked to np.allclose tolerance at write time)."""
    n = len(_SPD_BASIS)
    Lx = np.zeros((n, n), dtype=complex)
    Ly = np.zeros((n, n), dtype=complex)
    Lz = np.zeros((n, n), dtype=complex)

    def _pair(M, i, j, value):
        M[i, j] = value
        M[j, i] = np.conj(value)

    # index: 0=s, 1=px, 2=py, 3=pz, 4=dxy, 5=dyz, 6=dzx, 7=dx2-y2, 8=d3z2-r2
    _pair(Lx, 2, 3, -1j)                 # py, pz
    _pair(Lx, 4, 6, -1j)                 # dxy, dzx
    _pair(Lx, 5, 7, -1j)                 # dyz, dx2-y2
    _pair(Lx, 5, 8, -1j * np.sqrt(3))    # dyz, d3z2-r2

    _pair(Ly, 1, 3, 1j)                  # px, pz
    _pair(Ly, 4, 5, 1j)                  # dxy, dyz
    _pair(Ly, 6, 7, -1j)                 # dzx, dx2-y2
    _pair(Ly, 6, 8, 1j * np.sqrt(3))     # dzx, d3z2-r2

    _pair(Lz, 1, 2, -1j)                 # px, py
    _pair(Lz, 4, 7, 2j)                  # dxy, dx2-y2
    _pair(Lz, 5, 6, 1j)                  # dyz, dzx

    matrices = {"x": Lx, "y": Ly, "z": Lz}
    for name, M in matrices.items():
        if not np.allclose(M, M.conj().T):
            raise AssertionError(f"L{name} is not Hermitian -- transcription error")
    if not np.allclose(Lx @ Ly - Ly @ Lx, 1j * Lz):
        raise AssertionError("[Lx, Ly] != i*Lz -- transcription error")
    if not np.allclose(Ly @ Lz - Lz @ Ly, 1j * Lx):
        raise AssertionError("[Ly, Lz] != i*Lx -- transcription error")
    if not np.allclose(Lz @ Lx - Lx @ Lz, 1j * Ly):
        raise AssertionError("[Lz, Lx] != i*Ly -- transcription error")
    return matrices


def _spin_half_matrices() -> Dict[str, np.ndarray]:
    """Sx, Sy, Sz = Pauli/2 (2x2, dimensionless S/hbar), verified the same
    way as the L matrices (Hermiticity + [S_a, S_b] = i*epsilon_abc*S_c)."""
    sx = np.array([[0, 1], [1, 0]], dtype=complex) / 2
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2
    sz = np.array([[1, 0], [0, -1]], dtype=complex) / 2
    matrices = {"x": sx, "y": sy, "z": sz}
    for name, M in matrices.items():
        if not np.allclose(M, M.conj().T):
            raise AssertionError(f"S{name} is not Hermitian -- transcription error")
    if not np.allclose(sx @ sy - sy @ sx, 1j * sz):
        raise AssertionError("[Sx, Sy] != i*Sz -- transcription error")
    return matrices


def _spd_quadrupole_matrices() -> Dict[str, np.ndarray]:
    """The 5 rank-2 (quadrupole) tensor operators built from L, by analogy
    with the spin quadrupole moments: symmetrized products {La, Lb} for the
    off-diagonal components, and 3Lz^2 - L^2 for the diagonal one. Each is
    Hermitian by construction ({A,B} and A^2 are Hermitian whenever A, B
    are) -- checked anyway rather than assumed."""
    L = _spd_angular_momentum_matrices()
    Lx, Ly, Lz = L["x"], L["y"], L["z"]
    L2 = Lx @ Lx + Ly @ Ly + Lz @ Lz

    matrices = {
        "xy": Lx @ Ly + Ly @ Lx,
        "yz": Ly @ Lz + Lz @ Ly,
        "zx": Lz @ Lx + Lx @ Lz,
        "x2-y2": Lx @ Lx - Ly @ Ly,
        "3z2-r2": 3 * (Lz @ Lz) - L2,
    }
    for name, M in matrices.items():
        if not np.allclose(M, M.conj().T):
            raise AssertionError(f"quadrupole operator {name} is not Hermitian")
    return matrices


def atomic_L_matrices(spin_labels: Sequence[str] = ("u", "d")) -> Dict[str, np.ndarray]:
    """Lx, Ly, Lz for ONE atom's full spin x spd basis (18x18 for the default
    2 spins x 9 spd orbitals) -- i.e. embed_atomic_operator()'s per-atom
    block (kron(I_spin, L_9x9)) WITHOUT tiling across the lattice's other
    atoms, for callers that need the single-atom operator itself (e.g. as
    the vertex operator in a custom_one/custom_two orbital Hall Kubo-formula
    calculation, following the same Sz-weighted-vertex pattern already used
    for the spin Hall examples in this repo, kane_mele_spin_hall.py /
    rashba_edelstein_graphene.py -- just swapping Sz for Lz).

    Built from the same verified _spd_angular_momentum_matrices() used by
    standard_operators() (Hermiticity + angular-momentum algebra already
    checked there) -- not re-derived."""
    L = _spd_angular_momentum_matrices()
    n_spin = len(spin_labels)
    return {name: np.kron(np.eye(n_spin), M) for name, M in L.items()}


def embed_atomic_operator(op: np.ndarray, n_atoms: int, orbitals_per_atom: int,
                           spin_labels: Sequence[str], spatial: bool) -> np.ndarray:
    """Embed a single-atom operator into the full n_orb_total x n_orb_total
    system, block-diagonal across atoms (on-site operator: zero between
    different atoms, identical block repeated for every atom).

    op : (9, 9) if spatial=True (an orbital operator like Lx, identical for
    every spin -- embedded as kron(I_spin, op), matching this module's
    spin-major ordering, see _orbital_names()'s docstring), or (2, 2) if
    spatial=False (a spin operator like Sz, identical for every spatial
    orbital -- embedded as kron(op, I_orbital))."""
    n_spin = len(spin_labels)
    n_spatial = orbitals_per_atom // n_spin
    if spatial:
        if op.shape != (n_spatial, n_spatial):
            raise ValueError(f"expected a ({n_spatial}, {n_spatial}) spatial operator, "
                              f"got {op.shape}")
        atom_block = np.kron(np.eye(n_spin), op)
    else:
        if op.shape != (n_spin, n_spin):
            raise ValueError(f"expected a ({n_spin}, {n_spin}) spin operator, got {op.shape}")
        atom_block = np.kron(op, np.eye(n_spatial))

    n_orb_total = n_atoms * orbitals_per_atom
    full = np.zeros((n_orb_total, n_orb_total), dtype=complex)
    for atom in range(n_atoms):
        lo, hi = atom * orbitals_per_atom, (atom + 1) * orbitals_per_atom
        full[lo:hi, lo:hi] = atom_block
    return full


def standard_operators(n_atoms: int, orbitals_per_atom: int = 18,
                        spin_labels: Sequence[str] = ("u", "d")) -> Dict[str, np.ndarray]:
    """All standard operators (Lx, Ly, Lz, Sx, Sy, Sz, and the 5 orbital
    quadrupole moments Oxy/Oyz/Ozx/Ox2-y2/O3z2-r2), each embedded to the
    full (n_orb_total, n_orb_total) system via embed_atomic_operator().
    Requires orbitals_per_atom == 9 * len(spin_labels) (the spd-times-spin
    convention _SPD_BASIS/_spd_angular_momentum_matrices() are built for)."""
    n_spatial = len(_SPD_BASIS)
    if orbitals_per_atom != n_spatial * len(spin_labels):
        raise ValueError(
            f"standard_operators() assumes {n_spatial} spatial (spd) orbitals x "
            f"{len(spin_labels)} spins = {n_spatial * len(spin_labels)} per atom, "
            f"got orbitals_per_atom={orbitals_per_atom}"
        )

    L = _spd_angular_momentum_matrices()
    S = _spin_half_matrices()
    Q = _spd_quadrupole_matrices()

    out = {}
    for name, M in L.items():
        out[f"L{name}"] = embed_atomic_operator(M, n_atoms, orbitals_per_atom,
                                                 spin_labels, spatial=True)
    for name, M in S.items():
        out[f"S{name}"] = embed_atomic_operator(M, n_atoms, orbitals_per_atom,
                                                 spin_labels, spatial=False)
    for name, M in Q.items():
        out[f"O{name}"] = embed_atomic_operator(M, n_atoms, orbitals_per_atom,
                                                 spin_labels, spatial=True)
    return out


def register_operator(calculation, names: Sequence[str], matrix: np.ndarray,
                       label: str) -> None:
    """Register a dense (n, n) operator matrix as a KITE custom operator
    under `label` (must start with 'l', per Calculation.add_orbital_coupling's
    own requirement).

    All of `names` are (re-)registered via add_orbital_index first -- this
    MUST happen before any add_orbital_coupling call for this label, since
    KITE sizes a new label's internal array from however many orbital
    indices are registered at that moment (see add_orbital_coupling's
    implementation). Calling this multiple times with the same `names` is
    safe (add_orbital_index is a plain dict overwrite, idempotent)."""
    if not label or label[0] != "l":
        raise ValueError(f"label must start with 'l', got {label!r}")
    n = len(names)
    if matrix.shape != (n, n):
        raise ValueError(f"matrix shape {matrix.shape} does not match {n} orbital names")

    for idx, name in enumerate(names):
        calculation.add_orbital_index(name, idx)

    for row in range(n):
        for col in range(n):
            value = matrix[row, col]
            if value != 0:
                # add_orbital_coupling(start, last, c, label) sets
                # operator[row=last, col=start] = c -- see its own
                # docstring/lattice.py's add_one_hopping for the same
                # reversed-argument-order convention.
                calculation.add_orbital_coupling(names[col], names[row], value, label)


def register_standard_operators(calculation, names: Sequence[str], n_atoms: int,
                                 orbitals_per_atom: int = 18,
                                 spin_labels: Sequence[str] = ("u", "d"),
                                 which: Sequence[str] = None) -> Dict[str, str]:
    """Build (standard_operators()) and register (register_operator()) every
    requested standard operator in one call.

    which : names to register (e.g. ["Lz", "Sz"]); defaults to all of them.

    Returns {operator_name: kite_label}, e.g. {"Lz": "lLz", "Sz": "lSz"} --
    pass label values from this dict directly as ldos()/ldos_map()'s
    operators=[...] argument."""
    operators = standard_operators(n_atoms, orbitals_per_atom, spin_labels)
    if which is None:
        which = list(operators.keys())

    label_of = {}
    for name in which:
        label = f"l{name}"
        register_operator(calculation, names, operators[name], label)
        label_of[name] = label
    return label_of


def band_projection(lattice: latt.Lattice, k_path, operator_matrix: np.ndarray):
    """Eigenvector expectation value of `operator_matrix` for every band at
    every k-point of a path -- the spin/orbital-projected band structure
    (e.g. color bands by <Sz> or <Lz>, matching a spin/orbital-resolved
    ARPES-style plot), computed directly from kite.visualize.hamiltonian_k
    rather than duplicating its Bloch-Hamiltonian construction.

    Parameters
    ----------
    lattice : kite.lattice.Lattice
    k_path : array-like, shape (N, D), or the 3-tuple returned by
        kite.visualize.make_path
    operator_matrix : (lattice.nsub, lattice.nsub) array, Hermitian

    Returns
    -------
    bands : (N, nsub) real eigenvalues, ascending order (same convention as
        kite.visualize.compute_bands)
    expectation : (N, nsub) real expectation values of operator_matrix,
        matching the ordering of `bands`' eigenvectors
    """
    from kite import visualize

    if isinstance(k_path, tuple) and len(k_path) == 3:
        k_points, _, _ = k_path
    else:
        k_points = k_path
    k_points = np.asarray(k_points, dtype=float)

    n = lattice.nsub
    if operator_matrix.shape != (n, n):
        raise ValueError(f"operator_matrix shape {operator_matrix.shape} does not "
                          f"match lattice.nsub={n}")
    if not np.allclose(operator_matrix, operator_matrix.conj().T):
        raise ValueError("operator_matrix must be Hermitian")

    bands = np.empty((k_points.shape[0], n), dtype=float)
    expectation = np.empty((k_points.shape[0], n), dtype=float)
    for i, k in enumerate(k_points):
        H = visualize.hamiltonian_k(lattice, k)
        evals, evecs = np.linalg.eigh(H)
        bands[i] = evals
        expectation[i] = np.real(np.einsum("ib,ib->b", evecs.conj(), operator_matrix @ evecs))
    return bands, expectation
