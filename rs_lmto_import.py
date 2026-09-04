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
only needs to know which orbitals belong to each atom (contiguous blocks,
in the same atom order as the `_lat.dat` position list) to build one KITE
sublattice per orbital, all orbitals of one atom sharing that atom's
position. Atoms need NOT carry the same number of orbitals: KITE registers
one sublattice per orbital and never sees a per-atom count, so a basis
where sulfur has lost its deep 's' while molybdenum keeps all nine spd
orbitals exports exactly as well (see _normalize_basis()).

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
    file. The file's own order is SPIN-MAJOR OVER THE WHOLE SYSTEM (all
    atoms for spin up, then all atoms for spin down) -- NOT the atom-major
    order the rest of this module uses. Pass the result through
    reorder_spin_major_to_atom_major() before build_lattice() or
    drop_core_orbitals(); load_material() already does this for you.
    R is always the full 3-component (R1, R2, R3) as stored;
    build_lattice() decides which components are actually periodic."""
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
                   value_threshold: float = 1e-3,
                   max_hopping_range: int = None) -> Tuple[latt.Lattice, dict]:
    """Find, parse, and build the KITE Lattice for a material folder in one
    call -- the folder must contain exactly one `*_hr.dat` and one
    `*_lat.dat` (any prefix). See build_lattice() for the other parameters."""
    hr_path, lat_path = find_material_files(material_dir)
    lat = parse_lat_file(lat_path)
    hr = parse_hr_file(hr_path)
    # parse_hr_file() returns the file's own spin-major-over-the-whole-system
    # index order; everything below expects atom-major. See that function.
    hr = reorder_spin_major_to_atom_major(
        hr, n_atoms=len(lat.atom_labels), orbitals_per_atom=orbitals_per_atom,
        spin_labels=spin_labels)
    return build_lattice(lat, hr, orbitals_per_atom=orbitals_per_atom,
                          spin_labels=spin_labels, value_threshold=value_threshold,
                          max_hopping_range=max_hopping_range)


def _normalize_basis(n_atoms: int, orbitals_per_atom, spin_labels: Sequence[str]
                      ) -> List[List[int]]:
    """Canonical basis description: per atom, WHICH spatial orbitals it keeps.

    Atoms are not required to share a basis. `orbitals_per_atom` may be:

      - an int -- every atom has the same count, keeping spatial indices
        0..n_spatial-1 (the plain, undropped case);
      - a sequence of ints, one per atom -- total orbitals (spins included)
        for that atom, keeping its first count//n_spin spatial indices;
      - a sequence of index lists, one per atom -- exactly which spatial
        orbitals survive on that atom, in the ORIGINAL numbering (what the
        drop_*() functions return).

    The third form is the one that carries enough information for the L/S/
    quadrupole operators: after dropping, say, only sulfur's 's', the
    operators for that atom are the 9x9 spd matrices restricted to the rows
    and columns that remain, which a bare count cannot express.
    """
    n_spin = len(spin_labels)

    def _from_count(total: int) -> List[int]:
        if total % n_spin != 0:
            raise ValueError(
                f"orbitals per atom ({total}) is not divisible by "
                f"len(spin_labels) ({n_spin})"
            )
        return list(range(total // n_spin))

    if isinstance(orbitals_per_atom, (int, np.integer)):
        return [_from_count(int(orbitals_per_atom)) for _ in range(n_atoms)]

    spec = list(orbitals_per_atom)
    if len(spec) != n_atoms:
        raise ValueError(
            f"orbitals_per_atom has {len(spec)} entries but there are "
            f"{n_atoms} atoms"
        )
    basis: List[List[int]] = []
    for entry in spec:
        if isinstance(entry, (int, np.integer)):
            basis.append(_from_count(int(entry)))
        else:
            kept = [int(k) for k in entry]
            if sorted(kept) != kept or len(set(kept)) != len(kept):
                raise ValueError(
                    f"spatial orbital indices must be strictly increasing "
                    f"and unique, got {entry!r}"
                )
            basis.append(kept)
    return basis


def _basis_offsets(basis: Sequence[Sequence[int]], n_spin: int) -> List[int]:
    """Index of each atom's first orbital in the full system."""
    offsets, running = [], 0
    for kept in basis:
        offsets.append(running)
        running += len(kept) * n_spin
    return offsets


def _orbital_names(atom_labels: Sequence[str], orbitals_per_atom,
                    spin_labels: Sequence[str]) -> Tuple[List[str], List[int]]:
    """1-indexed orbital position -> (sublattice name, atom index).

    Contiguous per-atom blocks in the same order as atom_labels, spin-major
    within each atom (all of spin_labels[0]'s spatial orbitals, then all of
    spin_labels[1]'s, ...). This is the module's INTERNAL order; the file's
    own order is different, see reorder_spin_major_to_atom_major().

    Atoms may carry different bases -- see _normalize_basis(). Names use the
    ORIGINAL spatial index, so an orbital keeps its name when others around
    it are dropped (e.g. sulfur's px stays "S1_u1" even after "S1_u0" is
    removed), which keeps names stable across a drop_*() call.
    """
    basis = _normalize_basis(len(atom_labels), orbitals_per_atom, spin_labels)

    names: List[str] = []
    atom_of: List[int] = []
    for atom_idx, (atom_label, kept) in enumerate(zip(atom_labels, basis)):
        for spin in spin_labels:
            for orb in kept:
                names.append(f"{atom_label}_{spin}{orb}")
                atom_of.append(atom_idx)
    return names, atom_of


def reorder_spin_major_to_atom_major(
        hr: Dict[Tuple[RVec, int, int], complex], n_atoms: int,
        orbitals_per_atom: int, spin_labels: Sequence[str] = ("u", "d")
        ) -> Dict[Tuple[RVec, int, int], complex]:
    """Permute `_hr.dat` orbital indices from the FILE's order to this
    module's internal order.

    The RS-LMTO `_hr.dat` is written SPIN-MAJOR OVER THE WHOLE SYSTEM: all
    atoms' spatial orbitals for spin up first, then all atoms' again for
    spin down, i.e. file index (0-based)

        k_file = spin * (n_atoms * n_spatial) + atom * n_spatial + spatial

    Everything downstream in this module (_orbital_names,
    embed_atomic_operator, drop_core_orbitals) instead uses ATOM-MAJOR
    order, spin-major within each atom:

        k_internal = atom * orbitals_per_atom + spin * n_spatial + spatial

    Both describe the same Hamiltonian; the atom-major one is used on the
    KITE side because it keeps each atom's 2*n_spatial orbitals contiguous,
    which puts the large on-site and short-range blocks near the diagonal
    instead of spreading them across a half-matrix stride.

    Verified against the MoS2 dataset rather than assumed: the file's
    on-site diagonal repeats as (Mo, S, S | Mo, S, S) -- period 3 blocks of
    9, spin outermost -- and the difference between the two halves,
    H_upup - H_downdown, reproduces lambda*Lz in the _SPD_BASIS order to a
    relative residual of 1e-11, giving lambda_d(Mo) = 0.0748 eV (the known
    Mo 4d spin-orbit constant, which also fixes the file's energy unit as
    eV). Re-check this for any new dataset instead of assuming it.
    """
    n_spin = len(spin_labels)
    if orbitals_per_atom % n_spin != 0:
        raise ValueError(
            f"orbitals_per_atom ({orbitals_per_atom}) is not divisible by "
            f"len(spin_labels) ({n_spin})"
        )
    n_spatial = orbitals_per_atom // n_spin
    n_orb_total = n_atoms * orbitals_per_atom

    perm = {}
    for spin in range(n_spin):
        for atom in range(n_atoms):
            for spatial in range(n_spatial):
                k_file = spin * (n_atoms * n_spatial) + atom * n_spatial + spatial
                k_internal = atom * orbitals_per_atom + spin * n_spatial + spatial
                perm[k_file + 1] = k_internal + 1

    max_index = max(max(i, j) for (_r, i, j) in hr)
    if max_index != n_orb_total:
        raise ValueError(
            f"_hr.dat's highest orbital index is {max_index}, but "
            f"{n_atoms} atoms x {orbitals_per_atom} orbitals/atom = "
            f"{n_orb_total} were expected -- orbitals_per_atom is likely wrong."
        )

    return {(r, perm[i], perm[j]): value for (r, i, j), value in hr.items()}


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


def drop_core_orbitals(hr: Dict[Tuple[RVec, int, int], complex],
                        atom_labels: Sequence[str], orbitals_per_atom: int,
                        spin_labels: Sequence[str], energy_cutoff: float
                        ) -> Tuple[Dict[Tuple[RVec, int, int], complex],
                                   List[List[int]], dict]:
    """Drop deep, core-like orbitals (onsite energy < energy_cutoff) from the
    RAW parsed hr dict, BEFORE build_lattice() -- e.g. "removed orbitals with
    energy much much lower than 0... removing the core that does not matter
    for our calculation", the same kind of trim done previously on PAOFLOW
    DFT data for another material.

    Uniform per atom TYPE (label): every atom sharing a label is required to
    agree, PER SPIN CHANNEL, on that spin's onsite energy for each spatial
    orbital -- this preserves the "every atom of this label has the SAME
    orbitals, same order" assumption the rest of this module
    (_orbital_names, embed_atomic_operator, standard_operators) and
    mos2_vacancies.py's per-atom-type vacancy schemes both rely on. Raises
    ValueError if that agreement doesn't hold, rather than silently picking
    one atom's answer.

    The two spin channels are NOT required to agree with each other, so
    that a genuinely spin-polarised dataset still works. Note that in the
    MoS2 data they DO agree exactly: the on-site diagonal is identical for
    both spins, and the whole up/down difference is the off-diagonal
    lambda*Lz of spin-orbit coupling. (An earlier version of this docstring
    cited "Mo's spatial index 0 has onsite u=+6.08 eV, d=-14.01 eV" as
    evidence of a real 20 eV spin splitting -- that was an artifact of
    reading the file in the wrong orbital order, not physics. See
    reorder_spin_major_to_atom_major().) A spatial orbital is dropped only
    when BOTH spin channels fall below
    energy_cutoff (the conservative choice: keeps an orbital if either spin
    is still active near the energies of interest), never based on either
    spin alone.

    Atom labels may end up with DIFFERENT numbers of surviving orbitals --
    a cutoff that removes sulfur's deep 's' while leaving molybdenum
    untouched is a normal, supported outcome, not an error. The result is a
    per-atom basis and everything downstream takes it: build_lattice()
    registers one sublattice per orbital regardless, and the L/S/quadrupole
    operators restrict themselves to whatever each atom still has.

    Returns (hr_new, new_basis, dropped_info). hr_new has orbital indices
    REMAPPED to a contiguous 1-indexed scheme (same atom order, same
    spin-major order within each atom, just fewer spatial orbitals).
    new_basis is a per-atom list of the surviving spatial indices, to pass
    straight to build_lattice(..., orbitals_per_atom=new_basis, ...) and to
    standard_operators(). dropped_info maps each atom label to the
    list of (spatial_index, onsite_energy) actually dropped, so the caller
    can verify nothing important got cut before trusting it.
    """
    n_spin = len(spin_labels)
    n_spatial = orbitals_per_atom // n_spin
    names, atom_of = _orbital_names(atom_labels, orbitals_per_atom, spin_labels)

    onsite = {}
    for (r, i, j), value in hr.items():
        if r == (0, 0, 0) and i == j:
            onsite[i] = value.real

    label_of_atom = list(atom_labels)
    atoms_by_label: Dict[str, List[int]] = {}
    for atom_idx, label in enumerate(label_of_atom):
        atoms_by_label.setdefault(label, []).append(atom_idx)

    keep_spatial: Dict[str, List[int]] = {}
    dropped_info: Dict[str, List[Tuple[int, Tuple[float, ...]]]] = {}
    for label, atom_indices in atoms_by_label.items():
        keep: List[int] = []
        dropped: List[Tuple[int, Tuple[float, ...]]] = []
        for spatial in range(n_spatial):
            spin_energies = []
            for spin_idx in range(n_spin):
                energies = [
                    onsite.get(atom_idx * orbitals_per_atom + spin_idx * n_spatial + spatial + 1, 0.0)
                    for atom_idx in atom_indices
                ]
                if not np.allclose(energies, energies[0], atol=1e-6):
                    raise ValueError(
                        f"onsite energy for spatial orbital index {spatial}, spin "
                        f"{spin_labels[spin_idx]!r}, of atom label {label!r} is not "
                        f"the same across every atom instance of that label "
                        f"({energies}) -- cannot uniformly decide whether to drop it."
                    )
                spin_energies.append(energies[0])
            if all(e < energy_cutoff for e in spin_energies):
                dropped.append((spatial, tuple(spin_energies)))
            else:
                keep.append(spatial)
        keep_spatial[label] = keep
        dropped_info[label] = dropped

    # Different atom labels may end up with different numbers of surviving
    # orbitals -- that is allowed. The result is a per-atom basis, which
    # build_lattice() and the operator machinery both accept.
    new_basis = [list(keep_spatial[label]) for label in label_of_atom]

    # Build the remap: old 1-indexed orbital -> new 1-indexed orbital,
    # contiguous, same atom order, same spin-major order, fewer spatial slots.
    remap = {}
    new_k = 0
    for atom_idx, label in enumerate(label_of_atom):
        keep = keep_spatial[label]
        for spin_idx in range(n_spin):
            for spatial in keep:
                old_k = atom_idx * orbitals_per_atom + spin_idx * n_spatial + spatial + 1
                new_k += 1
                remap[old_k] = new_k

    hr_new = {}
    for (r, i, j), value in hr.items():
        if i in remap and j in remap:
            hr_new[(r, remap[i], remap[j])] = value

    return hr_new, new_basis, dropped_info


def drop_named_orbitals(hr: Dict[Tuple[RVec, int, int], complex],
                         atom_labels: Sequence[str], orbitals_per_atom: int,
                         spin_labels: Sequence[str], names_to_drop: Sequence[str]
                         ) -> Tuple[Dict[Tuple[RVec, int, int], complex], List[str]]:
    """Remove specific orbitals by name from the RAW parsed hr dict, BEFORE
    building a lattice -- unlike drop_core_orbitals() (a uniform energy
    cutoff, same count dropped per atom TYPE), this allows an ASYMMETRIC
    removal (e.g. drop an orbital from only one atom, not every atom
    sharing its label), for cases where a specific symmetry/physics
    argument -- not a blanket cutoff -- picks out a particular orbital.

    Concretely: verified (by comparing full vs reduced band structures
    directly along a k-path, not by assumption) that removing Mo's own
    s-orbital (both spin channels) leaves the rest of the spectrum
    essentially unchanged -- it only removes the 2 (of 4) deep bands that
    actually involve Mo, while the other 2 (an S1/S2 mirror-pair
    combination not involving Mo at all) and everything else survive
    intact to a few hundredths of an eV.

    names_to_drop: orbital names in the f"{atom_label}_{spin}{spatial}"
    convention _orbital_names() itself uses (e.g. "Mo_u0", "Mo_d0").

    Returns (hr_new, remaining_names, new_basis).

    hr_new has orbital indices REMAPPED to a contiguous 1-indexed scheme
    (dropped names simply skipped, relative order otherwise unchanged), and
    remaining_names is the ordered orbital-name list matching those indices.

    new_basis is the per-atom list of surviving spatial indices, ready for
    build_lattice(..., orbitals_per_atom=new_basis, ...) -- atoms carrying
    different bases is fine there.

    The removal must be SPIN-SYMMETRIC: dropping a spatial orbital for one
    spin but keeping it for the other (e.g. "Mo_u0" without "Mo_d0") raises
    ValueError. A spin-asymmetric basis has no physical meaning here -- the
    two channels describe the same spatial orbital -- and it would break the
    kron(I_spin, op) structure every operator in this module is built on.
    """
    names, atom_of = _orbital_names(atom_labels, orbitals_per_atom, spin_labels)
    drop_set = set(names_to_drop)
    unknown = drop_set - set(names)
    if unknown:
        raise ValueError(f"names_to_drop contains unknown orbital name(s): {unknown}")

    n_spin = len(spin_labels)
    remap = {}
    remaining_names: List[str] = []
    # per atom, per spin: which spatial indices survive
    kept: List[Dict[str, List[int]]] = [
        {spin: [] for spin in spin_labels} for _ in atom_labels
    ]
    new_k = 0
    for old_k, name in enumerate(names, start=1):
        if name in drop_set:
            continue
        new_k += 1
        remap[old_k] = new_k
        remaining_names.append(name)
        spin_and_orb = name.split("_")[-1]
        for spin in spin_labels:
            if spin_and_orb.startswith(spin):
                kept[atom_of[old_k - 1]][spin].append(int(spin_and_orb[len(spin):]))
                break

    # The removal must be spin-symmetric: both channels of an atom describe
    # the same spatial orbital, so they have to survive or go together.
    new_basis: List[List[int]] = []
    for atom_idx, per_spin in enumerate(kept):
        channels = [sorted(per_spin[spin]) for spin in spin_labels]
        if any(c != channels[0] for c in channels[1:]):
            asymmetric = sorted(set().union(*map(set, channels))
                                 - set.intersection(*map(set, channels)))
            raise ValueError(
                f"removal is not spin-symmetric on atom "
                f"{atom_labels[atom_idx]!r}: spatial orbital(s) {asymmetric} "
                f"survive in some spin channels but not others "
                f"({dict(zip(spin_labels, channels))}). Drop every spin "
                f"channel of an orbital or none of them."
            )
        new_basis.append(channels[0])

    hr_new = {}
    for (r, i, j), value in hr.items():
        if i in remap and j in remap:
            hr_new[(r, remap[i], remap[j])] = value

    return hr_new, remaining_names, new_basis


def build_lattice(lat: LatticeFile, hr: Dict[Tuple[RVec, int, int], complex],
                   orbitals_per_atom: int, spin_labels: Sequence[str] = ("u", "d"),
                   value_threshold: float = 1e-3,
                   max_hopping_range: int = None) -> Tuple[latt.Lattice, dict]:
    """Build a kite.lattice.Lattice from parsed RS-LMTO data.

    orbitals_per_atom describes the basis: an int when every atom carries
    the same one, or a per-atom sequence when they differ (see
    _normalize_basis() for the accepted forms). Atoms are NOT required to
    agree -- KITE registers one sublattice per orbital and never sees a
    per-atom count, so a lattice where, say, sulfur has lost its 's' while
    molybdenum keeps all nine spd orbitals exports exactly as well. The
    blocks stay contiguous and in lat.atom_labels order either way.

    value_threshold: hoppings smaller than this FRACTION of the largest
    hopping in the dataset are dropped (default 1e-3). The cut is relative,
    never absolute, for two reasons: it is invariant under the file's energy
    unit, and an absolute cut calibrated on one material silently destroys
    the physics of another -- in a twisted bilayer the interlayer hoppings
    are tiny in absolute terms and are exactly what the calculation is
    about. The reference is the largest HOPPING, not the largest matrix
    element: on-site energies are typically far larger and are never
    subject to this cut. Set to 0.0 to keep everything.

    Beyond tidiness, this matters because the RS-LMTO real-space cutoff
    sphere stores many explicit exact (or numerical-noise) zeros, which
    would otherwise be carried into the lattice as dead weight. stats
    reports both the absolute threshold this worked out to and the largest
    magnitude actually discarded, so the cut can be checked against the
    energy scales of the problem rather than trusted.

    max_hopping_range: opt-in truncation, default None (current behavior:
    fail closed below if any hopping reaches further than KITE's compiled
    NGHOSTS). When set to an integer (e.g. 2, matching NGHOSTS), every
    hopping with |R| > max_hopping_range in any direction is dropped BEFORE
    the NGHOSTS check runs, rather than raising. Dropping real hopping data
    changes the physics, if only slightly, so this reports exactly what was
    thrown away (count and largest dropped magnitude, in stats below) --
    verify that against the energy scale you actually care about (e.g. by
    comparing a band structure with and without this truncation) before
    trusting results computed with it. Setting this does NOT bypass the
    NGHOSTS check itself: if max_hopping_range is still larger than NGHOSTS,
    the same fail-closed error fires afterward, unchanged.
    """
    n_atoms = len(lat.atom_labels)
    basis = _normalize_basis(n_atoms, orbitals_per_atom, spin_labels)
    per_atom = [len(kept) * len(spin_labels) for kept in basis]
    n_orb_total = sum(per_atom)
    max_index = max(max(i, j) for (_r, i, j) in hr)
    if max_index != n_orb_total:
        raise ValueError(
            f"_hr.dat's highest orbital index is {max_index}, but the basis "
            f"({', '.join(f'{lab}:{n}' for lab, n in zip(lat.atom_labels, per_atom))}) "
            f"totals {n_orb_total} orbitals -- orbitals_per_atom is likely wrong."
        )

    names, atom_of = _orbital_names(lat.atom_labels, orbitals_per_atom, spin_labels)

    dropped_range = 0
    dropped_range_max_magnitude = 0.0
    if max_hopping_range is not None:
        kept_hr = {}
        for key, value in hr.items():
            r, _i, _j = key
            if max(abs(c) for c in r) > max_hopping_range:
                dropped_range += 1
                dropped_range_max_magnitude = max(dropped_range_max_magnitude, abs(value))
            else:
                kept_hr[key] = value
        hr = kept_hr

    # KITE's C++ engine hardcodes NGHOSTS=2 (Src/Generic.hpp) -- the ghost-cell
    # width used for periodic wraparound and inter-domain communication in
    # EVERY periodic direction, regardless of how many divisions are used.
    # A hopping reaching further than that is not just slow, it is silently
    # WRONG (the ghost region doesn't hold the needed neighbor-cell data) --
    # nothing else in the pipeline checks this, so fail closed here instead
    # (even after an opt-in max_hopping_range truncation above, in case that
    # value itself was still set larger than NGHOSTS).
    max_range = max((abs(component) for (r, _i, _j) in hr for component in r),
                     default=0)
    if max_range > _KITE_NGHOSTS:
        raise ValueError(
            f"_hr.dat has a hopping reaching {max_range} cells away in some "
            f"direction, but KITE's compiled NGHOSTS={_KITE_NGHOSTS} only "
            f"supports hoppings up to {_KITE_NGHOSTS} cells away in any "
            f"periodic direction -- results would be silently wrong, not just "
            f"slow. Recompile KITE with a larger NGHOSTS (Src/Generic.hpp), or "
            f"pass max_hopping_range<={_KITE_NGHOSTS} to explicitly truncate, "
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
        # Fixed tolerance, deliberately NOT value_threshold: that one is a
        # relative cut on hopping magnitudes, while this is a numerical-noise
        # check on a quantity that must be exactly real.
        if abs(value.imag) > 1e-8:
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

    # The threshold is RELATIVE to the largest hopping in this dataset, so it
    # is invariant under the file's energy unit and means the same thing for
    # every material -- an absolute cut in eV would silently change meaning
    # on a file written in Ry, or on a material with a different bandwidth.
    # The reference is the largest HOPPING, not the largest matrix element:
    # on-site energies are typically much larger and are not scaled by this.
    largest_hopping = max(
        (abs(value) for (r, i, j), value in hr.items()
         if not (r == (0, 0, 0) and i == j)),
        default=0.0,
    )
    absolute_threshold = value_threshold * largest_hopping

    hoppings = []
    dropped_small = 0
    dropped_small_max_magnitude = 0.0
    for (r, i, j), value in hr.items():
        r_active = tuple(r[axis] for axis in dims)
        if r == (0, 0, 0):
            if i >= j:
                continue  # onsite (i == j) or lower triangle (auto-mirrored from i < j)
        elif not _is_positive(r_active):
            continue  # this is the auto-mirrored conjugate of the +R, (j, i) term
        if abs(value) < absolute_threshold:
            dropped_small += 1
            dropped_small_max_magnitude = max(dropped_small_max_magnitude, abs(value))
            continue
        hoppings.append((list(r_active), names[i - 1], names[j - 1], value))

    lattice.add_hoppings(*hoppings)

    stats = {
        "n_orbitals": n_orb_total,
        "periodic_dims": dims,
        "onsite_terms_kept": onsite_kept,
        "hoppings_kept": len(hoppings),
        "dropped_below_threshold": dropped_small,
        "dropped_below_threshold_max_magnitude": dropped_small_max_magnitude,
        "largest_hopping": largest_hopping,
        "value_threshold_relative": value_threshold,
        "value_threshold_absolute": absolute_threshold,
        "dropped_by_max_hopping_range": dropped_range,
        "dropped_by_max_hopping_range_max_magnitude": dropped_range_max_magnitude,
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


def embed_atomic_operator(op: np.ndarray, n_atoms: int, orbitals_per_atom,
                           spin_labels: Sequence[str], spatial: bool) -> np.ndarray:
    """Embed a single-atom operator into the full n_orb_total x n_orb_total
    system, block-diagonal across atoms (an on-site operator: zero between
    different atoms).

    op : for spatial=True, the operator in the FULL spatial basis (9x9 for
    spd) -- an orbital operator like Lx, identical for every spin, embedded
    as kron(I_spin, op) within each atom's block. For spatial=False, a
    (n_spin, n_spin) spin operator like Sz, identical for every spatial
    orbital, embedded as kron(op, I_orbital).

    Atoms may carry different bases. When an atom keeps only some of the
    spatial orbitals, `op` is RESTRICTED to the rows and columns that atom
    still has, and it is that restriction which is placed in its block. So
    dropping an orbital simply removes its row and column from the operator
    -- correct for L and the quadrupoles, whose matrix elements between the
    surviving orbitals do not depend on which others were removed.

    Note this is a truncation, not a downfolding: matrix elements that ran
    THROUGH a dropped orbital are gone, not folded into the rest.
    """
    n_spin = len(spin_labels)
    basis = _normalize_basis(n_atoms, orbitals_per_atom, spin_labels)
    offsets = _basis_offsets(basis, n_spin)
    n_orb_total = sum(len(kept) * n_spin for kept in basis)

    if not spatial and op.shape != (n_spin, n_spin):
        raise ValueError(f"expected a ({n_spin}, {n_spin}) spin operator, got {op.shape}")
    if spatial:
        widest = max(max(kept) for kept in basis if kept) + 1
        if op.ndim != 2 or op.shape[0] != op.shape[1] or op.shape[0] < widest:
            raise ValueError(
                f"expected a square spatial operator of size at least {widest} "
                f"(the highest spatial index any atom keeps), got {op.shape}"
            )

    full = np.zeros((n_orb_total, n_orb_total), dtype=complex)
    for atom, kept in enumerate(basis):
        if spatial:
            block = np.kron(np.eye(n_spin), op[np.ix_(kept, kept)])
        else:
            block = np.kron(op, np.eye(len(kept)))
        lo = offsets[atom]
        hi = lo + len(kept) * n_spin
        full[lo:hi, lo:hi] = block
    return full


def standard_operators(n_atoms: int, orbitals_per_atom=18,
                        spin_labels: Sequence[str] = ("u", "d")) -> Dict[str, np.ndarray]:
    """All standard operators (Lx, Ly, Lz, Sx, Sy, Sz, and the 5 orbital
    quadrupole moments Oxy/Oyz/Ozx/Ox2-y2/O3z2-r2), each embedded to the
    full (n_orb_total, n_orb_total) system via embed_atomic_operator().

    Every atom's basis must be drawn from the 9-orbital spd set
    _SPD_BASIS is built for, but atoms need not keep the same subset of it
    -- each atom's operator block is the spd matrix restricted to the
    orbitals that atom still has."""
    n_spatial = len(_SPD_BASIS)
    basis = _normalize_basis(n_atoms, orbitals_per_atom, spin_labels)
    for atom, kept in enumerate(basis):
        if kept and max(kept) >= n_spatial:
            raise ValueError(
                f"atom {atom} keeps spatial orbital index {max(kept)}, but "
                f"standard_operators() is built for the {n_spatial}-orbital "
                f"spd basis {_SPD_BASIS} (indices 0..{n_spatial - 1})"
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
