""" MoS2 orbital Hall effect (OHE) via KITE's rank-two custom vertex machinery
(custom_two), the direct L_z analogue of examples/kane_mele_spin_hall.py's
S_z-weighted spin Hall calculation.

Vertex pair (matches docs/documentation/examples/custom_vertex_operators.md's
own "orbital Hall: {v_x,L_z}/2, v_y" row):
    A = (1/2){v_x, L_z} = [[0.5,"vx.l0"], [0.5,"l0.vx"]]
    B = v_y              = [[1.0,"vy"]]
NumVelocities = 2 (one from vx.l0/l0.vx, one from vy) -- same parity as spin
Hall, so kane_mele_spin_hall_process.py's Kubo-Bastin formula (adapted in
mos2_ohe_process.py) applies unchanged.

L_z structure (see rs_lmto_import.atomic_L_matrices()'s own docstring): the
physically meaningful object is an 18x18 matrix (2 spins x 9 spd orbitals),
IDENTICAL regardless of which atom (Mo, S1, S2) it sits on -- L_z depends only
on the local spd angular-momentum algebra, not atomic species. KITE's
custom_two needs this registered as ONE operator spanning the full 54-orbital
lattice (Mo+S1+S2); that full matrix is exactly the same 18x18 block repeated
block-diagonally at each atom -- built here via standard_operators() (which
does this tiling already) and cross-checked against atomic_L_matrices()
directly (see the assertion below) so that "one universal matrix, repeated"
is a verified fact, not just an assertion in a docstring.

Label "l0" (not "lLz"): a custom operator label used inside a custom_two
vertex string (e.g. "vx.l0") is parsed by KITEx as "l" + a PLAIN INTEGER
index (std::stoi) into its vector of registered operator matrices -- not an
arbitrary name. register_standard_operators()'s default f"l{name}" naming
("lLz") would be unparseable here (it's fine for ldos()/ldos_map()'s
operators=[...], which looks labels up in a Python dict, no stoi involved).
Since only one custom operator is needed, it is registered directly via
register_operator(..., "l0") -- exactly kane_mele_spin_hall.py's and
rashba_edelstein_graphene.py's convention (both warn: naming the sole
operator anything but "l0" is a C++-side out-of-bounds read).

Units: L_z here is dimensionless, in units of hbar -- the same convention as
kane_mele_spin_hall.py's s_z = (1/2)*sigma_z. Since both operators share that
unit, spin_hall()'s e/(4*pi)-type normalization carries over unchanged; no
separate OAM unit conversion is needed.

First-run defaults are deliberately cheap: custom_two (rank-two Kubo-Bastin)
is much more expensive than dos()/ldos_map(). 8x8 with divisions=(1,1) (8 is
exactly KITE's compiled TILE, so divisions must be 1 in each direction) and
moments=256 (must be even), CLEAN lattice (scheme=None, no disorder at all)
-- only pass scheme="Mo"/"S" once this first clean run is confirmed sensible.
"""
import sys

import numpy as np

import kite
from kite import custom
from rs_lmto_import import load_material, standard_operators, atomic_L_matrices, register_operator
from mos2_vacancies import mo_vacancy_disorder, sulfur_vacancy_disorder, _atom_orbital_names

LENGTH = (8, 8)
DIVISIONS = (1, 1)
MOMENTS = 256

CUSTOM_TWO_NUM_RANDOM = 1
CUSTOM_TWO_NUM_DISORDER = 1
CUSTOM_TWO_NUM_POINTS = 1000
CUSTOM_TWO_TEMPERATURE = 0.01

ATOM_ORDER = ("Mo", "S1", "S2")
ORBITALS_PER_ATOM = 18


def _orbital_names():
    """Full 54-name list in the exact registration order build_lattice() used
    (Mo_u0..Mo_d8, S1_u0..S1_d8, S2_u0..S2_d8) -- reusing mos2_vacancies'
    per-atom helper, not a new name-building path."""
    names = []
    for atom in ATOM_ORDER:
        names.extend(_atom_orbital_names(atom))
    return names


def build_Lz_operator():
    """Full 54x54 L_z: the same 18x18 atomic_L_matrices()["z"] block, tiled
    identically at each of the 3 atoms. Returns (names, Lz_full)."""
    names = _orbital_names()
    n_atoms = len(ATOM_ORDER)
    Lz_full = standard_operators(n_atoms=n_atoms, orbitals_per_atom=ORBITALS_PER_ATOM,
                                  spin_labels=("u", "d"))["Lz"]

    Lz_atomic = atomic_L_matrices(spin_labels=("u", "d"))["z"]
    for atom_idx in range(n_atoms):
        lo, hi = atom_idx * ORBITALS_PER_ATOM, (atom_idx + 1) * ORBITALS_PER_ATOM
        block = Lz_full[lo:hi, lo:hi]
        if not np.array_equal(block, Lz_atomic):
            raise AssertionError(
                f"standard_operators()['Lz']'s block for atom {ATOM_ORDER[atom_idx]!r} "
                f"does not exactly equal atomic_L_matrices()['z'] -- the 'one universal "
                f"18x18 matrix, tiled' invariant is broken."
            )
    off_diagonal = Lz_full.copy()
    for atom_idx in range(n_atoms):
        lo, hi = atom_idx * ORBITALS_PER_ATOM, (atom_idx + 1) * ORBITALS_PER_ATOM
        off_diagonal[lo:hi, lo:hi] = 0
    if np.any(off_diagonal != 0):
        raise AssertionError("standard_operators()['Lz'] has nonzero inter-atom "
                              "entries -- expected strictly block-diagonal.")

    return names, Lz_full


def run(scheme=None, concentration=0.05, output_file="mos2_ohe-output.h5"):
    lattice, stats = load_material("mos2", orbitals_per_atom=ORBITALS_PER_ATOM,
                                    spin_labels=("u", "d"))

    if scheme is None:
        disorder_structural = None
    elif scheme == "Mo":
        disorder_structural = mo_vacancy_disorder(lattice, concentration)
    elif scheme == "S":
        disorder_structural = sulfur_vacancy_disorder(lattice, LENGTH[0], LENGTH[1], concentration)
    else:
        raise ValueError(f"unknown scheme {scheme!r}, expected None, 'Mo', or 'S'")

    configuration = kite.Configuration(
        divisions=list(DIVISIONS), length=list(LENGTH),
        boundaries=["periodic", "periodic"], is_complex=True, precision=1,
    )
    calculation = kite.Calculation(configuration)

    names, Lz_full = build_Lz_operator()
    register_operator(calculation, names, Lz_full, "l0")

    A = custom.Vertex(MOMENTS, [[0.5, "vx.l0"], [0.5, "l0.vx"]])
    B = custom.Vertex(MOMENTS, [[1.0, "vy"]])
    calculation.custom_two(
        stream_=[A, B],
        num_random_=CUSTOM_TWO_NUM_RANDOM,
        num_disorder_=CUSTOM_TWO_NUM_DISORDER,
        num_points_=CUSTOM_TWO_NUM_POINTS,
        temperature_=CUSTOM_TWO_TEMPERATURE,
    )

    kite.config_system(lattice, configuration, calculation,
                        disorder_structural=disorder_structural, filename=output_file)
    return output_file


if __name__ == "__main__":
    scheme = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "None" else None
    concentration = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    out = f"mos2_ohe_{scheme or 'clean'}-output.h5"
    run(scheme, concentration, out)
    print(f"Wrote {out}")
