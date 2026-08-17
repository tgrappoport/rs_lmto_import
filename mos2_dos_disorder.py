""" MoS2 total DOS with vacancy disorder -- DOS only, no averaging over
disorder realizations (test settings: num_disorder=1, num_random=10).
scheme selects which sublattice is vacated: "Mo" or "S" (S1/S2, randomly
mixed per defect -- see mos2_vacancies.sulfur_vacancy_disorder).

Split out from mos2_disorder_scan.py per feedback: run each calculation
type as its own script, not bundled together, so a slow/bad step doesn't
waste time already spent on a fast one.

Also exposes the single-atom 18x18 (9 spd orbitals x 2 spins) orbital
angular momentum matrices Lx, Ly, Lz, for a later orbital Hall effect (OHE)
calculation: OHE's Kubo-formula vertex needs an Lz-weighted velocity
operator, the direct analogue of the Sz-weighted vertex already used for
spin Hall in this repo (kane_mele_spin_hall.py / rashba_edelstein_graphene.py).
Not wired into a KITE custom operator here -- dos() doesn't consume
operators, and any registered custom operator must span the full 54-orbital
lattice (Mo+S1+S2), not one atom's 18 -- so this stays a plain numpy array,
importable as ``from mos2_dos_disorder import L_MATRICES`` (or rebuilt via
rs_lmto_import.atomic_L_matrices()) by whatever script builds the actual
custom_one/custom_two OHE vertex over the full lattice (embed with
rs_lmto_import.embed_atomic_operator() / standard_operators() at that point).
"""
import sys

import numpy as np

import kite
from rs_lmto_import import load_material, atomic_L_matrices
from mos2_vacancies import mo_vacancy_disorder, sulfur_vacancy_disorder

LENGTH = (64, 64)
DIVISIONS = (2, 2)

DOS_NUM_POINTS = 1000
DOS_NUM_MOMENTS = 1000
DOS_NUM_RANDOM = 10
DOS_NUM_DISORDER = 1

# Single-atom 18x18 L matrices (spin u/d x 9 spd orbitals) -- see module
# docstring. L_MATRICES["z"] is the one relevant for OHE.
L_MATRICES = atomic_L_matrices(spin_labels=("u", "d"))


def run(scheme, concentration, output_file):
    lattice, stats = load_material("mos2", orbitals_per_atom=18, spin_labels=("u", "d"))

    if scheme == "Mo":
        disorder_structural = mo_vacancy_disorder(lattice, concentration)
    elif scheme == "S":
        disorder_structural = sulfur_vacancy_disorder(lattice, LENGTH[0], LENGTH[1], concentration)
    else:
        raise ValueError(f"unknown scheme {scheme!r}, expected 'Mo' or 'S'")

    configuration = kite.Configuration(
        divisions=list(DIVISIONS), length=list(LENGTH),
        boundaries=["periodic", "periodic"], is_complex=True, precision=1,
    )
    calculation = kite.Calculation(configuration)
    calculation.dos(num_points=DOS_NUM_POINTS, num_moments=DOS_NUM_MOMENTS,
                    num_random=DOS_NUM_RANDOM, num_disorder=DOS_NUM_DISORDER)

    kite.config_system(lattice, configuration, calculation,
                        disorder_structural=disorder_structural, filename=output_file)
    return output_file


if __name__ == "__main__":
    scheme = sys.argv[1] if len(sys.argv) > 1 else "S"
    concentration = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    out = f"mos2_{scheme}vac{int(round(concentration*100))}_dos-output.h5"
    run(scheme, concentration, out)
    print(f"Wrote {out}")
    print(f"\nLz (single-atom, 18x18, spin u/d x spd):")
    with np.printoptions(precision=2, suppress=True, linewidth=200):
        print(L_MATRICES["z"])
