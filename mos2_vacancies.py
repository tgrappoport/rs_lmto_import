""" Vacancy disorder schemes for the imported MoS2 RS-LMTO Hamiltonian.

Each scheme removes a WHOLE ATOM (all 18 orbitals: 9 spd x 2 spin) at each
vacancy site, never a partial orbital subset -- verified against the actual
C++ mechanism (Src/Hamiltonian/HamiltonianVacancies.cpp) and a real HDF5
export before relying on it: multiple add_vacancy() calls on the SAME
kite.StructuralDisorder object are merged into ONE HDF5 "Type" group (one
shared Concentration, one combined Orbitals list) -- the C++ side draws
exactly one random site per Type and places every listed orbital there
together. Separate StructuralDisorder objects get independent random
placement (see docs/documentation/disorder.md's "distribute vacancies on
both sublattices" note).

Two schemes:
  - Mo vacancy: one StructuralDisorder object covering all 18 Mo orbitals,
    KITE's own concentration-based random placement.
  - S vacancy: a SINGLE unified defect population at one total concentration
    -- each vacancy event independently lands on an S1 site OR an S2 site
    (50/50), never both, and removes that whole atom. This is NOT the same
    as two independent concentration-c populations (one for S1, one for S2),
    which would give ~2x the intended total defect density and could even
    place an S1 vacancy and an S2 vacancy at the same in-plane site by
    chance. Implemented with KITE's "exact position" StructuralDisorder mode
    (position=[[i,j],...], see its __init__) instead of concentration=,
    since the concentration-based random generator has no way to express
    "one shared random draw, split between two sublattices" -- the random
    site selection AND the S1/S2 coin flip are both done here, in Python,
    with a single shared random source, then handed to KITE as two lists of
    already-decided exact positions.
"""
import numpy as np

import kite

ORBITALS_PER_ATOM = 18
SPIN_LABELS = ("u", "d")
N_SPATIAL = ORBITALS_PER_ATOM // len(SPIN_LABELS)


def _atom_orbital_names(atom_label):
    return [f"{atom_label}_{spin}{orb}" for spin in SPIN_LABELS for orb in range(N_SPATIAL)]


def mo_vacancy_disorder(lattice, concentration):
    """One StructuralDisorder object, all 18 Mo orbitals -- each vacancy
    event removes the whole Mo atom at one random Mo site."""
    struc = kite.StructuralDisorder(lattice, concentration=concentration)
    for name in _atom_orbital_names("Mo"):
        struc.add_vacancy(name)
    return struc


def random_sulfur_sites(lx, ly, concentration, rng=None):
    """Pick int(concentration*lx*ly) unique (i, j) unit-cell sites (no
    replacement -- each is one physical defect), then independently flip a
    coin per site for S1 vs S2. Returns (positions_s1, positions_s2), each
    an (n, 2) int array suitable for StructuralDisorder(position=...)."""
    rng = rng if rng is not None else np.random.default_rng()
    n_sites = lx * ly
    n_vac = int(round(concentration * n_sites))
    if n_vac > n_sites:
        raise ValueError(f"concentration {concentration} needs {n_vac} sites, "
                          f"only {n_sites} available")
    chosen = rng.choice(n_sites, size=n_vac, replace=False)
    is_s1 = rng.random(n_vac) < 0.5
    positions = np.column_stack([chosen % lx, chosen // lx])
    return positions[is_s1], positions[~is_s1]


def sulfur_vacancy_disorder(lattice, lx, ly, concentration, rng=None):
    """A single unified S-vacancy population at `concentration` total
    (fraction of in-plane unit-cell sites), each defect randomly S1 or S2.
    Returns a list of StructuralDisorder objects (only the non-empty ones)
    ready for disorder_structural=."""
    pos_s1, pos_s2 = random_sulfur_sites(lx, ly, concentration, rng=rng)

    disorder = []
    for positions, atom_label in ((pos_s1, "S1"), (pos_s2, "S2")):
        if len(positions) == 0:
            continue
        struc = kite.StructuralDisorder(lattice, position=positions)
        for name in _atom_orbital_names(atom_label):
            struc.add_vacancy(name)
        disorder.append(struc)
    return disorder
