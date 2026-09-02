# rs_lmto_import

A generic RS-LMTO-ASA (real-space linear muffin-tin orbital) Hamiltonian
importer for [KITE](https://quantum-kite.com/), plus a MoS2-specific
toolkit (vacancy disorder, DOS, orbital Hall effect) built on top of it.

## Setup

This needs KITE's own Python package (`kite`) importable -- **not** the
`quantum-kite` package on PyPI, which is a different/older version without
the `kite.lattice`/`kite.custom` modules this code needs. Point `PYTHONPATH`
at your kite-v2 checkout's `src/` directory, e.g.:

```bash
export PYTHONPATH=/path/to/kite-v2/src
```

or install kite-v2 in editable mode once your environment's `pip` supports
PEP 660 editable installs for a `pyproject.toml`-only package (older `pip`
does not).

## Files

- **`rs_lmto_import.py`** -- the core importer. Parses a material folder's
  `*_lat.dat`/`*_hr.dat` pair (`load_material()`) and builds a
  `kite.lattice.Lattice`. Also has:
  - `atomic_L_matrices()` -- the single-atom 18x18 (2 spins x 9 spd
    orbitals) orbital angular momentum matrices Lx/Ly/Lz, identical at
    every atomic site.
  - `standard_operators()`/`embed_atomic_operator()` -- tile a single-atom
    operator (L, S, or the orbital quadrupole moments) across the whole
    lattice.
  - `max_hopping_range` (on `load_material()`/`build_lattice()`) -- opt-in
    truncation of hoppings beyond a chosen range, instead of the default
    fail-closed behavior when a hopping exceeds KITE's compiled `NGHOSTS`.
    Reports how many entries were dropped and their largest magnitude.
  - `drop_core_orbitals()` -- drop deep, core-like orbitals (onsite energy
    below a cutoff) uniformly per atom type, before building the lattice.
  - `drop_named_orbitals()` -- drop specific named orbitals (e.g.
    `"Mo_u0"`), for an asymmetric removal a uniform cutoff can't express.
- **`mos2_vacancies.py`** -- Mo-vacancy and S-vacancy (S1/S2 randomly
  mixed) disorder schemes for the MoS2 lattice.
- **`mos2_dos_disorder.py`** -- DOS calculation with either vacancy scheme.
- **`mos2_ohe.py`** / **`mos2_ohe_process.py`** -- orbital Hall effect via
  KITE's `custom_two` machinery (the Lz analogue of the built-in spin Hall
  example), and its Kubo-Bastin post-processing.
- **`plot_bands_neighbor_compare.py`** -- see below.
- **`mos2/`** -- the MoS2 RS-LMTO data (`mos2_hr.dat`, `mos2_lat.dat`).

## `plot_bands_neighbor_compare.py`

Compares the band structure of a `max_hopping_range`-truncated lattice
against the full, untruncated raw data, along a k-path -- to check whether
truncating to a shorter hopping range (e.g. to fit KITE's compiled
`NGHOSTS`) actually changes the physics, rather than assuming it doesn't.

```bash
python plot_bands_neighbor_compare.py <material_dir> <orbitals_per_atom> <max_hopping_range>

# e.g.
python plot_bands_neighbor_compare.py mos2 18 2
```

This builds the truncated lattice through the real `load_material()`
pipeline and prints its import stats (including how many hoppings were
dropped by the truncation and their largest magnitude), then plots both
band structures overlaid and reports the maximum energy difference between
them across the whole path.

**The k-path is not lattice-agnostic.** The default path (`K'-G-K-M-G`) is
specific to a hexagonal Brillouin zone (a BZ vertex near 0 degrees is called
K, an edge midpoint is called M) and is wrong for other lattice shapes --
e.g. a square lattice's edge midpoint is conventionally called X, not M, and
it has no K point at all. For any other geometry, call `main()` directly
and pass your own `k_points_labels` (a list of `(point, label)` pairs in
Cartesian reciprocal-space coordinates -- the same units
`kite.visualize.make_path()` itself requires) instead of relying on the
hexagonal default:

```python
from plot_bands_neighbor_compare import main
main("my_square_material", orbitals_per_atom=..., max_hopping_range=2,
     k_points_labels=[(Gamma, "G"), (X, "X"), (M, "M"), (Gamma, "G")])
```
