""" Compare the band structure of a truncated (max_hopping_range<=NGHOSTS)
lattice against the FULL, untruncated raw data, along a proper k-path.

The truncated lattice is built through the real project pipeline
(load_material(..., max_hopping_range=...)), reusing kite.lattice's own
verified reciprocal_vectors()/brillouin_zone() for the k-path geometry
(never hand-derived) and kite.visualize.make_path()/compute_bands() for the
actual band structure -- the same, real API a downstream KITEx run would be
built from.

The FULL (untruncated) comparison case cannot go through build_lattice()
at all -- its NGHOSTS check exists specifically to refuse building a lattice
KITEx cannot correctly run, and there is no legitimate way to "opt out" of
that for an actual run. This script only needs the full data's raw
EIGENVALUES for comparison, evaluated at the exact same Cartesian k-points
the truncated lattice's own geometry produced, so it builds H(k) directly
from the raw parsed hr.dat (bypassing kite.lattice entirely, used ONLY for
this diagnostic purpose, never for an actual KITE run).

The k-path itself is NOT lattice-agnostic: which BZ vertices/edge-midpoints
count as which named high-symmetry point (K, M, X, ...) depends on the
lattice's actual point-group symmetry, not just its BZ's raw vertex count.
_mos2_hexagonal_k_points() below picks K/K'/M for a hexagonal BZ
specifically (a vertex near 0 degrees = K, its inversion partner = K', an
edge midpoint = M) and is NOT valid for other lattice shapes -- e.g. a
square lattice's edge midpoint is conventionally "X", not "M", and it has
no "K" point at all. For any other geometry, pass k_points_labels explicitly
(a list of (point, label) pairs in Cartesian reciprocal coordinates, i.e.
lattice.reciprocal_vectors() units -- see kite.visualize.make_path()'s own
docstring for that units requirement) instead of relying on the default.

Usage:
    python plot_bands_neighbor_compare.py mos2 18 2
    (material_dir, orbitals_per_atom, max_hopping_range)
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from rs_lmto_import import load_material, find_material_files, parse_lat_file, parse_hr_file
from kite import visualize


def full_hamiltonian_builder(material_dir):
    """H(k) built directly from the raw, untruncated hr.dat/lat.dat -- for
    comparison only, bypassing build_lattice()'s NGHOSTS safety check
    entirely (never use this path for an actual KITE run)."""
    hr_path, lat_path = find_material_files(material_dir)
    lat = parse_lat_file(lat_path)
    hr = parse_hr_file(hr_path)
    vectors = np.array(lat.vectors)

    Rs, Is, Js, Vs = [], [], [], []
    for (r, i, j), v in hr.items():
        Rs.append(r); Is.append(i - 1); Js.append(j - 1); Vs.append(v)
    Rs = np.array(Rs); Is = np.array(Is); Js = np.array(Js); Vs = np.array(Vs)
    Rcart = Rs @ vectors
    n_orb = Is.max() + 1

    def H_at(k):
        k3 = np.zeros(3)
        k3[:len(k)] = k
        phase = np.exp(1j * (Rcart @ k3))
        H = np.zeros((n_orb, n_orb), dtype=complex)
        np.add.at(H, (Is, Js), Vs * phase)
        return H

    return H_at


def _mos2_hexagonal_k_points(lattice):
    """K'-G-K-M-G for a HEXAGONAL BZ specifically (see module docstring) --
    not valid for other lattice shapes. Returns a list of (point, label)
    pairs, matching the k_points_labels argument main() otherwise expects
    the caller to supply directly."""
    bz = lattice.brillouin_zone()  # verified, Voronoi-based, never hand-derived
    angles = np.degrees(np.arctan2(bz[:, 1], bz[:, 0])) % 360
    K = bz[np.argmin(np.abs(angles - 0))]
    Kp = -K
    M = (bz[0] + bz[1]) / 2  # midpoint of one BZ edge
    G = np.zeros_like(K)
    return [(Kp, "K'"), (G, "G"), (K, "K"), (M, "M"), (G, "G")]


def main(material_dir="mos2", orbitals_per_atom=18, max_hopping_range=2,
         out_path=None, k_points_labels=None):
    lattice, stats = load_material(
        material_dir, orbitals_per_atom=orbitals_per_atom,
        spin_labels=("u", "d"), max_hopping_range=max_hopping_range,
    )
    print("Truncated lattice stats:", stats)

    if k_points_labels is None:
        k_points_labels = _mos2_hexagonal_k_points(lattice)
    points, point_labels = zip(*k_points_labels)

    k_path = visualize.make_path(*points, step=0.05, point_labels=list(point_labels))
    k_points, tick_indices, tick_labels = k_path

    bands_truncated = visualize.compute_bands(lattice, k_path)

    H_full = full_hamiltonian_builder(material_dir)
    bands_full = np.array([np.linalg.eigvalsh(H_full(k)) for k in k_points])

    max_diff = np.abs(np.sort(bands_truncated, axis=1) - np.sort(bands_full, axis=1)).max()
    print(f"max |E(truncated) - E(full)| over the whole path: {max_diff:.3e} eV")

    fig, ax = plt.subplots(figsize=(9, 7))
    x = np.arange(len(k_points))
    for b in range(bands_full.shape[1]):
        ax.plot(x, bands_full[:, b], color="0.5", linewidth=1.8, alpha=0.7,
                label="full data" if b == 0 else None)
    for b in range(bands_truncated.shape[1]):
        ax.plot(x, bands_truncated[:, b], color="C0", linewidth=1, linestyle="--",
                label=f"max_hopping_range={max_hopping_range}" if b == 0 else None)
    for pos, lab in zip(tick_indices, tick_labels):
        ax.axvline(pos, color="0.9", linewidth=0.8)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("Energy (eV)")
    ax.legend(fontsize=9)
    ax.set_title(f"{material_dir}: max_hopping_range={max_hopping_range} vs full data "
                 f"(max diff = {max_diff:.2e} eV)")
    plt.tight_layout()

    out_path = out_path or f"{material_dir}_neighbor_compare.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    return out_path


if __name__ == "__main__":
    material_dir = sys.argv[1] if len(sys.argv) > 1 else "mos2"
    orbitals_per_atom = int(sys.argv[2]) if len(sys.argv) > 2 else 18
    max_hopping_range = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    main(material_dir, orbitals_per_atom, max_hopping_range)
