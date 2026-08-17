""" Post-processing for mos2_ohe.py: Kubo-Bastin orbital Hall integral.

Adapted from examples/kane_mele_spin_hall_process.py's spin_hall() -- that
function is already provider-agnostic (general mod-4-aware Kubo-Bastin
reconstruction from /Calculation/CustomTwo/Gamma + NumVelocities, nothing
Kane-Mele-specific in the math), so orbital_hall() below is the same body,
renamed. NumVelocities=2 for this vertex pair (A=(1/2){v_x,L_z}, B=v_y),
matching the spin Hall case exactly -- no formula changes needed.

Units: L_z is dimensionless, in units of hbar, same convention as
kane_mele_spin_hall.py's s_z = (1/2)*sigma_z -- so spin_hall()'s e/(4*pi)-type
normalization carries over unchanged (see mos2_ohe.py's module docstring).

There is NO KITE-tools step for custom_two -- the Kubo-Bastin energy integral
is done here in Python, reading the raw double-Chebyshev moment matrix
directly from HDF5.

Usage:
    python mos2_ohe_process.py mos2_ohe_clean-output.h5
"""

import sys

import h5py
import numpy as np
from scipy.integrate import simpson


def green_coef(m, z):
    sqrt_1_z2 = np.sqrt(1.0 - z**2, dtype=complex)
    alpha = z - 1j * sqrt_1_z2
    return -1j * (alpha**m) / sqrt_1_z2


def dgreen_coef(m, z):
    sqrt_1_z2 = np.sqrt(1.0 - z**2, dtype=complex)
    alpha = z - 1j * sqrt_1_z2
    return (alpha**m) / (1.0 - z**2) * (m - 1j * z / sqrt_1_z2)


def fill_delta(E_grid, deltascat_dim, Moments_G):
    z = E_grid + 1j * deltascat_dim
    greenR = np.zeros((Moments_G, len(E_grid)), dtype=complex)
    for m in range(Moments_G):
        factor = -2.0 / ((2.0 if m == 0 else 1.0) * np.pi)
        greenR[m, :] = green_coef(m, z).imag * factor
    return greenR


def fill_dgreenR(E_grid, scat_dim, Moments_D):
    z = E_grid + 1j * scat_dim
    dgreenR = np.zeros((len(E_grid), Moments_D), dtype=complex)
    for m in range(Moments_D):
        factor = 2.0 / (2.0 if m == 0 else 1.0)
        dgreenR[:, m] = dgreen_coef(m, z) * factor
    return dgreenR


def fermi_function(E, mu, beta):
    return 1.0 / (1.0 + np.exp(np.clip(beta * (E - mu), -100, 100)))


def orbital_hall(file_path, mu_values, k_BT=0.01, scat_phys=0.04,
                  deltascat_phys=0.04, n_egrid=2000):
    """Return (mu_values, sigma^{Lz}_xy).

    General Kubo-Bastin reconstruction for custom_two(), valid for any
    NumVelocities parity (see rashba_edelstein_graphene_process.py's
    edelstein() docstring for the full mod-4 derivation). This vertex pair
    (A=(1/2){v_x,L_z}, B=v_y) has NumVelocities=2, the same parity as
    kane_mele_spin_hall.py's spin Hall vertex, so this reduces to the same
    ".imag branch" formula.
    """
    with h5py.File(file_path, "r") as f:
        num_orbitals = np.array(f["NOrbitals"]).item()
        latt_vecs = f["LattVectors"][:]
        unit_cell_area = np.abs(
            latt_vecs[0, 0] * latt_vecs[1, 1] - latt_vecs[0, 1] * latt_vecs[1, 0]
        )
        energy_scale = np.array(f["EnergyScale"]).item()
        num_velocities = int(np.array(f["/Calculation/CustomTwo/NumVelocities"]))
        moments_matrix = f["/Calculation/CustomTwo/Gamma"][:].T

    Moments_D, Moments_G = moments_matrix.shape
    spin_degeneracy = 1.0

    beta = energy_scale / k_BT
    scat_dim = scat_phys / energy_scale
    deltascat_dim = deltascat_phys / energy_scale

    # Integration grid in normalized energy (dimensionless, -1..1).
    E_grid = np.linspace(-0.995, 0.995, n_egrid)

    delta = fill_delta(E_grid, deltascat_dim, Moments_G)
    dgreenR = fill_dgreenR(E_grid, scat_dim, Moments_D)
    Z = np.einsum("ni,nm,im->i", delta, moments_matrix, dgreenR)

    p = num_velocities % 4
    X = {0: -2.0 * Z.imag, 1: 2.0 * Z.real, 2: 2.0 * Z.imag, 3: -2.0 * Z.real}[p]

    cond = np.empty(len(mu_values))
    for i, mu in enumerate(mu_values):
        integrand = X * fermi_function(E_grid, mu / energy_scale, beta)
        cond[i] = simpson(integrand, E_grid)

    units = 1.0 / (2.0 * np.pi)
    density_scale = (num_orbitals * spin_degeneracy) / (unit_cell_area * units)
    energy_scale_correction = energy_scale ** (num_velocities - 2)
    return mu_values, 2.0 * cond * density_scale * energy_scale_correction


if __name__ == "__main__":
    fname = sys.argv[1] if len(sys.argv) > 1 else "mos2_ohe_clean-output.h5"
    mus = np.linspace(-2.0, 2.0, 201)
    mu, sxy = orbital_hall(fname, mus)
    for m, s in zip(mu, sxy):
        print(f"{m: .4f}  {s: .6e}")
