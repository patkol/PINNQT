#!/usr/bin/env python3

# Copyright (c) 2024 ETH Zurich, Patrice Kolb


"""
Solving Poisson with density from MATLAB
"""

import numpy as np
import torch
import matplotlib.pyplot as plt  # type: ignore

from kolpinn.grids import Grid
from kolpinn import quantities
from kolpinn.quantities import QuantityDict

import physical_constants as consts
import physics
from classes import Device


def get_V_electrostatic(q: QuantityDict) -> torch.Tensor:
    """From transformations.py, but assuming that n_pdV has been computed already"""
    newton_raphson_rate = 1
    dV_poisson = 1e-5 * consts.EV

    assert q.grid.dimensions_labels[-1] == "x"

    # Construct the discretized Laplace operator M assuming an
    # equispaced x grid
    # OPTIM: do this only once
    Nx = q.grid.dim_size["x"]
    dx = q.grid["x"][1] - q.grid["x"][0]
    M = torch.zeros(Nx, Nx)
    permittivity = quantities.squeeze_to(["x"], q["permittivity"], q.grid)
    for i in range(1, Nx):
        M[i, i - 1] = (permittivity[i] + permittivity[i - 1]) / (2 * dx**2)
    for i in range(0, Nx - 1):
        M[i, i + 1] = (permittivity[i] + permittivity[i + 1]) / (2 * dx**2)
    for i in range(1, Nx - 1):
        M[i, i] = -M[i, i - 1] - M[i, i + 1]

    # Von Neumann BC
    M[0, 0] = -(permittivity[0] + permittivity[1]) / (2 * dx**2)
    M[-1, -1] = -(permittivity[-1] + permittivity[-2]) / (2 * dx**2)

    # TODO: implementation that works if x is not the last coordinate,
    #       kolpinn function
    #       Then remove the assertion above
    rho = consts.Q_E * (q["doping"] - q["n"])
    Phi = q["V_el"] / -consts.Q_E
    F = torch.einsum("ij,...j->...i", M.to(torch.float64), Phi.to(torch.float64)) + rho

    dn_dV = (q["n_pdV"] - q["n"]) / dV_poisson
    drho_dV = -consts.Q_E * dn_dV
    drho_dPhi = -consts.Q_E * drho_dV
    torch.unsqueeze(M, 0)
    torch.unsqueeze(M, 0)
    J = M + torch.diag_embed(drho_dPhi)

    dPhi = newton_raphson_rate * torch.linalg.solve(-J, F)
    dV = dPhi * -consts.Q_E
    V_el = q["V_el"] + dV

    return V_el


# Load matlab data
matlab_path = "matlab_results/"
xs = torch.from_numpy(
    np.loadtxt(
        f"{matlab_path}x.txt",
        delimiter=",",
    )
)
ns = torch.from_numpy(
    np.loadtxt(
        f"{matlab_path}n.txt",
        delimiter=",",
    )
)
ns /= consts.CM**3
n_pdVs = torch.from_numpy(
    np.loadtxt(
        f"{matlab_path}np.txt",
        delimiter=",",
    )
)
n_pdVs /= consts.CM**3
V_el_old = torch.from_numpy(
    np.loadtxt(
        f"{matlab_path}Vact_old.txt",
        delimiter=",",
    )
)
V_el_new_MATLAB = torch.from_numpy(
    np.loadtxt(
        f"{matlab_path}Vact_new.txt",
        delimiter=",",
    )
)

# Put it into a QuantityDict
bulk_grid = Grid({"x": xs})
q_bulk = QuantityDict(bulk_grid, {"n": ns, "n_pdV": n_pdVs, "V_el": V_el_old})

# Add additional quantities
device = Device(**physics.device_kwargs_dict["barrier1_extended"])
dopings = torch.zeros_like(ns)
permittivities = torch.zeros_like(ns)
for layer_index in range(1, device.n_layers + 1):
    layer_slice = torch.logical_and(
        xs >= device.boundaries[layer_index - 1],
        xs < device.boundaries[layer_index] + 1e-6,
    )
    dopings[layer_slice] = device.dopings[layer_index]
    permittivities[layer_slice] = device.permittivities[layer_index]
q_bulk["doping"] = dopings
q_bulk["permittivity"] = permittivities

# Compute V_el
V_el_new_PINNQT = get_V_electrostatic(q_bulk)

# Visualize
plt.plot(xs, dopings, label="doping")
plt.plot(xs, permittivities, label="permittivity")
plt.legend()
plt.show()
plt.plot(xs, V_el_new_MATLAB, label="MATLAB")
plt.plot(xs, V_el_new_PINNQT, label="PINNQT")
plt.legend()
plt.show()
