#!/usr/bin/env python3

import numpy as np
import torch
import matplotlib.pyplot as plt  # type: ignore


torch.set_default_device("cpu")

q_bulk = torch.load("../data/0446/newton_step0000/q_bulk.pkl")
matlab_path = "../matlab_results/barrier/0V/newton_step0000/"
matlab_data = {}
for name in ("E", "x", "n", "Vact_old", "Vact_new"):
    matlab_data[name] = np.loadtxt(
        f"{matlab_path}{name}.txt",
        delimiter=",",
    )
matlab_data["phiLs"] = np.loadtxt(
    f"{matlab_path}phiLs_real.txt",
    delimiter=",",
) + 1j * np.loadtxt(
    f"{matlab_path}phiLs_imag.txt",
    delimiter=",",
)

# Match units
matlab_data["n"] *= 1e-21

# Scale WF to be 1 at the input contact
matlab_data["phiLs"] /= matlab_data["phiLs"][:, 0:1]
q_bulk.overwrite("phi_L", q_bulk["phi_L"] / q_bulk["phi_L"][:, :, 0:1])

voltage_index = 0
energy_index = 9
energy_index_matlab = energy_index
grid = q_bulk.grid

energy_PINNQT = q_bulk["E_L"][voltage_index, energy_index, 0]
energy_MATLAB = matlab_data["E"][energy_index_matlab]
MATLAB_energy_offset = matlab_data["Vact_old"][0]
energy_MATLAB -= MATLAB_energy_offset
print(f"Energy: {energy_PINNQT:.10f} PINNQT / {energy_MATLAB:.10f} MATLAB eV")

plt.title("phiL")
plt.plot(
    grid["x"].cpu(),
    torch.real(q_bulk["phi_L"][voltage_index, energy_index, :]).cpu(),
    color="blue",
    label="PINNQT",
)
plt.plot(
    grid["x"].cpu(),
    torch.imag(q_bulk["phi_L"][voltage_index, energy_index, :]).cpu(),
    color="blue",
    linestyle="dashed",
)
plt.plot(
    matlab_data["x"],
    np.real(matlab_data["phiLs"][energy_index_matlab, :]),
    color="red",
    label="MATLAB",
)
plt.plot(
    matlab_data["x"],
    np.imag(matlab_data["phiLs"][energy_index_matlab, :]),
    color="red",
    linestyle="dashed",
)
plt.legend()
plt.grid()
plt.show()

plt.title("n")
plt.plot(grid["x"].cpu(), q_bulk["n"][voltage_index, 0, :].cpu(), label="PINNQT")
plt.plot(matlab_data["x"], matlab_data["n"], label="MATLAB")
plt.legend()
plt.grid()
plt.show()

plt.title("V_el_old")
plt.plot(grid["x"].cpu(), q_bulk["V_el"][voltage_index, 0, :].cpu(), label="PINNQT")
plt.plot(
    matlab_data["x"], matlab_data["Vact_old"] - MATLAB_energy_offset, label="MATLAB"
)
plt.legend()
plt.grid()
plt.show()

plt.title("V_el_new")
plt.plot(grid["x"].cpu(), q_bulk["V_el_new"][voltage_index, 0, :].cpu(), label="PINNQT")
plt.plot(matlab_data["x"], matlab_data["Vact_new"], label="MATLAB")
plt.legend()
plt.grid()
plt.show()
