#!/usr/bin/env python3

# Copyright (c) 2025 ETH Zurich, Patrice Kolb


"""
Testing whether the loss vanishes for the reference solution coming from matlab
"""


from typing import Sequence
import copy
import time
import numpy as np
import torch

from kolpinn.grids import Grid
from kolpinn import quantities
from kolpinn.model import MultiModel
from kolpinn import training

import parameters as params
from classes import Contact
import trainer_construction
import plotting
import main
from main import device, trainer


def expand_to_3d(tensor: torch.Tensor):
    assert len(tensor.size()) <= 3

    tensor_out = tensor
    while len(tensor_out.size()) < 3:
        tensor_out = torch.unsqueeze(tensor_out, 0)

    return tensor_out


def matlab_phi_trafo(
    qs, i: int, contact: Contact, grid_names: Sequence[str], matlab_grid: Grid
):
    """
    Only works for a single voltage
    """

    # Set loaded phi
    for grid_name in grid_names:
        q = qs[grid_name]

        phi_matlab = expand_to_3d(torch.tensor(matlab_data[f"phi{contact}s"]))
        q[f"phi{i}_{contact}"] = quantities.interpolate(
            phi_matlab, matlab_grid, q.grid, dimension_label="x"
        )

    return qs


# Load MATLAB data
matlab_path = "matlab_results/barrier/0V_0p01eV/newton_step0001/"
matlab_data = {}
for name in ("E", "x", "Vact_old"):
    matlab_data[name] = np.loadtxt(
        f"{matlab_path}{name}.txt",
        delimiter=",",
    )
for contact in device.contacts:
    matlab_data[f"phi{contact}s"] = np.loadtxt(
        f"{matlab_path}phi{contact}s_real.txt",
        delimiter=",",
    ) + 1j * np.loadtxt(
        f"{matlab_path}phi{contact}s_imag.txt",
        delimiter=",",
    )

# Create the grid the MATLAB data lies on
q_full = trainer.state.const_qs["bulk"]
matlab_dimensions = copy.copy(q_full.grid.dimensions)
matlab_dimensions["x"] = torch.tensor(matlab_data["x"])
MATLAB_energies = torch.tensor(matlab_data["E"])
offset_index = 0 if contact.index == 0 else -1
MATLAB_energy_offset = matlab_data["Vact_old"][offset_index]
MATLAB_energies -= MATLAB_energy_offset
energies = q_full[f"E_{contact}"].reshape((q_full.grid.dim_size["DeltaE"],))
assert torch.allclose(energies, MATLAB_energies), (energies, MATLAB_energies)
matlab_grid = Grid(matlab_dimensions)

# Replace the trainer's V_el by the matlab one
V_el_matlab = expand_to_3d(torch.tensor(matlab_data["Vact_old"])) - MATLAB_energy_offset
trainer = main.get_updated_trainer(
    trainer,
    V_el=V_el_matlab.to(params.device),
    V_el_grid=matlab_grid,
    unbatched_grids=main.unbatched_grids,
    quantities_requiring_grad=main.quantities_requiring_grad,
)

# Replace the trainer's phi_trafo by the matlab one
dx_dict = trainer_construction.get_dx_dict()
for contact in device.contacts:
    for i in range(1, device.n_layers + 1):
        bulk = f"bulk{i}"
        bulks = [bulk]
        boundary_in = f"boundary{contact.get_in_boundary_index(i)}"
        boundary_out = f"boundary{contact.get_out_boundary_index(i)}"
        boundaries_in = [boundary_in + dx_string for dx_string in dx_dict.keys()]
        boundaries_out = [boundary_out + dx_string for dx_string in dx_dict.keys()]
        matlab_phi_model = MultiModel(
            matlab_phi_trafo,
            f"phi{i}_{contact}",
            kwargs={
                "i": i,
                "contact": contact,
                "grid_names": boundaries_in + bulks + boundaries_out,
                "matlab_grid": matlab_grid,
            },
        )
        trainer.state.dependent_models = [
            matlab_phi_model if m.name == matlab_phi_model.name else m
            for m in trainer.state.dependent_models
        ]


if __name__ == "__main__":
    trainer.state.training_start_time = time.perf_counter()
    losses, extended_qs = training.get_losses(trainer, return_extended_qs=True)
    training.print_progress(trainer)
    plotting.save_plots(extended_qs, trainer, device, prefix="MATLAB_test/")
