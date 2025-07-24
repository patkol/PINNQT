#!/usr/bin/env python3

# Copyright (c) 2025 ETH Zurich, Patrice Kolb


"""
Solving the 1D Schrödinger equation with open bc using PINN.
"""

from typing import Dict, Optional, Sequence
import os
import time
import shutil
import copy
import random
import numpy as np
import torch

from kolpinn.grids import Grid
from kolpinn import quantities
from kolpinn.quantities import QuantityDict
from kolpinn import model
from kolpinn.model import MultiModel
from kolpinn import storage
from kolpinn import training
from kolpinn.training import Trainer

from classes import Device
import physical_constants as consts
import parameters as params
import devices
from constant_models_construction import get_constant_models
from eval_models_construction import get_eval_models
import trainer_construction
import saving
import plotting


# Setup

start_time = time.perf_counter()
random.seed(params.seed)
torch.manual_seed(params.seed)
torch.set_default_device(params.device)
torch.set_default_dtype(params.si_real_dtype)
# torch.autograd.set_detect_anomaly(True)  # For debugging, very expensive


device_kwargs = devices.device_kwargs_dict[params.simulated_device_name]
device = Device(**device_kwargs)
eval_models = get_eval_models(device)
saved_parameters_index = storage.get_next_parameters_index()
print("saved_parameters_index =", saved_parameters_index)


def get_trainer(grid_ranges: Dict[str, Dict[str, float]]):
    (
        trainer,
        unbatched_grids,
        quantities_requiring_grad,
    ) = trainer_construction.get_trainer(
        device=device,
        grid_ranges=grid_ranges,
        batch_sizes=params.batch_sizes,
        batch_bounds={},
        loss_aggregate_function=params.loss_aggregate_function,
        saved_parameters_index=saved_parameters_index,
        save_optimizer=params.save_optimizer,
        max_n_steps=params.max_n_training_steps,
        max_time=params.max_time,
        min_loss=params.min_loss,
        optimizer_reset_tol=params.optimizer_reset_tol,
        Optimizer=params.Optimizer,
        optimizer_kwargs=params.optimizer_kwargs,
        Scheduler=params.Scheduler,
        scheduler_kwargs=params.scheduler_kwargs,
    )

    return trainer, unbatched_grids, quantities_requiring_grad


def get_updated_trainer(
    trainer: Trainer,
    *,
    V_el: torch.Tensor,
    V_el_grid: Grid,
    unbatched_grids,
    quantities_requiring_grad,
    extra_pre_constant_models: Optional[Sequence[MultiModel]] = None,
) -> Trainer:
    if extra_pre_constant_models is None:
        extra_pre_constant_models = []

    def V_el_function(q: QuantityDict):
        return quantities.interpolate(V_el, V_el_grid, q.grid, dimension_label="x")

    dx_dict = trainer_construction.get_dx_dict()
    constant_models = get_constant_models(
        device,
        dx_dict=dx_dict,
        V_el_function=V_el_function,
    )
    constant_models = [*extra_pre_constant_models, *constant_models]
    const_qs = model.get_qs(
        unbatched_grids,
        constant_models,
        quantities_requiring_grad,
    )
    new_state = training.get_trainer_state(
        trainer.config,
        const_qs,
        trainer.state.trained_models,
        trainer.state.dependent_models,
    )

    return training.Trainer(new_state, trainer.config)


grid_ranges = {
    "voltage": {
        "start": params.VOLTAGE_MIN,
        "step": params.VOLTAGE_STEP,
        "end": params.VOLTAGE_MAX,
    },
    "voltage2": {
        "start": params.VOLTAGE2_MIN,
        "step": params.VOLTAGE2_STEP,
        "end": params.VOLTAGE2_MAX,
    },
    "DeltaE": {
        "start": params.E_MIN,
        "step": params.E_STEP,
        "end": params.E_MAX,
    },
    "x": {
        "step": params.X_STEP,
    },
}

trainer, unbatched_grids, quantities_requiring_grad = get_trainer(grid_ranges)

if params.loaded_parameters_index is not None:
    training.load(
        params.loaded_parameters_index,
        trainer,
        subpath=f"newton_step{params.loaded_parameters_NR_step:04d}/",
        load_optimizer=False,
        load_scheduler=False,
    )

if params.loaded_V_el_index is not None:
    # Replace V_el by the one from the loaded file
    loaded_step, loaded_label = params.loaded_V_el_NR_step, "V_el"
    if params.use_V_el_new:
        loaded_step -= 1
        loaded_label += "_new"
    loaded_q_bulk = torch.load(
        f"data/{params.loaded_V_el_index:04d}/newton_step{loaded_step:04d}/q_bulk.pkl"
    )
    trainer = get_updated_trainer(
        trainer,
        V_el=loaded_q_bulk[loaded_label].to(params.device),
        V_el_grid=loaded_q_bulk.grid,
        unbatched_grids=unbatched_grids,
        quantities_requiring_grad=quantities_requiring_grad,
    )
    del loaded_q_bulk

if params.imported_V_el_path is not None:
    # Import V_el from QT lecture code
    U_loaded = torch.tensor(np.load(params.imported_V_el_path + "U_drain.npy"))
    U2_loaded = torch.tensor(np.load(params.imported_V_el_path + "U_gate.npy"))
    assert torch.allclose(unbatched_grids["bulk"]["voltage"], U_loaded)
    assert torch.allclose(unbatched_grids["bulk"]["voltage2"], U2_loaded)
    x_loaded = torch.tensor(np.load(params.imported_V_el_path + "x.npy"))
    V_el = np.load(params.imported_V_el_path + "V_1d.npy")
    # PINNQT dimension order is U, U2, E, x
    # Lecture code order is x, E, U, U2
    V_el = torch.tensor(V_el.transpose((2, 3, 1, 0)))
    V_el_grid_dimensions = copy.copy(unbatched_grids["bulk"].dimensions)
    V_el_grid_dimensions["x"] = x_loaded
    V_el_grid = Grid(V_el_grid_dimensions)

    left_grid = unbatched_grids["boundary0"]
    right_grid = unbatched_grids[f"boundary{device.n_layers}"]
    V_el_left = quantities.interpolate(V_el, V_el_grid, left_grid, dimension_label="x")
    V_el_right = quantities.interpolate(
        V_el, V_el_grid, right_grid, dimension_label="x"
    )

    assert torch.allclose(
        V_el_left, torch.zeros(()), atol=torch.max(V_el) * 1e-5
    ), V_el_left

    # The voltages might not correspond to the grid because the fermi energies are
    # fixed instead. The actual voltages, which are stored in the qs, are not
    # equispaced and depend on voltage2.
    def correct_voltages(qs, *, voltage_corrected):
        for grid_name, q in qs.items():
            q["voltage_original"] = q["voltage"]
            q.overwrite("voltage", voltage_corrected)

    voltage_corrected = -V_el_right / consts.EV
    voltage_correcting_model = MultiModel(
        correct_voltages,
        "correct_voltages",
        kwargs={"voltage_corrected": voltage_corrected},
    )

    trainer = get_updated_trainer(
        trainer,
        V_el=V_el.to(params.device),
        V_el_grid=V_el_grid,
        unbatched_grids=unbatched_grids,
        quantities_requiring_grad=quantities_requiring_grad,
        extra_pre_constant_models=[voltage_correcting_model],
    )

    # Otherwise I need to correct the voltages again below
    assert params.n_newton_raphson_steps == 1

DeltaE_max = torch.max(unbatched_grids["bulk"]["DeltaE"]).item()
energy_cutoffs = np.arange(
    start=params.energy_cutoff_start,
    stop=DeltaE_max,
    step=params.energy_cutoff_delta,
)
if len(energy_cutoffs) == 0 or energy_cutoffs[-1] < DeltaE_max:
    energy_cutoffs = np.append(energy_cutoffs, [DeltaE_max])


if __name__ == "__main__":
    # Copy parameters.py
    saved_parameters_path = storage.get_parameters_path(saved_parameters_index)
    os.makedirs(saved_parameters_path, exist_ok=True)
    shutil.copy("parameters.py", saved_parameters_path)

    newton_raphson_step = (
        0 if params.loaded_V_el_index is None else params.loaded_V_el_NR_step
    )
    while True:  # Newton-Raphson loop
        save_subpath = f"newton_step{newton_raphson_step:04d}/"
        save_path = saved_parameters_path + save_subpath
        os.makedirs(save_path, exist_ok=True)

        for energy_cutoff in energy_cutoffs:
            print()
            print(f"Energy cutoff {energy_cutoff/consts.EV:.3g} eV")
            print()

            # Set the trained energy range
            trainer.config.get_batched_qs_kwargs["batch_bounds"]["DeltaE"] = (
                -float("inf"),
                energy_cutoff,
            )

            # Reset the state of the trainer
            training.reset_trainer_state(trainer)

            # Train
            training.train(
                trainer,
                report_each=params.report_each,
                save_if_best=True,
                save_subpath=save_subpath,
            )
            print(
                "Total time passed:", time.perf_counter() - start_time, "s", flush=True
            )

        # Print evaluation times
        eval_times = dict(
            sorted(
                trainer.state.evaluation_times.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )
        total_eval_time = sum(eval_times.values())
        eval_relatives = dict(
            (key, value / total_eval_time) for key, value in eval_times.items()
        )
        print("Total time passed:", time.perf_counter() - start_time, "s", flush=True)
        print("Evaluation time: ", total_eval_time * 1e-9, "s", flush=True)
        for _, model_name in zip(range(25), eval_relatives):
            print(f"{eval_relatives[model_name]:.1%} {model_name}", flush=True)

        # Get all quantities
        extended_qs = training.get_extended_qs(
            trainer.state, additional_models=eval_models
        )

        # Save quantities and plots
        print("Saving quantities...", flush=True)
        saved_quantities = [
            "voltage",
            "DeltaE",
            "x",
            "E_L",
            "E_R",
            "V_int",
            "V_el",
            "V_el_new",
            "m_eff",
            "n",
            "T_L",
            "T_R",
            "R_L",
            "R_R",
            "phi_L",
            "phi_R",
            "incoming_coeff_L",
            "incoming_coeff_R",
        ]
        if params.use_voltage2:
            saved_quantities.append("voltage2")
        if params.imported_V_el_path is not None:
            # "voltage" won't correspond to the grid value,
            # "voltage_original" holds that
            saved_quantities.append("voltage_original")
        saving.save_q_bulk(
            extended_qs["bulk"],
            path_prefix=f"data/{trainer.config.saved_parameters_index:04d}/"
            + save_subpath,
            included_quantities_labels=saved_quantities,
        )
        print("Done saving quantities", flush=True)
        print("Saving plots...", flush=True)
        plotting.save_plots(extended_qs, trainer, device, prefix=save_subpath)
        print("Done saving plots", flush=True)

        # Set up the next Newton-Raphson step
        newton_raphson_step += 1
        if newton_raphson_step >= params.n_newton_raphson_steps:
            break

        print()
        print("Newton-Raphson step", newton_raphson_step, flush=True)
        print()

        # Set up a new trainer with the updated V_el
        if params.reset_weights_per_nr_step:
            trainer, unbatched_grids, quantities_requiring_grad = get_trainer(
                grid_ranges
            )
        trainer = get_updated_trainer(
            trainer,
            V_el=extended_qs["bulk"]["V_el_new"],
            V_el_grid=extended_qs["bulk"].grid,
            unbatched_grids=unbatched_grids,
            quantities_requiring_grad=quantities_requiring_grad,
        )
