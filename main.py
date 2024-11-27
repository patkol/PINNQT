#!/usr/bin/env python3

# Copyright (c) 2024 ETH Zurich, Patrice Kolb


"""
Solving the 1D Schrödinger equation with open bc using PINN.
"""

import os
import shutil
import random
import torch

from kolpinn.grids import Grid
from kolpinn import quantities
from kolpinn.quantities import QuantityDict
from kolpinn import model
from kolpinn import storage
from kolpinn import training
from kolpinn.training import Trainer

import parameters as params
import physics
from classes import Device
from constant_models_construction import get_constant_models
from eval_models_construction import get_eval_models
import trainer_construction
import saving
import plotting


# Setup

random.seed(params.seed)
torch.manual_seed(params.seed)
torch.set_default_device(params.device)
torch.set_default_dtype(params.si_real_dtype)
# torch.autograd.set_detect_anomaly(True)  # For debugging, very expensive

device = Device(**physics.device_kwargs)
eval_models = get_eval_models(device)
saved_parameters_index = storage.get_next_parameters_index()
print("saved_parameters_index =", saved_parameters_index)


def get_trainer():
    (
        trainer,
        unbatched_grids,
        quantities_requiring_grad,
    ) = trainer_construction.get_trainer(
        device=device,
        batch_sizes=params.batch_sizes,
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
) -> Trainer:
    def V_el_function(q: QuantityDict):
        return quantities.interpolate(V_el, V_el_grid, q.grid, dimension_label="x")

    dx_dict = trainer_construction.get_dx_dict()
    constant_models = get_constant_models(
        device,
        dx_dict=dx_dict,
        V_el_function=V_el_function,
    )
    const_qs = model.get_qs(
        unbatched_grids,
        constant_models,
        quantities_requiring_grad,
    )
    new_state = trainer_construction.get_trainer_state(
        trainer.config,
        const_qs,
        trainer.state.trained_models,
        trainer.state.dependent_models,
    )

    return training.Trainer(new_state, trainer.config)


trainer, unbatched_grids, quantities_requiring_grad = get_trainer()

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
    loaded_q_bulk = torch.load(
        f"data/{params.loaded_V_el_index:04d}/newton_step{params.loaded_V_el_NR_step:04d}/q_bulk.pkl"
    )
    trainer = get_updated_trainer(
        trainer,
        V_el=loaded_q_bulk["V_el"].to(params.device),
        V_el_grid=loaded_q_bulk.grid,
        unbatched_grids=unbatched_grids,
        quantities_requiring_grad=quantities_requiring_grad,
    )
    del loaded_q_bulk


if __name__ == "__main__":
    # Copy parameters.py
    saved_parameters_path = storage.get_parameters_path(saved_parameters_index)
    os.makedirs(saved_parameters_path, exist_ok=True)
    shutil.copy("parameters.py", saved_parameters_path)

    newton_raphson_step = (
        0 if params.loaded_V_el_index is None else params.loaded_V_el_NR_step
    )
    while True:
        save_subpath = f"newton_step{newton_raphson_step:04d}/"
        save_path = saved_parameters_path + save_subpath
        os.makedirs(save_path, exist_ok=True)

        # Train
        training.train(
            trainer,
            report_each=params.report_each,
            save_if_best=True,
            save_subpath=save_subpath,
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
        print("Evaluation time: ", total_eval_time * 1e-9, "s")
        for _, model_name in zip(range(25), eval_relatives):
            print(f"{eval_relatives[model_name]:.1%} {model_name}")

        # Get all quantities
        extended_qs = training.get_extended_qs(
            trainer.state, additional_models=eval_models
        )

        # Save quantities and plots
        saving.save_q_bulk(
            extended_qs["bulk"],
            path_prefix=f"data/{trainer.config.saved_parameters_index:04d}/"
            + save_subpath,
            included_quantities_labels=[
                "voltage",
                "DeltaE",
                "x",
                "E_L",
                "E_R",
                "V_el",
                "V_el_new",
                "n",
                "T_L",
                "T_R",
                "R_L",
                "R_R",
                "phi_L",
            ],
        )
        plotting.save_plots(extended_qs, trainer, device, prefix=save_subpath)

        # Set up the next Newton-Raphson step
        newton_raphson_step += 1
        if newton_raphson_step >= params.n_newton_raphson_steps:
            break

        print()
        print("Newton-Raphson step", newton_raphson_step)
        print()

        # Set up a new trainer with the updated V_el
        if params.reset_weights_per_nr_step:
            trainer, unbatched_grids, quantities_requiring_grad = get_trainer()
        trainer = get_updated_trainer(
            trainer,
            V_el=extended_qs["bulk"]["V_el_new"],
            V_el_grid=extended_qs["bulk"].grid,
            unbatched_grids=unbatched_grids,
            quantities_requiring_grad=quantities_requiring_grad,
        )
