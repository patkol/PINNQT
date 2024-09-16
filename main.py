#!/usr/bin/env python3

# Copyright (c) 2024 ETH Zurich, Patrice Kolb


"""
Solving the 1D Schrödinger equation with open bc using PINN.
"""

import os
import shutil
import random
import torch

from kolpinn import storage
from kolpinn import training

import parameters as params
import physics
from classes import Device
import trainer_construction

# import saving
import plotting


# Setup

random.seed(params.seed)
torch.manual_seed(params.seed)
torch.set_default_device(params.device)
torch.set_default_dtype(params.si_real_dtype)
# torch.autograd.set_detect_anomaly(True)  # For debugging, very expensive


device = Device(**physics.device_kwargs)
saved_parameters_index = storage.get_next_parameters_index()
print("saved_parameters_index =", saved_parameters_index)

trainer = trainer_construction.get_trainer(
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
training.load(
    params.loaded_parameters_index,
    trainer,
    load_optimizer=False,
    load_scheduler=False,
)


if __name__ == "__main__":
    # Copy parameters.py
    saved_parameters_path = storage.get_parameters_path(saved_parameters_index)
    os.makedirs(saved_parameters_path, exist_ok=True)
    shutil.copy("parameters.py", saved_parameters_path)

    for newton_raphson_step in range(params.n_newton_raphson_steps):
        # Train
        training.train(trainer, report_each=params.report_each, save_if_best=True)

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

        # Save quantities and plots
        # saving.save_q_full(device, excluded_quantities_labels=['phi_L', 'phi_R'])
        plotting.save_plots(
            trainer, device, prefix=f"newton_raphson{newton_raphson_step}/"
        )

        if newton_raphson_step + 1 < params.n_newton_raphson_steps:
            # Newton-Raphson step
            training.reset_optimizer(trainer)
            # trainer.const_qs["bulk"]["V_el"] =
