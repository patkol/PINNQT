#!/usr/bin/env python3

# Copyright (c) 2024 ETH Zurich, Patrice Kolb


"""
Solving the 1D Schrödinger equation with open bc using PINN.
"""

import pdb
import random
import torch

import parameters as params
import physics
from device import Device
from visualization import visualize



# Setup

random.seed(params.seed)
torch.manual_seed(params.seed)
torch.set_default_device(params.device)
torch.set_default_dtype(params.si_real_dtype)
#torch.autograd.set_detect_anomaly(True) # For debugging, very expensive


device = Device(**physics.device_kwargs)


if __name__ == "__main__":
    previous_trainer = None
    for energy_string, trainer in device.trainers.items():
        if params.continuous_training and not previous_trainer is None:
            trainer.load_models(previous_trainer.models)

        trainer.train(
            report_each = params.report_each,
            max_n_steps = params.max_n_training_steps,
            max_time = params.max_time,
            min_loss = params.min_loss,
        )
        eval_times = dict(sorted(
            trainer.evaluation_times.items(),
            key=lambda item: item[1],
            reverse=True,
        ))
        total_eval_time = sum(eval_times.values())
        eval_relatives = dict((key, value / total_eval_time)
                              for key, value in eval_times.items())
        print("Evaluation time: ", total_eval_time * 1e-9, "s")
        for _, model_name in zip(range(25), eval_relatives):
            print(f"{eval_relatives[model_name]:.1%} {model_name}")

        previous_trainer = trainer

    visualize(device)
