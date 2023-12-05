#!/usr/bin/env python3

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


device = Device(**physics.device_kwargs)


if __name__ == "__main__":
    device.trainer.train(
        params.n_training_steps,
        params.report_each,
        max_time = params.max_time,
    )
    visualize(device)
