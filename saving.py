# Copyright (c) 2024 ETH Zurich, Patrice Kolb


import os
import torch

from kolpinn import training
from kolpinn.training import Trainer


def save_q_full(trainer: Trainer, *, excluded_quantities_labels=None):
    if excluded_quantities_labels is None:
        excluded_quantities_labels = []
    path_prefix = f'data/{trainer.config.saved_parameters_index:04d}/'
    os.makedirs(path_prefix, exist_ok=True)
    qs = training.get_extended_qs(trainer.state)
    q_full = qs['full']
    for excluded_quantity_label in excluded_quantities_labels:
        q_full.pop(excluded_quantity_label)
    torch.save(q_full, path_prefix + 'q_full.pkl')
