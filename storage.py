import os
import torch

from device import Device



def save_q_full(device: Device, *, excluded_quantities_labels=None):
    if excluded_quantities_labels is None:
        excluded_quantities_labels = []
    path_prefix = f'data/{device.trainer.saved_parameters_index:04d}/'
    os.makedirs(path_prefix, exist_ok=True)
    qs = device.get_extended_qs()
    q_full = qs['full']
    for excluded_quantity_label in excluded_quantities_labels:
        q_full.pop(excluded_quantity_label)
    torch.save(q_full, path_prefix + 'q_full.pkl')
