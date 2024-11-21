# Copyright (c) 2024 ETH Zurich, Patrice Kolb


import os
import torch

from kolpinn import quantities
from kolpinn import training
from kolpinn.training import Trainer


def save_q_bulk(
    trainer: Trainer,
    subpath: str,
    *,
    included_quantities_labels=None,
    excluded_quantities_labels=None,
) -> None:
    if excluded_quantities_labels is None:
        excluded_quantities_labels = []

    qs = training.get_extended_qs(trainer.state)
    q_bulk = qs["bulk"]

    # Put on CPU & filter
    if included_quantities_labels is None:
        included_quantities_labels = q_bulk.keys()
    q_bulk = quantities.QuantityDict(
        q_bulk.grid,
        dict((label, q_bulk[label].cpu()) for label in included_quantities_labels),
    )
    for quantity_label in excluded_quantities_labels:
        q_bulk.pop(quantity_label)

    path_prefix = f"data/{trainer.config.saved_parameters_index:04d}/" + subpath
    os.makedirs(path_prefix, exist_ok=True)
    torch.save(q_bulk, path_prefix + "q_bulk.pkl")
