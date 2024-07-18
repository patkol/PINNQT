# Copyright (c) 2024 ETH Zurich, Patrice Kolb


from typing import Dict
import copy
import torch

from kolpinn.grids import Grid
from kolpinn import batching

import parameters as params
from classes import Device


def get_voltages(*, V_min: float, V_max: float, V_step: float) -> torch.Tensor:
    voltages = torch.arange(V_min, V_max, V_step, dtype=params.si_real_dtype)
    return voltages


def get_energies(*, E_min: float, E_max: float, E_step: float) -> torch.Tensor:
    energies = torch.arange(E_min, E_max, E_step, dtype=params.si_real_dtype)
    return energies


def get_xs(device: Device, *, N_x: int) -> torch.Tensor:
    x_left = device.boundaries[0]
    x_right = device.boundaries[-1]
    # We're excluding the left- and rightmost points to avoid special cases
    x_step = (x_right - x_left) / (N_x + 2 - 1)
    xs = torch.linspace(
        x_left + x_step,
        x_right - x_step,
        N_x,
    )
    return xs


def get_layer_subgrid(layer_index: int, parent_grid: Grid, device: Device):
    x_left = device.boundaries[layer_index - 1]
    x_right = device.boundaries[layer_index]
    layer_subgrid = parent_grid.get_subgrid(
        {'x': lambda x: torch.logical_and(x >= x_left, x < x_right)},
        copy_all=False,
    )
    return layer_subgrid


def get_layer_subgrids(parent_grid: Grid, device: Device):
    layer_subgrids: Dict[str, Grid] = {}
    for layer_index in range(1, device.n_layers + 1):
        grid_name = f'bulk{layer_index}'
        layer_subgrids[grid_name] = get_layer_subgrid(
            layer_index,
            parent_grid,
            device,
        )

    return layer_subgrids


def update_layer_subgrids(grids: Dict[str, Grid], device: Device) -> None:
    grids.update(get_layer_subgrids(grids['bulk'], device))


def get_unbatched_grids(
    device: Device,
    *,
    V_min: float,
    V_max: float,
    V_step: float,
    E_min: float,
    E_max: float,
    E_step: float,
    N_x: int,
    dx_dict: Dict[str, float],
) -> Dict[str, Grid]:
    voltages = get_voltages(V_min=V_min, V_max=V_max, V_step=V_step)
    energies = get_energies(E_min=E_min, E_max=E_max, E_step=E_step)
    xs = get_xs(device, N_x=N_x)
    grids: Dict[str, Grid] = {}

    # Bulk
    grids['bulk'] = Grid({
        'voltage': voltages,
        'DeltaE': energies,
        'x': xs,
    })

    # Layers
    update_layer_subgrids(grids, device)

    # Boundaries
    for i in range(device.n_layers + 1):
        for dx_string, dx_shift in dx_dict.items():
            grid_name = f'boundary{i}' + dx_string
            x = device.boundaries[i] + dx_shift
            grids[grid_name] = Grid({
                'voltage': voltages,
                'DeltaE': energies,
                'x': torch.tensor([x], dtype=params.si_real_dtype),
            })

    return grids


def get_batched_grids(
    grids: dict[str, Grid],
    *,
    device: Device,
    batch_sizes: Dict[str, int] = {},
) -> Dict[str, Grid]:
    grids = copy.copy(grids)

    grids['bulk'] = batching.get_random_subgrid(
        grids['bulk'],
        batch_sizes,
    )
    update_layer_subgrids(grids, device)

    return grids
