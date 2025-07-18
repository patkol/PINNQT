# Copyright (c) 2025 ETH Zurich, Patrice Kolb


from typing import Dict, Callable, Any, Optional
import torch

from kolpinn import grids
from kolpinn.grids import Grid, Subgrid

from classes import Device
import parameters as params


def get_xs(device: Device, *, x_step: float) -> torch.Tensor:
    x_left = device.boundaries[0]
    x_right = device.boundaries[-1]
    # We're excluding the left- and rightmost points to avoid special cases
    xs = torch.arange(
        start=x_left + x_step / 2,
        end=x_right - x_step / 2,
        step=x_step,
    )
    return xs


def get_layer_subgrid(layer_index: int, parent_grid: Grid, device: Device):
    x_left = device.boundaries[layer_index - 1]
    x_right = device.boundaries[layer_index]
    layer_subgrid = parent_grid.get_subgrid(
        {"x": lambda x: torch.logical_and(x >= x_left, x < x_right)},
        copy_all=False,
    )
    return layer_subgrid


def get_layer_subgrids(parent_grid: Grid, device: Device):
    layer_subgrids: Dict[str, Grid] = {}
    for layer_index in range(1, device.n_layers + 1):
        grid_name = f"bulk{layer_index}"
        layer_subgrids[grid_name] = get_layer_subgrid(
            layer_index,
            parent_grid,
            device,
        )

    return layer_subgrids


def update_layer_subgrids(grids: Dict[str, Grid], device: Device) -> None:
    grids.update(get_layer_subgrids(grids["bulk"], device))


def get_unbatched_grids(
    device: Device,
    *,
    grid_ranges: Dict[str, Dict[str, float]],
    dx_dict: Dict[str, float],
    use_voltage2: bool,
    V2_min: Optional[float] = None,
    V2_max: Optional[float] = None,
    V2_step: Optional[float] = None,
) -> Dict[str, Grid]:
    voltages = torch.arange(**grid_ranges["voltage"], dtype=params.si_real_dtype)
    energies = torch.arange(**grid_ranges["DeltaE"], dtype=params.si_real_dtype)
    xs = get_xs(device, x_step=grid_ranges["x"]["step"])
    grids: Dict[str, Grid] = {}

    # Bulk
    grid_items = [("voltage", voltages), ("DeltaE", energies), ("x", xs)]
    if use_voltage2:
        assert V2_min is not None and V2_max is not None and V2_step is not None
        voltages2 = torch.arange(**grid_ranges["voltage2"], dtype=params.si_real_dtype)
        grid_items.insert(1, ("voltage2", voltages2))
    grids["bulk"] = Grid(dict(grid_items))

    # Layers
    update_layer_subgrids(grids, device)

    # Boundaries
    for boundary_index in range(device.n_layers + 1):
        for dx_string, dx_shift in dx_dict.items():
            grid_name = f"boundary{boundary_index}" + dx_string
            x = device.boundaries[boundary_index] + dx_shift
            assert grid_items[-1][0] == "x"
            grid_items[-1] = ("x", torch.tensor([x], dtype=params.si_real_dtype))
            grids[grid_name] = Grid(dict(grid_items))

    return grids


def get_batched_grids(
    unbatched_grids: Dict[str, Grid],
    *,
    batched_indices_dict_fn: Callable,
    batching_kwargs: Dict[str, Any],
    device: Device,
    dx_dict: Dict[str, float],
) -> Dict[str, Grid]:
    """
    Batch "bulk" according to "batched_indices_dict_fn", then batch the layers
    and boundaries consistently. The batched layers will be subgrids of
    the batched "bulk".
    """

    batched_grids: dict[str, Grid] = {}
    batched_indices_dict = batched_indices_dict_fn(
        unbatched_grids["bulk"],
        **batching_kwargs,
    )

    # Bulk
    batched_grids["bulk"] = Subgrid(
        unbatched_grids["bulk"],
        batched_indices_dict,
        copy_all=False,
    )

    # Layers
    update_layer_subgrids(batched_grids, device)

    # Boundaries
    # Not batching the boundaries in 'x'
    batched_indices_dict_excluding_x = dict(
        (dim, indices) for dim, indices in batched_indices_dict.items() if dim != "x"
    )
    for boundary_index in range(device.n_layers + 1):
        for dx_string, dx_shift in dx_dict.items():
            grid_name = f"boundary{boundary_index}" + dx_string
            batched_grids[grid_name] = Subgrid(
                unbatched_grids[grid_name],
                batched_indices_dict_excluding_x,
                copy_all=False,
            )

    return batched_grids


def get_batched_layer_grids_as_subgrids(
    batched_grids: Dict[str, Grid], unbatched_grids: Dict[str, Grid], *, device: Device
) -> Dict[str, Grid]:
    """
    Return the batched grids "bulk{i}", which are subgrids of the batched "bulk",
    as subgrids of the unbatched "bulk{i}".
    Initial hierarchy:
        unbatched_grids["bulk"] -> batched_grids["bulk"] -> batched_grids["bulk{i}"]
    Hierarchy of output:
        unbatched_grids["bulk"] -> unbatched_grids["bulk{i}"] -> layer_subgrids["bulk{i}"]
    """

    layer_subgrids: Dict[str, Grid] = {}
    for layer_index in range(1, device.n_layers + 1):
        layer_name = f"bulk{layer_index}"
        subgrid_of_batched_bulk = batched_grids[layer_name]
        assert isinstance(subgrid_of_batched_bulk, Subgrid)
        subgrid_of_unbatched_bulk = grids.get_as_subgrid(
            subgrid_of_batched_bulk, copy_all=False
        )

        unbatched_layer = unbatched_grids[layer_name]
        assert isinstance(unbatched_layer, Subgrid)
        subgrid_of_unbatched_layer = grids.get_as_subsubgrid(
            subgrid_of_unbatched_bulk,
            unbatched_layer,
            copy_all=False,
        )

        layer_subgrids[layer_name] = subgrid_of_unbatched_layer

    return layer_subgrids
