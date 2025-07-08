# Copyright (c) 2025 ETH Zurich, Patrice Kolb


from collections.abc import Sequence
from typing import Dict, Callable, Optional, Any, Tuple
import copy

from kolpinn.grids import Grid, Subgrid
from kolpinn import quantities
from kolpinn.quantities import QuantityDict
from kolpinn import batching
from kolpinn import model
from kolpinn.model import MultiModel
from kolpinn import training

from classes import Device
import parameters as params
import formulas
import grid_construction
from constant_models_construction import get_constant_models
from trained_models_construction import get_trained_models
from loss_models_construction import get_loss_models


def get_dx_dict():
    dx_dict = {"": 0.0}
    if params.fd_first_derivatives:
        dx_dict["_mdx"] = -params.dx
        dx_dict["_pdx"] = params.dx

    return dx_dict


def get_batched_qs_fn(
    *,
    unbatched_grids: Dict[str, Grid],
    device: Device,
    dx_dict: Dict[str, float],
) -> Callable:
    def get_batched_qs(
        full_qs: Dict[str, QuantityDict],
        randomize: bool,
        *,
        batch_sizes: Dict[str, int],
        batch_bounds: Dict[str, Tuple[float, float]],
    ) -> Dict[str, QuantityDict]:
        batched_indices_dict_fns = [
            lambda grid, randomize: batching.get_equispaced_batched_indices_dict(
                grid, batch_sizes=batch_sizes, randomize=randomize
            ),
            lambda grid, randomize: batching.get_bounds_batched_indices_dict(
                grid, batch_bounds=batch_bounds
            ),
        ]

        batched_grids = grid_construction.get_batched_grids(
            unbatched_grids,
            batched_indices_dict_fn=batching.get_combined_batched_indices_dict,
            batching_kwargs={
                "batched_indices_dict_fns": batched_indices_dict_fns,
                "randomize": randomize,
            },
            device=device,
            dx_dict=dx_dict,
        )
        layer_subgrids = grid_construction.get_batched_layer_grids_as_subgrids(
            batched_grids,
            unbatched_grids,
            device=device,
        )
        # Use direct subgrids of the unbatched layers for the restriction
        batched_grids_for_restriction = copy.copy(batched_grids)
        batched_grids_for_restriction.update(layer_subgrids)

        batched_qs: Dict[str, QuantityDict] = {}
        for grid_name, grid in batched_grids.items():
            batched_grid_for_restriction = batched_grids_for_restriction[grid_name]
            assert isinstance(grid, Subgrid)
            assert isinstance(batched_grid_for_restriction, Subgrid)
            batched_qs[grid_name] = quantities.restrict_quantities(
                full_qs[grid_name],
                grid,
                subgrid_for_restriction=batched_grid_for_restriction,
            )

        return batched_qs

    return get_batched_qs


def get_trainer(
    device: Device,
    *,
    grid_ranges: Dict[str, Dict[str, float]],
    batch_sizes: Dict[str, int],
    batch_bounds: Dict[str, Tuple[float, float]],
    loss_aggregate_function: Callable[[Sequence], Any],
    saved_parameters_index: int,
    save_optimizer: bool,
    max_n_steps: Optional[int],
    max_time: Optional[float],
    min_loss: Optional[float],
    optimizer_reset_tol: float,
    Optimizer: type,
    optimizer_kwargs: Dict[str, Any],
    Scheduler: Optional[type],
    scheduler_kwargs: Dict[str, Any],
    extra_pre_constant_models: Optional[Sequence[MultiModel]] = None,
):
    """
    extra_pre_constant_models: models to be evaluated before the ones from
        constand_models_construction.get_constant_models, but after the grids have been
        added.
    """

    if extra_pre_constant_models is None:
        extra_pre_constant_models = []

    dx_dict = get_dx_dict()
    unbatched_grids = grid_construction.get_unbatched_grids(
        device,
        grid_ranges=grid_ranges,
        dx_dict=dx_dict,
        use_voltage2=params.use_voltage2,
        V2_min=params.VOLTAGE2_MIN,
        V2_max=params.VOLTAGE2_MAX,
        V2_step=params.VOLTAGE2_STEP,
    )

    x_grad_required = (
        not params.fd_first_derivatives or not params.fd_second_derivatives
    )
    quantities_requiring_grad: Dict[str, Sequence[str]] = {}
    for grid_name in unbatched_grids.keys():
        quantities_requiring_grad[grid_name] = ["x"] if x_grad_required else []

    constant_models = get_constant_models(
        device,
        dx_dict=dx_dict,
        V_el_function=lambda q: formulas.get_V_el_guess(
            q, params.V_el_guess_type, **params.V_el_guess_kwargs
        ),
    )
    constant_models = [*extra_pre_constant_models, *constant_models]
    trained_models, trained_models_labels = get_trained_models(
        device,
        dx_dict=dx_dict,
    )
    dependent_models, loss_quantities = get_loss_models(
        device,
        dx_dict=dx_dict,
    )

    const_qs = model.get_qs(
        unbatched_grids,
        constant_models,
        quantities_requiring_grad,
    )

    get_batched_qs = get_batched_qs_fn(
        unbatched_grids=unbatched_grids,
        device=device,
        dx_dict=dx_dict,
    )

    config = training.TrainerConfig(
        get_batched_qs=get_batched_qs,
        get_batched_qs_kwargs={
            "batch_sizes": batch_sizes,
            "batch_bounds": batch_bounds,
        },
        loss_quantities=loss_quantities,
        loss_aggregate_function=loss_aggregate_function,
        saved_parameters_index=saved_parameters_index,
        save_optimizer=save_optimizer,
        Optimizer=Optimizer,
        optimizer_kwargs=optimizer_kwargs,
        max_n_steps=max_n_steps,
        max_time=max_time,
        min_loss=min_loss,
        optimizer_reset_tol=optimizer_reset_tol,
    )
    state = training.get_trainer_state(
        config,
        const_qs,
        trained_models,
        dependent_models,
    )
    trainer = training.Trainer(state, config)

    return trainer, unbatched_grids, quantities_requiring_grad
