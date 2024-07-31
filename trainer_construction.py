# Copyright (c) 2024 ETH Zurich, Patrice Kolb


from collections.abc import Sequence
from typing import Dict, Callable, Optional, Any
import copy

from kolpinn.grids import Subgrid
from kolpinn import quantities
from kolpinn.quantities import QuantityDict
from kolpinn import model
from kolpinn import training

import parameters as params
import physics
from classes import Device
import grid_construction
from constant_models_construction import get_constant_models
from trained_models_construction import get_trained_models
from dependent_models_construction import get_dependent_models


def get_trainer(
    device: Device,
    *,
    batch_sizes: Dict[str, int],
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
) -> training.Trainer:
    dx_dict = {"": 0.0}
    if params.fd_first_derivatives:
        dx_dict["_mdx"] = -physics.dx
        dx_dict["_pdx"] = physics.dx

    unbatched_grids = grid_construction.get_unbatched_grids(
        device,
        V_min=physics.VOLTAGE_MIN,
        V_max=physics.VOLTAGE_MAX,
        V_step=physics.VOLTAGE_STEP,
        E_min=physics.E_MIN,
        E_max=physics.E_MAX,
        E_step=physics.E_STEP,
        N_x=params.N_x,
        dx_dict=dx_dict,
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
    )
    trained_models, trained_models_labels = get_trained_models(
        device,
        dx_dict=dx_dict,
    )
    dependent_models, loss_quantities = get_dependent_models(
        device,
        dx_dict=dx_dict,
    )

    const_qs = model.get_qs(
        unbatched_grids,
        constant_models,
        quantities_requiring_grad,
    )

    def get_batched_qs(full_qs: Dict[str, QuantityDict]) -> Dict[str, QuantityDict]:
        batched_grids = grid_construction.get_batched_grids(
            unbatched_grids,
            batch_sizes=batch_sizes,
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

    config = training.TrainerConfig(
        get_batched_qs=get_batched_qs,
        loss_quantities=loss_quantities,
        loss_aggregate_function=loss_aggregate_function,
        saved_parameters_index=saved_parameters_index,
        save_optimizer=save_optimizer,
        max_n_steps=max_n_steps,
        max_time=max_time,
        min_loss=min_loss,
        optimizer_reset_tol=optimizer_reset_tol,
    )
    optimizer = training.get_optimizer(
        Optimizer,
        trained_models,
        **optimizer_kwargs,
    )
    scheduler = training.get_scheduler(
        Scheduler,
        optimizer,
        **scheduler_kwargs,
    )
    state = training.TrainerState(
        const_qs,
        trained_models,
        dependent_models,
        optimizer,
        scheduler,
    )
    trainer = training.Trainer(state, config)

    return trainer
