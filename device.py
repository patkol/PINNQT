# IDEA: properly seperate self.models & trainer_models
# OPTIM: set create_graph = False where possible


from typing import Optional
import copy
import os
from pathlib import Path
import itertools
import numpy as np
import torch

from kolpinn import mathematics
from kolpinn import io
from kolpinn import grid_quantities
from kolpinn.grid_quantities import Grid, Subgrid, QuantityDict, \
                                    restrict_quantities, get_fd_derivative
from kolpinn.batching import Batcher
from kolpinn.model import Model, SimpleNNModel, ConstModel, FunctionModel, \
                          TransformedModel, get_model, get_extended_q_batchwise
from kolpinn.training import Trainer

import parameters as params
import physics
import loss



def get_values_at(
        dimensions_to_replace: dict[str,torch.Tensor],
        model: Model,
        q: QuantityDict,
        passed_quantity_labels: Optional[list[str]] = None,
    ) -> torch.Tensor:
    """Values in a modified grid"""

    if passed_quantity_labels is None:
        passed_quantity_labels = []

    dimensions = copy.copy(q.grid.dimensions)
    dimensions.update(dimensions_to_replace)
    grid = Grid(dimensions)
    new_q = physics.quantities_factory(grid)

    # Provide the quantities that certainly do not depend on the replaced
    # dimensions to the model.
    # The `passed_quantity_labels` are provided even if they might depend on
    # replaced dimension.
    for label, quantity in q.items():
        skip = False
        for replaced_dim_label in dimensions_to_replace:
            if grid_quantities.might_depend_on(replaced_dim_label, quantity, q.grid):
                skip = True
                break
        if label in new_q:
            skip = True
        if label in passed_quantity_labels:
            skip = False
        if skip:
            continue

        new_q[label] = quantity

    return model.apply(new_q)[0]

def get_model_at(
        dimensions_to_replace: dict[str,torch.Tensor],
        model: Model,
        *,
        passed_quantity_labels: Optional[list[str]] = None,
    ) -> FunctionModel:
    """
    Get a new model that corresponds to `model` but is evaluated on
    a modified grid.
    """
    return FunctionModel(
        lambda q, *, dimensions_to_replace, model, passed_quantity_labels: \
            get_values_at(dimensions_to_replace, model, q, passed_quantity_labels),
        dimensions_to_replace = dimensions_to_replace,
        model = model,
        passed_quantity_labels = passed_quantity_labels,
    )

gaussian = lambda x, sigma: torch.exp(-x**2 / (2 * sigma**2))

smoother_function = lambda x: \
    x * (1 - gaussian(x, physics.smoothing_range))

smooth_k_left_function = lambda q, i: \
    physics.k_function(
        q[f'm_eff{i}_left'],
        smoother_function(q['E']-q[f'V{i}_left']),
    )

smooth_k_right_function = lambda q, i: \
    physics.k_function(
        q[f'm_eff{i}_right'],
        smoother_function(q['E']-q[f'V{i}_right']),
    )

def transition_function(a, b, x):
    """Smoothly transition from a at x=0 to b at x->inf."""
    exp = torch.exp(-x / physics.transition_distance)
    return exp * a + (1-exp) * b

def get_a_left(q, i):
    if i==1:
        return q[f'a_output{i}_left']

    m_L = q[f'm_eff{i-1}_right']
    m_R = q[f'm_eff{i}_left']
    k_L = smooth_k_right_function(q, i-1)
    k_R = smooth_k_left_function(q, i)
    z_a = 1j * k_R + q[f'a_output_dx{i}_left']
    z_b = 1j * k_R - q[f'b_output_dx{i}_left']
    a_L = q[f'a{i-1}_right_propagated']
    b_L = q[f'b{i-1}_right_propagated']
    a_dx_L = q[f'a_dx{i-1}_right_propagated']
    b_dx_L = q[f'b_dx{i-1}_right_propagated']

    return ((a_L + b_L
             + m_R / m_L / z_b * (a_dx_L + b_dx_L + 1j * k_L * (a_L - b_L)))
            / (1 + z_a / z_b))

def get_b_left(q, i):
    if i==1:
        return q[f'b_output{i}_left']

    return q[f'a{i-1}_right_propagated'] + q[f'b{i-1}_right_propagated'] - q[f'a{i}_left']

def get_derivative_at(
        dimensions_to_replace,
        quantity_label,
        dim_label,
        q,
        **kwargs,
    ):
    """Derive by a replaced dimension"""

    grad = mathematics.grad(
        q[quantity_label],
        dimensions_to_replace[dim_label],
        **kwargs,
    )
    dimensions = copy.copy(q.grid.dimensions)
    dimensions.update(dimensions_to_replace)
    grad = grid_quantities.unsqueeze_to(
        Grid(dimensions),
        grad,
        [dim_label],
    )

    return grad


class Device:
    def __init__(
            self,
            boundaries,
            potentials,
            m_effs,
        ):
        """
        boundaries: [x_b0, ..., x_bN] with N the number of layers
        potentials: [V_0, ..., V_N+1] (including contacts),
                    constants or functions of q, grid
        m_effs: [m_0, ..., m_N+1], like potentials
        Layer i in [1,N] has x_b(i-1) on the left and x_bi on the right.
        """

        N = len(potentials)-2
        assert len(m_effs) == N+2
        assert len(boundaries) == N+1
        assert sorted(boundaries) == boundaries, boundaries

        self.n_layers = N
        self.boundaries = boundaries
        self.potentials = potentials
        self.m_effs = m_effs

        self.models = {} # Shared by multiple trainers
        self.grids_training = {}
        self.grids_validation = {}
        self.loss_functions = {}
        self.trainers = {}

        saved_parameters_index = io.get_next_parameters_index()
        print('saved_parameters_index =', saved_parameters_index)

        energies = torch.arange(
            physics.E_MIN,
            physics.E_MAX,
            physics.E_STEP,
            dtype=params.si_real_dtype,
        )

        if params.loaded_parameters_index is not None and not params.continuous_energy:
            parameters_path = io.get_parameters_path(params.loaded_parameters_index)
            saved_energies = sorted([float(Path(s).stem[2:]) * physics.EV
                                     for s in os.listdir(parameters_path)])
            energies = torch.tensor(saved_energies, dtype=params.si_real_dtype)


        # Models, Grids

        ## Layers
        for i in range(1,N+1):
            name = 'bulk' + str(i)
            x_left = self.boundaries[i-1]
            x_right = self.boundaries[i]

            self.grids_training[name] = Grid({
                'E': energies,
                'x': torch.linspace(x_left, x_right, params.N_x_training),
            })
            self.grids_validation[name] = Grid({
                'E': energies,
                'x': torch.linspace(x_left, x_right, params.N_x_validation),
            })

            se_loss_function = lambda q, with_grad, i=i: \
                loss.get_SE_loss(q, with_grad=with_grad, i=i)
            const_j_loss_function = lambda q, with_grad, i=i: \
                loss.get_const_j_loss(q, with_grad=with_grad, i=i)
            self.loss_functions[name] = {
                'SE'+str(i): se_loss_function,
                'const_j'+str(i): const_j_loss_function,
            }
            self.models[f'phi_dx_fd{i}'] = FunctionModel(
                lambda q, i=i, **kwargs: get_fd_derivative('x', q[f'phi{i}'], q.grid),
            )
            self.models[f'phi_dx{i}'] = FunctionModel(
                lambda q, i=i, **kwargs: mathematics.grad(q[f'phi{i}'], q['x'], **kwargs),
                retain_graph = True,
                create_graph = True, # OPTIM: Set to false at boundary while validating (together with with_grad)
            )
            self.models[f'a_dx{i}'] = FunctionModel(
                lambda q, i=i, **kwargs: mathematics.grad(q[f'a{i}'], q['x'], **kwargs),
                retain_graph = True,
                create_graph = True,
            )
            self.models[f'b_dx{i}'] = FunctionModel(
                lambda q, i=i, **kwargs: mathematics.grad(q[f'b{i}'], q['x'], **kwargs),
                retain_graph = True,
                create_graph = True,
            )

        ## Boundaries
        for i in range(0,N+1):
            name = 'boundary' + str(i)
            self.grids_training[name] = Grid({
                'E': energies,
                'x': torch.tensor(
                         [self.boundaries[i]],
                         dtype=params.si_real_dtype,
                     ),
            })
            self.grids_validation[name] = Grid({
                'E': energies,
                'x': torch.tensor(
                         [self.boundaries[i]],
                         dtype=params.si_real_dtype,
                     ),
            })

            self.loss_functions[name] = {}
            if not i in (0,N) and not params.model_ab:
                # The wave function is continuous on the left and right
                # by construction
                wc_loss_function = lambda q, with_grad, i=i: \
                    loss.get_wc_loss(q, with_grad=with_grad, i=i)
                self.loss_functions[name]['wc'+str(i)] = wc_loss_function
            if i==0 or i==N or not params.model_ab:
                cc_loss_function = lambda q, with_grad, i=i, N=N: \
                    loss.get_cc_loss(q, with_grad=with_grad, i=i, N=N)
                self.loss_functions[name]['cc'+str(i)] = cc_loss_function

        ## Layers and contacts
        for i in range(0,N+2):
            self.models[f'V{i}'] = get_model(
                potentials[i],
                model_dtype = params.si_real_dtype,
                output_dtype = params.si_real_dtype,
            )
            self.models[f'm_eff{i}'] = get_model(
                m_effs[i],
                model_dtype = params.si_real_dtype,
                output_dtype = params.si_real_dtype,
            )
            k_function = lambda q, i=i: \
                physics.k_function(q['m_eff'+str(i)], q['E']-q['V'+str(i)])
            self.models['k'+str(i)] = FunctionModel(k_function)

            # smooth_k: Fixing the non-smoothness of k in V at E=V
            smooth_k_function = lambda q, i=i: \
                physics.k_function(
                    q['m_eff'+str(i)],
                    smoother_function(q['E']-q['V'+str(i)]),
                )
            self.models['smooth_k'+str(i)] = FunctionModel(smooth_k_function)


        # Trainers

        if params.continuous_energy:
            #energy_strings = [f'{energies[0]/physics.EV:.16e}_to_{energies[.1]/physics.EV:.16e}']
            energy_strings = [f'all_energies']
            energy_subgrids_training_list = [self.grids_training]
            energy_subgrids_validation_list = [self.grids_validation]
        else:
            energy_strings = []
            energy_subgrids_training_list = []
            energy_subgrids_validation_list = []

            for energy in energies.cpu().numpy():
                energy_strings.append(f'E={energy/physics.EV:.16e}')
                energy_subgrids_training_list.append(dict(
                    (label, grid.get_subgrid({'E': lambda E: E == energy},
                                             copy_all = True))
                    for label, grid in self.grids_training.items()
                ))
                energy_subgrids_validation_list.append(dict(
                    (label, grid.get_subgrid({'E': lambda E: E == energy},
                                             copy_all = True))
                    for label, grid in self.grids_validation.items()
                ))

        for energy_string, energy_subgrids_training, energy_subgrids_validation \
                in zip(energy_strings, energy_subgrids_training_list, energy_subgrids_validation_list):
            qs_training = physics.quantities_factory.get_quantities_dict(energy_subgrids_training)
            qs_validation = physics.quantities_factory.get_quantities_dict(energy_subgrids_validation)

            batchers_training = {}
            batchers_validation = {}
            trainer_models = {}
            models_dict = {}
            trained_models_labels = []

            trainer_models['m_eff0_right'] = self.models['m_eff0']
            trainer_models['V0_right'] = self.models['V0']

            ## Layers
            for i in range(1,N+1):
                name = 'bulk' + str(i)
                x_left = self.boundaries[i-1]
                x_right = self.boundaries[i]

                batchers_training[name] = Batcher(
                    qs_training[name],
                    energy_subgrids_training[name],
                    ['x'],
                    [params.batch_size_x],
                )
                batchers_validation[name] = Batcher(
                    qs_validation[name],
                    energy_subgrids_validation[name],
                    ['x'],
                    [params.batch_size_x],
                )

                inputs_labels = ['x']
                if params.continuous_energy:
                    inputs_labels.append('E')
                nn_model = SimpleNNModel(
                    inputs_labels,
                    params.activation_function,
                    n_neurons_per_hidden_layer = params.n_neurons_per_hidden_layer,
                    n_hidden_layers = params.n_hidden_layers,
                    model_dtype = params.model_dtype,
                    output_dtype = params.si_complex_dtype,
                    device = params.device,
                )
                model_transformations = {
                    'x': lambda x, q, x_left=x_left, x_right=x_right:
                             (x - x_left) / (x_right - x_left),
                    'E': lambda E, q: E / physics.EV,
                }
                if params.model_ab:
                    nn_model2 = SimpleNNModel(
                        inputs_labels,
                        params.activation_function,
                        n_neurons_per_hidden_layer = params.n_neurons_per_hidden_layer,
                        n_hidden_layers = params.n_hidden_layers,
                        model_dtype = params.model_dtype,
                        output_dtype = params.si_complex_dtype,
                        device = params.device,
                    )

                    # OPTIM: require_grad only in training passes
                    left_dimensions = {
                        'x': torch.tensor([x_left], dtype=params.si_real_dtype, requires_grad=True),
                    }
                    right_dimensions = {
                        'x': torch.tensor([x_right], dtype=params.si_real_dtype, requires_grad=True),
                    }
                    trainer_models[f'm_eff{i}_left'] = get_model_at(
                        left_dimensions,
                        self.models[f'm_eff{i}'],
                    )
                    trainer_models[f'm_eff{i}_right'] = get_model_at(
                        right_dimensions,
                        self.models[f'm_eff{i}'],
                    )
                    trainer_models[f'V{i}_left'] = get_model_at(
                        left_dimensions,
                        self.models[f'V{i}'],
                    )
                    trainer_models[f'V{i}_right'] = get_model_at(
                        right_dimensions,
                        self.models[f'V{i}'],
                    )

                    if i == 1:
                        output_transformation = lambda o, q: o
                    else:
                        output_transformation = lambda o, q, x_left=x_left: \
                            transition_function(1, o, q['x']-x_left)
                    trainer_models[f'a_output{i}'] = TransformedModel(
                        nn_model,
                        input_transformations = model_transformations,
                        output_transformation = output_transformation,
                    )
                    trained_models_labels.append(f'a_output{i}')
                    trainer_models[f'b_output{i}'] = TransformedModel(
                        nn_model2,
                        input_transformations = model_transformations,
                        output_transformation = output_transformation,
                    )
                    trained_models_labels.append(f'b_output{i}')
                    trainer_models[f'a_output{i}_left'] = get_model_at(
                        left_dimensions,
                        trainer_models[f'a_output{i}'],
                    )
                    trainer_models[f'b_output{i}_left'] = get_model_at(
                        left_dimensions,
                        trainer_models[f'b_output{i}'],
                    )
                    trainer_models[f'a_output{i}_right'] = get_model_at(
                        right_dimensions,
                        trainer_models[f'a_output{i}'],
                    )
                    trainer_models[f'b_output{i}_right'] = get_model_at(
                        right_dimensions,
                        trainer_models[f'b_output{i}'],
                    )

                    trainer_models[f'a_output_dx{i}_left'] = FunctionModel(
                        lambda q, i=i, left_dimensions=left_dimensions, **kwargs:
                            get_derivative_at(
                                left_dimensions,
                                f'a_output{i}_left',
                                'x',
                                q,
                                **kwargs,
                            ),
                        retain_graph = True,
                        create_graph = True,
                    )
                    trainer_models[f'b_output_dx{i}_left'] = FunctionModel(
                        lambda q, i=i, left_dimensions=left_dimensions, **kwargs:
                            get_derivative_at(
                                left_dimensions,
                                f'b_output{i}_left',
                                'x',
                                q,
                                **kwargs,
                            ),
                        retain_graph = True,
                        create_graph = True,
                    )

                    trainer_models[f'a{i}_left'] = FunctionModel(
                        lambda q, i=i: get_a_left(q, i),
                        output_dtype = params.si_complex_dtype,
                    )
                    trainer_models[f'b{i}_left'] = FunctionModel(
                        lambda q, i=i: get_b_left(q, i),
                        output_dtype = params.si_complex_dtype,
                    )

                    if i == 1:
                        trainer_models[f'a{i}'] = trainer_models[f'a_output{i}']
                        trainer_models[f'b{i}'] = trainer_models[f'b_output{i}']
                        trainer_models[f'a{i}_right'] = trainer_models[f'a_output{i}_right']
                        trainer_models[f'b{i}_right'] = trainer_models[f'b_output{i}_right']
                    else:
                        trainer_models[f'a{i}'] = FunctionModel(
                            lambda q, i=i: q[f'a_output{i}'] * q[f'a{i}_left'], # Duplicated in a_right_propagated below
                        )
                        trainer_models[f'b{i}'] = FunctionModel(
                            lambda q, i=i: q[f'b_output{i}'] * q[f'b{i}_left'],
                        )
                        trainer_models[f'a{i}_right'] = FunctionModel(
                            lambda q, i=i: q[f'a_output{i}_right'] * q[f'a{i}_left'],
                        )
                        trainer_models[f'b{i}_right'] = FunctionModel(
                            lambda q, i=i: q[f'b_output{i}_right'] * q[f'b{i}_left'],
                        )

                    trainer_models[f'a{i}_right_propagated'] = FunctionModel(
                        lambda q, i=i, x_left=x_left, x_right=x_right:
                            (q[f'a{i}_right']
                             * torch.exp(1j * smooth_k_right_function(q, i) * (x_right - x_left))),
                    )
                    trainer_models[f'b{i}_right_propagated'] = FunctionModel(
                        lambda q, i=i, x_left=x_left, x_right=x_right:
                            (q[f'b{i}_right']
                             * torch.exp(-1j * smooth_k_right_function(q, i) * (x_right - x_left))),
                    )
                    trainer_models[f'a_dx{i}_right_propagated'] = FunctionModel(
                        lambda q, i=i, right_dimensions=right_dimensions, **kwargs:
                            get_derivative_at(
                                right_dimensions,
                                f'a{i}_right_propagated',
                                'x',
                                q,
                                **kwargs,
                            ),
                        retain_graph = True,
                        create_graph = True,
                    )
                    trainer_models[f'b_dx{i}_right_propagated'] = FunctionModel(
                        lambda q, i=i, right_dimensions=right_dimensions, **kwargs:
                            get_derivative_at(
                                right_dimensions,
                                f'b{i}_right_propagated',
                                'x',
                                q,
                                **kwargs,
                            ),
                        retain_graph = True,
                        create_graph = True,
                    )
                    # The shifts by x_left/x_right are important for
                    # energies smaller than V, it keeps them from exploding.
                    trainer_models[f'phi{i}'] = FunctionModel(
                        lambda q, i=i, x_left=x_left, x_right=x_right: (
                            q[f'a{i}'] * torch.exp(1j * q[f'smooth_k{i}'] * (q['x'] - x_left))
                            + q[f'b{i}'] * torch.exp(-1j * q[f'smooth_k{i}'] * (q['x'] - x_left)) # Explodes for large layers
                        ),
                        output_dtype = params.si_complex_dtype,
                    )
                else:
                    trainer_models['phi' + str(i)] = TransformedModel(
                        nn_model,
                        input_transformations = model_transformations,
                    )
                    trained_models_labels.append('phi' + str(i))

                models_dict[name] = {
                    f'V{i}': self.models[f'V{i}'],
                    f'm_eff{i}': self.models[f'm_eff{i}'],
                }
                if params.model_ab:
                    for j in range(1,i+1):
                        models_dict[name][f'm_eff{j-1}_right'] = trainer_models[f'm_eff{j-1}_right']
                        models_dict[name][f'm_eff{j}_left'] = trainer_models[f'm_eff{j}_left']
                        models_dict[name][f'V{j-1}_right'] = trainer_models[f'V{j-1}_right']
                        models_dict[name][f'V{j}_left'] = trainer_models[f'V{j}_left']
                        if j > 1:
                            models_dict[name][f'a_output{j-1}_right'] = trainer_models[f'a_output{j-1}_right']
                            models_dict[name][f'b_output{j-1}_right'] = trainer_models[f'b_output{j-1}_right']
                            models_dict[name][f'a{j-1}_right'] = trainer_models[f'a{j-1}_right']
                            models_dict[name][f'b{j-1}_right'] = trainer_models[f'b{j-1}_right']
                            models_dict[name][f'a{j-1}_right_propagated'] = trainer_models[f'a{j-1}_right_propagated']
                            models_dict[name][f'b{j-1}_right_propagated'] = trainer_models[f'b{j-1}_right_propagated']
                            models_dict[name][f'a_dx{j-1}_right_propagated'] = trainer_models[f'a_dx{j-1}_right_propagated']
                            models_dict[name][f'b_dx{j-1}_right_propagated'] = trainer_models[f'b_dx{j-1}_right_propagated']
                        models_dict[name][f'a_output{j}'] = trainer_models[f'a_output{j}']
                        models_dict[name][f'b_output{j}'] = trainer_models[f'b_output{j}']
                        models_dict[name][f'a_output{j}_left'] = trainer_models[f'a_output{j}_left']
                        models_dict[name][f'b_output{j}_left'] = trainer_models[f'b_output{j}_left']
                        models_dict[name][f'a_output_dx{j}_left'] = trainer_models[f'a_output_dx{j}_left']
                        models_dict[name][f'b_output_dx{j}_left'] = trainer_models[f'b_output_dx{j}_left']
                        models_dict[name][f'a{j}_left'] = trainer_models[f'a{j}_left']
                        models_dict[name][f'b{j}_left'] = trainer_models[f'b{j}_left']
                    models_dict[name][f'a{i}'] = trainer_models[f'a{i}']
                    models_dict[name][f'b{i}'] = trainer_models[f'b{i}']
                    models_dict[name][f'smooth_k{i}'] = self.models[f'smooth_k{i}']
                # Can depend on a, b, k
                models_dict[name][f'phi{i}'] = trainer_models[f'phi{i}']
                if params.fd_first_derivatives:
                    models_dict[name][f'phi_dx{i}'] = self.models[f'phi_dx_fd{i}']
                else:
                    models_dict[name][f'phi_dx{i}'] = self.models[f'phi_dx{i}']

            ## Boundaries
            for i in range(0,N+1):
                name = 'boundary' + str(i)
                batchers_training[name] = Batcher(
                    qs_training[name],
                    energy_subgrids_training[name],
                    [],
                    [],
                )
                batchers_validation[name] = Batcher(
                    qs_validation[name],
                    energy_subgrids_validation[name],
                    [],
                    [],
                )
                models_dict[name] = {
                    'V' + str(i): self.models['V' + str(i)],
                    'V' + str(i+1): self.models['V' + str(i+1)],
                    'm_eff' + str(i): self.models['m_eff' + str(i)],
                    'm_eff' + str(i+1): self.models['m_eff' + str(i+1)],
                }

                neighbour_layers = [i, i+1]
                if i == 0:
                    models_dict[name]['k' + str(i)] = self.models['k' + str(i)]
                    neighbour_layers.remove(0)
                if i == N:
                    models_dict[name]['k' + str(i+1)] = self.models['k' + str(i+1)]
                    neighbour_layers.remove(N+1)
                for n in neighbour_layers:
                    if params.model_ab:
                        for j in range(1,n+1):
                            models_dict[name][f'm_eff{j-1}_right'] = trainer_models[f'm_eff{j-1}_right']
                            models_dict[name][f'm_eff{j}_left'] = trainer_models[f'm_eff{j}_left']
                            models_dict[name][f'V{j-1}_right'] = trainer_models[f'V{j-1}_right']
                            models_dict[name][f'V{j}_left'] = trainer_models[f'V{j}_left']
                            if j > 1:
                                models_dict[name][f'a_output{j-1}_right'] = trainer_models[f'a_output{j-1}_right']
                                models_dict[name][f'b_output{j-1}_right'] = trainer_models[f'b_output{j-1}_right']
                                models_dict[name][f'a{j-1}_right'] = trainer_models[f'a{j-1}_right']
                                models_dict[name][f'b{j-1}_right'] = trainer_models[f'b{j-1}_right']
                                models_dict[name][f'a{j-1}_right_propagated'] = trainer_models[f'a{j-1}_right_propagated']
                                models_dict[name][f'b{j-1}_right_propagated'] = trainer_models[f'b{j-1}_right_propagated']
                                models_dict[name][f'a_dx{j-1}_right_propagated'] = trainer_models[f'a_dx{j-1}_right_propagated']
                                models_dict[name][f'b_dx{j-1}_right_propagated'] = trainer_models[f'b_dx{j-1}_right_propagated']
                            models_dict[name][f'a_output{j}'] = trainer_models[f'a_output{j}']
                            models_dict[name][f'b_output{j}'] = trainer_models[f'b_output{j}']
                            models_dict[name][f'a_output{j}_left'] = trainer_models[f'a_output{j}_left']
                            models_dict[name][f'b_output{j}_left'] = trainer_models[f'b_output{j}_left']
                            models_dict[name][f'a_output_dx{j}_left'] = trainer_models[f'a_output_dx{j}_left']
                            models_dict[name][f'b_output_dx{j}_left'] = trainer_models[f'b_output_dx{j}_left']
                            models_dict[name][f'a{j}_left'] = trainer_models[f'a{j}_left']
                            models_dict[name][f'b{j}_left'] = trainer_models[f'b{j}_left']
                        models_dict[name][f'a{n}'] = trainer_models[f'a{n}']
                        models_dict[name][f'b{n}'] = trainer_models[f'b{n}']
                        models_dict[name][f'smooth_k{n}'] = self.models[f'smooth_k{n}']
                    models_dict[name][f'phi{n}'] = trainer_models[f'phi{n}']
                    # Always using the exact derivatives on the boundaries
                    models_dict[name][f'phi_dx{n}'] = self.models[f'phi_dx{n}']


            # Add the loss models

            used_losses = {}
            quantities_requiring_grad_dict = {}
            for batcher_name, loss_functions_dict in self.loss_functions.items():
                # OPTIM: Use fd on boundary and require no grad
                quantities_requiring_grad_dict[batcher_name] = ['x']
                used_losses[batcher_name] = []
                for loss_name, loss_function in loss_functions_dict.items():
                    loss_model = FunctionModel(loss_function, with_grad = True)
                    models_dict[batcher_name][loss_name] = loss_model
                    used_losses[batcher_name].append(loss_name)


            # Trainer

            self.trainers[energy_string] = Trainer(
                models_dict = models_dict,
                batchers_training = batchers_training,
                batchers_validation = batchers_validation,
                used_losses = used_losses,
                quantities_requiring_grad_dict = quantities_requiring_grad_dict,
                trained_models_labels = trained_models_labels,
                Optimizer = params.Optimizer,
                optimizer_kwargs = params.optimizer_kwargs,
                Scheduler = params.Scheduler,
                scheduler_kwargs = params.scheduler_kwargs,
                saved_parameters_index = saved_parameters_index,
                name = energy_string,
            )
            self.trainers[energy_string].load(
                params.loaded_parameters_index,
                load_optimizer = params.load_optimizer,
                load_scheduler = params.load_scheduler,
            )


    def get_extended_qs(self):
        qs_list = []
        for trainer in self.trainers.values():
            qs_list.append(trainer.get_extended_qs())

        combined_qs = {}
        for grid_name, grid in self.grids_validation.items():
            q_list = [qs[grid_name] for qs in qs_list]
            combined_qs[grid_name] = grid_quantities.combine_quantities(q_list, grid)

        return combined_qs
