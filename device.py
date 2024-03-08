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
                          TransformedModel, get_model, \
                          MultiModel, get_multi_model, get_multi_models
from kolpinn.training import Trainer

import parameters as params
import physics
import loss


dx_strings = ['', '_pdx', '_mdx']
dx_shifts = [0, physics.dx, -physics.dx]

k_function = lambda q, i: \
    physics.k_function(q[f'm_eff{i}'], q['E']-q[f'V{i}'])

gaussian = lambda x, sigma: torch.exp(-x**2 / (2 * sigma**2))

smoother_function = lambda x: \
    x * (1 - gaussian(x, physics.smoothing_range))

# smooth_k: Fixing the non-smoothness of k in V at E=V
smooth_k_function = lambda q, i: \
    physics.k_function(
        q[f'm_eff{i}'],
        smoother_function(q['E']-q[f'V{i}']),
    )

def transition_function(a, b, x):
    """Smoothly transition from a at x=0 to b at x->inf."""
    exp = torch.exp(-x / physics.transition_distance)
    return exp * a + (1-exp) * b

def get_dx_model(mode, quantity_name, grid_name):
    name = quantity_name + '_dx'
    if mode == 'exact':
        def dx_qs_trafo(qs):
            q = qs[grid_name]
            q[name] = mathematics.grad(
                qs[grid_name][quantity_name],
                q['x'],
                retain_graph=True,  # OPTIM: not always necessary
                create_graph=True,  # OPTIM: not always necessary
            )
            return qs

    elif mode == 'multigrid':
        def dx_qs_trafo(qs):
            quantity_right = qs[grid_name + '_pdx'][quantity_name]
            quantity_left = qs[grid_name + '_mdx'][quantity_name]
            qs[grid_name][name] = \
                (quantity_right - quantity_left) / (2 * physics.dx)
            return qs

    elif mode == 'singlegrid':
        def dx_qs_trafo(qs):
            q = qs[grid_name]
            q[name] = get_fd_derivative('x', q[quantity_name], q.grid)
            return qs

    else:
        raise Exception('Unknown dx mode:', mode)

    return MultiModel(dx_qs_trafo, name)

def get_a_left(q, i):
    """ Return a at 'boundary{i-1}' """

    if i==1:
        return q[f'a_output{i}']

    m_L = q[f'm_eff{i-1}']
    m_R = q[f'm_eff{i}']
    k_L = q[f'smooth_k{i-1}']
    k_R = q[f'smooth_k{i}']
    z_a = 1j * k_R + q[f'a_output{i}_dx']
    z_b = 1j * k_R - q[f'b_output{i}_dx']
    a_L = q[f'a{i-1}_propagated']
    b_L = q[f'b{i-1}_propagated']
    a_dx_L = q[f'a{i-1}_propagated_dx']
    b_dx_L = q[f'b{i-1}_propagated_dx']

    a_R = ((a_L + b_L
            + m_R / m_L / z_b * (a_dx_L + b_dx_L + 1j * k_L * (a_L - b_L)))
           / (1 + z_a / z_b))

    return a_R

def get_b_left(q, i):
    """ Return b at 'boundary{i-1}' """

    if i==1:
        return q[f'b_output{i}']

    a_L = q[f'a{i-1}_propagated']
    b_L = q[f'b{i-1}_propagated']
    a_R = q[f'a{i}']

    b_R = a_L + b_L - a_R

    return b_R

def add_coeff(c: str, qs: dict[str,QuantityDict], grid_name: str, i: int):
    left_q = qs[f'boundary{i-1}']
    q = qs[grid_name]
    coeff = q[f'{c}_output{i}']
    if i > 1:
        coeff = coeff * left_q[f'{c}{i}']

    q[f'{c}{i}'] = coeff

    return qs

def add_coeffs(qs: dict[str,QuantityDict], grid_name: str, i: int):
    for c in ['a', 'b']:
        add_coeff(c, qs, grid_name, i)
    return qs


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

        # Shared by multiple trainers, shared_models[model_name] = model
        self.shared_models = {}
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
            grid_name = f'bulk{i}'
            x_left = self.boundaries[i-1]
            x_right = self.boundaries[i]

            self.grids_training[grid_name] = Grid({
                'E': energies,
                'x': torch.linspace(x_left, x_right, params.N_x_training),
            })
            self.grids_validation[grid_name] = Grid({
                'E': energies,
                'x': torch.linspace(x_left, x_right, params.N_x_validation),
            })

            se_loss_function = lambda q, with_grad, i=i: \
                loss.get_SE_loss(q, with_grad=with_grad, i=i)
            const_j_loss_function = lambda q, with_grad, i=i: \
                loss.get_const_j_loss(q, with_grad=with_grad, i=i)
            self.loss_functions[grid_name] = {
                f'SE{i}': se_loss_function,
                f'const_j{i}': const_j_loss_function,
            }

            if params.model_ab:
                if i == 1:
                    self.shared_models[f'a_output{i}'] = FunctionModel(
                        lambda q, i=i: q[f'a_output_untransformed{i}']
                    )
                    self.shared_models[f'b_output{i}'] = FunctionModel(
                        lambda q, i=i: q[f'b_output_untransformed{i}']
                    )
                else:
                    self.shared_models[f'a_output{i}'] = FunctionModel(
                        lambda q, x_left=x_left, i=i:
                            transition_function(
                                1,
                                q[f'a_output_untransformed{i}'],
                                q['x']-x_left
                            )
                    )
                    self.shared_models[f'b_output{i}'] = FunctionModel(
                        lambda q, x_left=x_left, i=i:
                            transition_function(
                                1,
                                q[f'b_output_untransformed{i}'],
                                q['x']-x_left
                            )
                    )
                self.shared_models[f'a{i}_left'] = FunctionModel(
                    lambda q, i=i: get_a_left(q, i),
                )
                self.shared_models[f'b{i}_left'] = FunctionModel(
                    lambda q, i=i: get_b_left(q, i),
                )

                self.shared_models[f'a{i}_propagated'] = FunctionModel(
                    lambda q, i=i, x_left=x_left, x_right=x_right:
                        (q[f'a{i}']
                         * torch.exp(1j * q[f'smooth_k{i}'] * (x_right - x_left))),
                )
                self.shared_models[f'b{i}_propagated'] = FunctionModel(
                    lambda q, i=i, x_left=x_left, x_right=x_right:
                        (q[f'b{i}']
                         * torch.exp(-1j * q[f'smooth_k{i}'] * (x_right - x_left))),
                )
                # The shifts by x_left/x_right are important for
                # energies smaller than V, it keeps them from exploding.
                self.shared_models[f'phi{i}'] = FunctionModel(
                    lambda q, i=i, x_left=x_left, x_right=x_right: (
                        q[f'a{i}'] * torch.exp(1j * q[f'smooth_k{i}'] * (q['x'] - x_left))
                        + q[f'b{i}'] * torch.exp(-1j * q[f'smooth_k{i}'] * (q['x'] - x_left)) # Explodes for large layers
                    ),
                    output_dtype = params.si_complex_dtype,
                )

        ## Boundaries
        for i in range(0,N+1):
            grid_name = f'boundary{i}'
            for dx_string, dx_shift in zip(dx_strings, dx_shifts):
                self.grids_training[grid_name + dx_string] = Grid({
                    'E': energies,
                    'x': torch.tensor(
                             [self.boundaries[i] + dx_shift],
                             dtype=params.si_real_dtype,
                         ),
                })
                self.grids_validation[grid_name + dx_string] = Grid({
                    'E': energies,
                    'x': torch.tensor(
                             [self.boundaries[i] + dx_shift],
                             dtype=params.si_real_dtype,
                         ),
                })

                self.loss_functions[grid_name + dx_string] = {}

            if not i in (0,N) and not params.model_ab:
                # The wave function is continuous on the left and right
                # by construction
                wc_loss_function = lambda q, *, with_grad, i=i: \
                    loss.get_wc_loss(q, with_grad=with_grad, i=i)
                self.loss_functions[grid_name][f'wc{i}'] = wc_loss_function
            if i==0 or i==N or not params.model_ab:
                cc_loss_function = lambda q, *, with_grad, i=i, N=N: \
                    loss.get_cc_loss(q, with_grad=with_grad, i=i, N=N)
                self.loss_functions[grid_name][f'cc{i}'] = cc_loss_function

        ## Layers and contacts
        for i in range(0,N+2):
            self.shared_models[f'V{i}'] = get_model(
                potentials[i],
                model_dtype = params.si_real_dtype,
                output_dtype = params.si_real_dtype,
            )
            self.shared_models[f'm_eff{i}'] = get_model(
                m_effs[i],
                model_dtype = params.si_real_dtype,
                output_dtype = params.si_real_dtype,
            )
            self.shared_models[f'k{i}'] = FunctionModel(
                lambda q, i=i: k_function(q, i),
            )
            self.shared_models[f'smooth_k{i}'] = FunctionModel(
                lambda q, i=i: smooth_k_function(q, i),
            )



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
            models = []
            trained_models_labels = []


            # Layers
            for i in range(1,N+1):
                grid_name = f'bulk{i}'
                x_left = self.boundaries[i-1]
                x_right = self.boundaries[i]

                batchers_training[grid_name] = Batcher(
                    qs_training[grid_name],
                    energy_subgrids_training[grid_name],
                    ['x'],
                    [params.batch_size_x],
                )
                batchers_validation[grid_name] = Batcher(
                    qs_validation[grid_name],
                    energy_subgrids_validation[grid_name],
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

                    trainer_models[f'a_output_untransformed{i}'] = TransformedModel(
                        nn_model,
                        input_transformations = model_transformations,
                    )
                    trained_models_labels.append(f'a_output_untransformed{i}')
                    trainer_models[f'b_output_untransformed{i}'] = TransformedModel(
                        nn_model2,
                        input_transformations = model_transformations,
                    )
                    trained_models_labels.append(f'b_output_untransformed{i}')

                else:
                    trainer_models['phi' + str(i)] = TransformedModel(
                        nn_model,
                        input_transformations = model_transformations,
                    )
                    trained_models_labels.append('phi' + str(i))


            # Boundaries
            for i in range(0,N+1):
                for grid_name in [f'boundary{i}', f'boundary{i}_pdx', f'boundary{i}_mdx']:
                    batchers_training[grid_name] = Batcher(
                        qs_training[grid_name],
                        energy_subgrids_training[grid_name],
                        [],
                        [],
                    )
                    batchers_validation[grid_name] = Batcher(
                        qs_validation[grid_name],
                        energy_subgrids_validation[grid_name],
                        [],
                        [],
                    )


            for dx_string in dx_strings:
                models += get_multi_models(
                    self.shared_models,
                    'boundary0' + dx_string,
                    used_models_names = [
                        'V0', 'm_eff0', 'k0',
                    ],
                )
                models += get_multi_models(
                    self.shared_models,
                    f'boundary{N}' + dx_string,
                    used_models_names = [
                        f'V{N+1}', f'm_eff{N+1}', f'k{N+1}',
                    ],
                )

            # Compose `models` layer by layer
            for i in range(1,N+1):
                left_boundary_name = f'boundary{i-1}'
                left_boundary_names = [left_boundary_name + dx_string
                                       for dx_string in dx_strings]
                bulk_name = f'bulk{i}'
                bulk_names = [bulk_name]
                right_boundary_name = f'boundary{i}'
                right_boundary_names = [right_boundary_name + dx_string
                                        for dx_string in dx_strings]

                for grid_name in left_boundary_names + bulk_names + right_boundary_names:
                    models += get_multi_models(
                        self.shared_models,
                        grid_name,
                        used_models_names = [
                            f'V{i}', f'm_eff{i}', f'smooth_k{i}',
                        ],
                    )

                if params.model_ab:
                    for grid_name in left_boundary_names + bulk_names + right_boundary_names:
                        models += get_multi_models(
                            trainer_models,
                            grid_name,
                            used_models_names = [
                                f'a_output_untransformed{i}', f'b_output_untransformed{i}',
                            ],
                        )
                        models += get_multi_models(
                            self.shared_models,
                            grid_name,
                            used_models_names = [
                                f'a_output{i}', f'b_output{i}',
                            ],
                        )

                    models.append(get_dx_model('multigrid', f'a_output{i}', left_boundary_name))
                    models.append(get_dx_model('multigrid', f'b_output{i}', left_boundary_name))
                    models.append(get_multi_model(
                        self.shared_models[f'a{i}_left'],
                        f'a{i}',
                        left_boundary_name,
                        multi_model_name = f'a{i}_left',
                    ))
                    models.append(get_multi_model(
                        self.shared_models[f'b{i}_left'],
                        f'b{i}',
                        left_boundary_name,
                        multi_model_name = f'b{i}_left',
                    ))

                    for grid_name in ([left_boundary_name + '_pdx', left_boundary_name + '_mdx']
                                       + bulk_names + right_boundary_names):
                        models.append(MultiModel(
                            lambda qs, grid_name=grid_name, i=i:
                                add_coeffs(qs, grid_name, i),
                            f'coeffs{i}',
                        ))

                    for grid_name in right_boundary_names:
                        models += get_multi_models(
                            self.shared_models,
                            grid_name,
                            used_models_names = [
                                f'a{i}_propagated', f'b{i}_propagated',
                            ],
                        )
                    models.append(get_dx_model('multigrid', f'a{i}_propagated', right_boundary_name))
                    models.append(get_dx_model('multigrid', f'b{i}_propagated', right_boundary_name))

                for grid_name in left_boundary_names + bulk_names + right_boundary_names:
                    models += get_multi_models(
                        self.shared_models if params.model_ab else trainer_models,
                        grid_name,
                        used_models_names = [
                            f'phi{i}',
                        ],
                    )

                models.append(get_dx_model('singlegrid', f'phi{i}', bulk_name))

            models.append(get_dx_model('multigrid', 'phi1', 'boundary0'))
            models.append(get_dx_model('multigrid', f'phi{N}', f'boundary{N}'))


            # Add the loss models

            used_losses = {}
            quantities_requiring_grad_dict = {}
            for grid_name, loss_functions_dict in self.loss_functions.items():
                quantities_requiring_grad_dict[grid_name] = []
                if not params.fd_first_derivatives or not params.fd_second_derivatives:
                    quantities_requiring_grad_dict[grid_name].append('x')
                used_losses[grid_name] = []
                for loss_name, loss_function in loss_functions_dict.items():
                    loss_model = FunctionModel(loss_function, with_grad = True)
                    models.append(get_multi_model(
                        loss_model,
                        loss_name,
                        grid_name,
                    ))
                    used_losses[grid_name].append(loss_name)


            # Trainer

            self.trainers[energy_string] = Trainer(
                models = models,
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
            qs_list.append(trainer.get_extended_qs(for_training=False))

        combined_qs = {}
        for grid_name, grid in self.grids_validation.items():
            q_list = [qs[grid_name] for qs in qs_list]
            combined_qs[grid_name] = grid_quantities.combine_quantities(q_list, grid)

        return combined_qs
