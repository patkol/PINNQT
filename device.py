import os
from pathlib import Path
import itertools
import torch

from kolpinn import mathematics
from kolpinn import storage
from kolpinn import grid_quantities
from kolpinn.grid_quantities import Grid, QuantityDict, get_fd_derivative
from kolpinn.batching import Batcher
from kolpinn.model import SimpleNNModel, ConstModel, FunctionModel, \
                          TransformedModel, get_model, \
                          MultiModel, get_multi_model, get_multi_models, \
                          get_combined_multi_model, get_qs
from kolpinn.training import Trainer

import parameters as params
import physics
import loss


dx_strings = ['']
dx_shifts = [0]
if params.fd_first_derivatives:
    dx_strings += ['_pdx', '_mdx']
    dx_shifts += [physics.dx, -physics.dx]

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

def j_exact_qs_trafo(qs):
    k_left = qs['boundary0']['k0']
    m_eff_left = qs['boundary0']['m_eff0']
    j_exact = -physics.H_BAR * k_left / m_eff_left
    for q in qs.values():
        q['j_exact'] = j_exact

    return qs

def transition_function(a, b, transition_exp):
    """
    Smoothly transition from a at x=0 to b at x->inf.
    transition_exp(x) = torch.exp(-(x-x_left) / transition_distance)
    """
    return transition_exp * a + (1-transition_exp) * b

def get_dx_model(mode, quantity_name, grid_name):
    name = quantity_name + '_dx'
    if mode == 'exact':
        dx_model_single = FunctionModel(
            lambda q, *, q_full, with_grad: mathematics.grad(
                q[quantity_name],
                q_full['x'],
                retain_graph=True,  # OPTIM: not always necessary
                create_graph=True,  # OPTIM: not always necessary
            ),
            q_full = None,
            with_grad = True,
        )

        return get_multi_model(dx_model_single, name, grid_name)

    if mode == 'multigrid':
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

def get_a_left(q, i, N, constant_outputs=False):
    """ Return a at 'boundary{i-1}' """

    m_L = q[f'm_eff{i-1}']
    m_R = q[f'm_eff{i}']
    k_L = q[f'k{i-1}'] if i==1 else q[f'smooth_k{i-1}']
    k_R = q[f'k{i}'] if i==N+1 else q[f'smooth_k{i}']
    z_a = 1j * k_R
    z_b = 1j * k_R
    if not constant_outputs:
        z_a += q[f'a_output{i}_dx']
        z_b -= q[f'b_output{i}_dx']
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

    a_L = q[f'a{i-1}_propagated']
    b_L = q[f'b{i-1}_propagated']
    a_R = q[f'a{i}']

    b_R = a_L + b_L - a_R

    return b_R

def add_coeff(c: str, qs: dict[str,QuantityDict], grid_name: str, i: int):
    left_q = qs[f'boundary{i-1}']
    q = qs[grid_name]
    q[f'{c}{i}'] = left_q[f'{c}{i}'] * q[f'{c}_output{i}']

    return qs

def add_coeffs(qs: dict[str,QuantityDict], grid_name: str, i: int):
    for c in ['a', 'b']:
        add_coeff(c, qs, grid_name, i)
    return qs

def b_oom_trafo(qs, N: int):
    qs['boundary0']['b_oom0'] = torch.ones(
        (1,) * qs['boundary0'].grid.n_dim, 
        dtype=params.si_real_dtype,
    )
    for i in range(1, N+1):
        q = qs[f'bulk{i}']
        q_left = qs[f'boundary{i-1}']
        q_right = qs[f'boundary{i}']

        q[f'b_oom{i}'] = q_left[f'b_oom{i-1}'] * torch.abs(q[f'b_phase{i}'])
        q_right[f'b_oom{i}'] = q_left[f'b_oom{i-1}'] * torch.abs(q_right[f'b_phase{i}'])

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
        device_thickness = boundaries[-1] - boundaries[0]
        assert len(m_effs) == N+2
        assert len(boundaries) == N+1
        assert sorted(boundaries) == boundaries, boundaries

        self.n_layers = N
        self.boundaries = boundaries
        self.potentials = potentials
        self.m_effs = m_effs

        self.loss_functions = {}
        self.trainers = {}

        saved_parameters_index = storage.get_next_parameters_index()
        print('saved_parameters_index =', saved_parameters_index)

        energies = torch.arange(
            physics.E_MIN,
            physics.E_MAX,
            physics.E_STEP,
            dtype=params.si_real_dtype,
        )
        voltages = torch.arange(
            physics.VOLTAGE_MIN,
            physics.VOLTAGE_MAX,
            physics.VOLTAGE_STEP,
            dtype=params.si_real_dtype,
        )

        if params.loaded_parameters_index is not None and not params.continuous_energy:
            parameters_path = storage.get_parameters_path(params.loaded_parameters_index)
            saved_energies = sorted([float(Path(s).stem[2:]) * physics.EV
                                     for s in os.listdir(parameters_path)])
            energies = torch.tensor(saved_energies, dtype=params.si_real_dtype)


        # Grids

        self.grids_training = {}
        self.grids_validation = {}

        ## Layers
        for i in range(1,N+1):
            grid_name = f'bulk{i}'
            x_left = self.boundaries[i-1]
            x_right = self.boundaries[i]

            self.grids_training[grid_name] = Grid({
                'voltage': voltages,
                'E': energies,
                'x': torch.linspace(x_left, x_right, params.N_x_training),
            })
            self.grids_validation[grid_name] = Grid({
                'voltage': voltages,
                'E': energies,
                'x': torch.linspace(x_left, x_right, params.N_x_validation),
            })

        self.n_dim = next(iter(self.grids_training.values())).n_dim

        ## Boundaries
        for i in range(0,N+1):
            for dx_string, dx_shift in zip(dx_strings, dx_shifts):
                grid_name = f'boundary{i}' + dx_string
                x = self.boundaries[i] + dx_shift
                self.grids_training[grid_name] = Grid({
                    'voltage': voltages,
                    'E': energies,
                    'x': torch.tensor([x], dtype=params.si_real_dtype),
                })
                self.grids_validation[grid_name] = Grid({
                    'voltage': voltages,
                    'E': energies,
                    'x': torch.tensor([x], dtype=params.si_real_dtype),
                })

        quantities_requiring_grad = {}
        for grid_name in self.grids_training:
            quantities_requiring_grad[grid_name] = []
            if not params.fd_first_derivatives or not params.fd_second_derivatives:
                quantities_requiring_grad[grid_name].append('x')


        # Constant models

        # const_models_dict[i][name] = model
        const_models_dict = dict((i,{}) for i in range(0,N+2))

        ## Layers and contacts
        for i, models_dict in const_models_dict.items():
            models_dict[f'V_no_voltage{i}'] = get_model(
                potentials[i],
                model_dtype = params.si_real_dtype,
                output_dtype = params.si_real_dtype,
            )
            models_dict[f'V{i}'] = FunctionModel(
                lambda q, i=i: (q[f'V_no_voltage{i}']
                                - q['voltage'] * physics.EV * q['x']
                                  / device_thickness),
            )
            models_dict[f'm_eff{i}'] = get_model(
                m_effs[i],
                model_dtype = params.si_real_dtype,
                output_dtype = params.si_real_dtype,
            )
            models_dict[f'k{i}'] = FunctionModel(
                lambda q, i=i: k_function(q, i),
            )

        ## Layers
        for i in range(1,N+1):
            x_left = self.boundaries[i-1]
            x_right = self.boundaries[i]
            models_dict = const_models_dict[i]
            models_dict[f'smooth_k{i}'] = FunctionModel(
                lambda q, i=i: smooth_k_function(q, i),
            )
            # The shifts by x_left/x_right are important for
            # energies smaller than V, it keeps them from exploding.
            models_dict[f'a_phase{i}'] = FunctionModel(
                lambda q, i=i, x_left=x_left:
                    torch.exp(1j * q[f'smooth_k{i}'] * (q['x'] - x_left)),
            )
            # b_phase explodes for large layers and imaginary smooth_k
            models_dict[f'b_phase{i}'] = FunctionModel(
                lambda q, i=i, x_left=x_left:
                    torch.exp(-1j * q[f'smooth_k{i}'] * (q['x'] - x_left)),
            )
            models_dict[f'a_phase{i}_propagated'] = FunctionModel(
                lambda q, i=i, x_left=x_left, x_right=x_right:
                    torch.exp(1j * q[f'smooth_k{i}'] * (x_right - x_left)),
            )
            models_dict[f'b_phase{i}_propagated'] = FunctionModel(
                lambda q, i=i, x_left=x_left, x_right=x_right:
                    torch.exp(-1j * q[f'smooth_k{i}'] * (x_right - x_left)),
            )
            models_dict[f'transition_exp{i}'] = FunctionModel(
                lambda q, x_left=x_left:
                    torch.exp(-(q['x'] - x_left) / physics.transition_distance),
            )

        ## Left boundary
        zero_model = ConstModel(0, model_dtype=params.si_real_dtype)
        one_model = ConstModel(1, model_dtype=params.si_real_dtype)
        const_models_dict[0]['a0_propagated'] = zero_model
        const_models_dict[0]['b0_propagated'] = one_model
        const_models_dict[0]['a0_propagated_dx'] = zero_model
        const_models_dict[0]['b0_propagated_dx'] = zero_model

        const_models = []

        ## Layers
        for i in range(1,N+1):
            grid_name = f'bulk{i}'
            models_dict = const_models_dict[i]
            for model_name, model in models_dict.items():
                const_models.append(get_multi_model(model, model_name, grid_name))

        ## Boundaries
        for i in range(0,N+1):
            for dx_string in dx_strings:
                grid_name = f'boundary{i}' + dx_string
                for j in (i, i+1):
                    models_dict = const_models_dict[j]
                    for model_name, model in models_dict.items():
                        const_models.append(get_multi_model(model, model_name, grid_name))

        const_models.append(MultiModel(
            j_exact_qs_trafo,
            'j_exact',
        ))
        const_models.append(MultiModel(
            lambda qs, N=N: b_oom_trafo(qs, N),
            'b_oom',
        ))


        # Parameter-dependent models shared by multiple trainers

        # shared_models_dict[name] = model
        shared_models_dict = {}

        ## Layers
        for i in range(1,N+1):
            shared_models_dict[f'a_output{i}'] = FunctionModel(
                lambda q, i=i:
                    transition_function(
                        1,
                        q[f'a_output_untransformed{i}'],
                        q[f'transition_exp{i}'],
                    )
            )
            shared_models_dict[f'b_output{i}'] = FunctionModel(
                lambda q, i=i:
                    transition_function(
                        1,
                        q[f'b_output_untransformed{i}'],
                        q[f'transition_exp{i}'],
                    )
            )
            shared_models_dict[f'a{i}_left'] = FunctionModel(
                lambda q, i=i: get_a_left(q, i, N),
            )
            shared_models_dict[f'b{i}_left'] = FunctionModel(
                lambda q, i=i: get_b_left(q, i),
            )

            shared_models_dict[f'a{i}_propagated'] = FunctionModel(
                lambda q, i=i:
                    q[f'a{i}'] * q[f'a_phase{i}_propagated']
            )
            shared_models_dict[f'b{i}_propagated'] = FunctionModel(
                lambda q, i=i:
                    q[f'b{i}'] * q[f'b_phase{i}_propagated']
            )
            shared_models_dict[f'phi{i}'] = FunctionModel(
                lambda q, i=i: (
                    q[f'a{i}'] * q[f'a_phase{i}'] + q[f'b{i}'] * q[f'b_phase{i}']
                ),
                output_dtype = params.si_complex_dtype,
            )
        
        ## Right contact
        shared_models_dict[f'a{N+1}_left'] = FunctionModel(
            lambda q: get_a_left(q, N+1, N, constant_outputs=True),
        )
        shared_models_dict[f'b{N+1}_left'] = FunctionModel(
            lambda q: get_b_left(q, N+1),
        )

        shared_models = []
        self.used_losses = {}

        # Compose `shared_models` layer by layer
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
                shared_models += get_multi_models(
                    shared_models_dict,
                    grid_name,
                    used_models_names = [
                        f'a_output{i}', f'b_output{i}',
                    ],
                )

            for c in ['a', 'b']:
                shared_models.append(get_dx_model(
                    'multigrid' if params.fd_first_derivatives else 'exact',
                    f'{c}_output{i}',
                    left_boundary_name,
                ))

            for c in ['a', 'b']:
                shared_models.append(get_multi_model(
                    shared_models_dict[f'{c}{i}_left'],
                    f'{c}{i}',
                    left_boundary_name,
                    multi_model_name = f'{c}{i}_left',
                ))

            remaining_grids = bulk_names + right_boundary_names
            if params.fd_first_derivatives:
                remaining_grids += [
                    left_boundary_name + '_pdx',
                    left_boundary_name + '_mdx',
                ]
            for grid_name in remaining_grids:
                shared_models.append(MultiModel(
                    lambda qs, grid_name=grid_name, i=i:
                        add_coeffs(qs, grid_name, i),
                    f'coeffs{i}',
                ))

            for grid_name in left_boundary_names + bulk_names + right_boundary_names:
                shared_models.append(get_multi_model(
                    shared_models_dict[f'phi{i}'],
                    f'phi{i}',
                    grid_name,
                ))


            for grid_name in right_boundary_names:
                shared_models += get_multi_models(
                    shared_models_dict,
                    grid_name,
                    used_models_names = [
                        f'a{i}_propagated', f'b{i}_propagated',
                    ],
                )

            for c in ['a', 'b']:
                shared_models.append(get_dx_model(
                    'multigrid' if params.fd_first_derivatives else 'exact',
                    f'{c}{i}_propagated',
                    right_boundary_name,
                ))

            shared_models.append(get_dx_model(
                'singlegrid' if params.fd_first_derivatives else 'exact',
                f'phi{i}',
                bulk_name,
            ))

        shared_models.append(get_dx_model(
            'multigrid' if params.fd_first_derivatives else 'exact',
            'phi1',
            'boundary0',
        ))
        shared_models.append(get_dx_model(
            'multigrid' if params.fd_first_derivatives else 'exact',
            f'phi{N}',
            f'boundary{N}',
        ))

        for c in ['a', 'b']:
            shared_models.append(get_multi_model(
                shared_models_dict[f'{c}{N+1}_left'],
                f'{c}{N+1}',
                f'boundary{N}',
                multi_model_name = f'{c}{N+1}_left',
            ))

        # Losses
        for i in range(1,N+1):
            bulk_name = f'bulk{i}'
            self.used_losses[bulk_name] = []

            shared_models.append(MultiModel(
                loss.SE_loss_trafo,
                f'SE_loss{i}',
                kwargs = {'qs_full': None, 'with_grad': True, 'i': i, 'N': N},
            ))
            self.used_losses[bulk_name].append(f'SE_loss{i}')

            shared_models.append(MultiModel(
                loss.j_loss_trafo,
                f'j_loss{i}',
                kwargs = {'i': i, 'N': N},
            ))
            self.used_losses[bulk_name].append(f'j_loss{i}')


        # Trainers

        trainer_voltages = ['all']
        trainer_energies = ['all']
        if not params.continuous_voltage:
            trainer_voltages = voltages.cpu().numpy()
        if not params.continuous_energy:
            trainer_energies = energies.cpu().numpy()

        trainer_names = []
        trainer_subgrids_training_list = []
        trainer_subgrids_validation_list = []

        for voltage, energy in itertools.product(trainer_voltages, trainer_energies):
            subgrid_dict = {}
            if voltage == 'all':
                voltage_string = 'all'
            else:
                voltage_string = f'{voltage:.16e}'
                subgrid_dict['voltage'] = lambda x, voltage=voltage: x == voltage
            if energy == 'all':
                energy_string = 'all'
            else:
                energy_string = f'{energy:.16e}'
                subgrid_dict['E'] = lambda x, energy=energy: x == energy
            trainer_names.append(f'voltage={voltage_string}_energy={energy_string}')
            trainer_subgrids_training_list.append(dict(
                (label, grid.get_subgrid(subgrid_dict, copy_all = True))
                for label, grid in self.grids_training.items()
            ))
            trainer_subgrids_validation_list.append(dict(
                (label, grid.get_subgrid(subgrid_dict, copy_all = True))
                for label, grid in self.grids_validation.items()
            ))

        for trainer_name, trainer_subgrids_training, trainer_subgrids_validation \
                in zip(trainer_names, trainer_subgrids_training_list, trainer_subgrids_validation_list):
            qs_training = get_qs(trainer_subgrids_training, const_models, quantities_requiring_grad)
            qs_validation = get_qs(trainer_subgrids_validation, const_models, quantities_requiring_grad)


            # Batchers

            batchers_training = {}
            batchers_validation = {}

            ## Layers
            for i in range(1,N+1):
                grid_name = f'bulk{i}'

                batchers_training[grid_name] = Batcher(
                    qs_training[grid_name],
                    trainer_subgrids_training[grid_name],
                    ['x'],
                    [params.batch_size_x],
                )
                batchers_validation[grid_name] = Batcher(
                    qs_validation[grid_name],
                    trainer_subgrids_validation[grid_name],
                    ['x'],
                    [params.batch_size_x],
                )

            ## Boundaries
            for i in range(0,N+1):
                for grid_name in [f'boundary{i}' + dx_string
                                  for dx_string in dx_strings]:
                    batchers_training[grid_name] = Batcher(
                        qs_training[grid_name],
                        trainer_subgrids_training[grid_name],
                        [],
                        [],
                    )
                    batchers_validation[grid_name] = Batcher(
                        qs_validation[grid_name],
                        trainer_subgrids_validation[grid_name],
                        [],
                        [],
                    )


            # Trainer-specific models

            models = []
            trained_models_labels = []

            ## Layers
            for i in range(1,N+1):
                left_boundary_name = f'boundary{i-1}'
                left_boundary_names = [left_boundary_name + dx_string
                                       for dx_string in dx_strings]
                bulk_name = f'bulk{i}'
                bulk_names = [bulk_name]
                right_boundary_name = f'boundary{i}'
                right_boundary_names = [right_boundary_name + dx_string
                                        for dx_string in dx_strings]

                inputs_labels = []
                if params.continuous_voltage:
                    inputs_labels.append('voltage')
                if params.continuous_energy:
                    inputs_labels.append('E')
                inputs_labels.append('x')
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

                nn_model2 = SimpleNNModel(
                    inputs_labels,
                    params.activation_function,
                    n_neurons_per_hidden_layer = params.n_neurons_per_hidden_layer,
                    n_hidden_layers = params.n_hidden_layers,
                    model_dtype = params.model_dtype,
                    output_dtype = params.si_complex_dtype,
                    device = params.device,
                )

                for c in ['a', 'b']:
                    c_model = TransformedModel(
                        nn_model if c=='a' else nn_model2,
                        input_transformations = model_transformations,
                    )
                    models.append(get_combined_multi_model(
                        c_model,
                        f'{c}_output_untransformed{i}',
                        left_boundary_names + bulk_names + right_boundary_names,
                        combined_dimension_name = 'x',
                        required_quantities_labels = ['voltage', 'E', 'x'],
                    ))
                    trained_models_labels.append(f'{c}_output_untransformed{i}')

            models += shared_models


            self.trainers[trainer_name] = Trainer(
                models = models,
                batchers_training = batchers_training,
                batchers_validation = batchers_validation,
                used_losses = self.used_losses,
                trained_models_labels = trained_models_labels,
                Optimizer = params.Optimizer,
                optimizer_kwargs = params.optimizer_kwargs,
                Scheduler = params.Scheduler,
                scheduler_kwargs = params.scheduler_kwargs,
                saved_parameters_index = saved_parameters_index,
                save_optimizer = params.save_optimizer,
                loss_aggregate_function = params.loss_aggregate_function,
                name = trainer_name,
            )
            self.trainers[trainer_name].load(
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
