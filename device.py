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

k_function = lambda q, i, contact: \
    physics.k_function(q[f'm_eff{i}'], q[f'E_{contact}']-q[f'V{i}'])

gaussian = lambda x, sigma: torch.exp(-x**2 / (2 * sigma**2))

smoother_function = lambda x, smoothing_range: \
    x * (1 - gaussian(x, smoothing_range))

# smooth_k: Fixing the non-smoothness of k in V at E=V
smooth_k_function = lambda q, i, contact: \
    physics.k_function(
        q[f'm_eff{i}'],
        smoother_function(q[f'E_{contact}']-q[f'V{i}'], physics.energy_smoothing_range),
    )

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
        raise ValueError('Unknown dx mode:', mode)

    return MultiModel(dx_qs_trafo, name)

def factors_trafo(qs, i, contact):
    """
    Use the boundary conditions to find the factors to multiply a/b_output with
    given that a and b are known in layer `i_next`
    """

    if contact == 'L':
        i_next = i+1
        i_boundary = i
    elif contact == 'R':
        i_next = i-1
        i_boundary = i-1
    else:
        raise ValueError(f'Unknown contact: {contact}')

    q = qs[f'boundary{i_boundary}']

    m = q[f'm_eff{i}']
    m_next = q[f'm_eff{i_next}']
    k = q[f'smooth_k{i}_{contact}']
    k_next = q[f'smooth_k{i_next}_{contact}']
    o_a = q[f'a_output{i}_{contact}']
    o_b = q[f'b_output{i}_{contact}']
    z_a = 1j * k * o_a + q[f'a_output{i}_{contact}_dx']
    z_b = 1j * k * o_b - q[f'b_output{i}_{contact}_dx']
    a_next = q[f'a{i_next}_propagated_{contact}']
    b_next = q[f'b{i_next}_propagated_{contact}']
    a_dx_next = q[f'a{i_next}_propagated_{contact}_dx']
    b_dx_next = q[f'b{i_next}_propagated_{contact}_dx']

    a_factor = ((a_next + b_next
                 + m / m_next * o_b / z_b 
                   * (a_dx_next + b_dx_next + 1j * k_next * (a_next - b_next)))
                / (o_a + z_a / z_b * o_b))

    b_factor = (a_next + b_next - a_factor * o_a) / o_b

    q[f'a_factor{i}_{contact}'] = a_factor
    q[f'b_factor{i}_{contact}'] = b_factor

    return qs

def add_coeff(
        c: str, 
        qs: dict[str,QuantityDict], 
        contact: str, 
        grid_name: str, 
        i: int, 
        N: int,
    ):

    if contact == 'L':
        boundary_out_index = i
    elif contact == 'R':
        boundary_out_index = i-1
    else:
        raise ValueError(f'Unknown contact: {contact}')

    boundary_q = qs[f'boundary{boundary_out_index}']
    q = qs[grid_name]

    coeff = boundary_q[f'{c}_factor{i}_{contact}'] * q[f'{c}_output{i}_{contact}']
    q[f'{c}{i}_{contact}'] = coeff

    return qs

def add_coeffs(
        qs: dict[str,QuantityDict], contact: str, grid_name: str, i: int, N: int,
    ):
    for c in ['a', 'b']:
        add_coeff(c, qs, contact, grid_name, i, N)

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

        layer_indices_dict = {'L': range(N+1,-1,-1), 'R': range(0,N+2)}


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
                'DeltaE': energies,
                'x': torch.linspace(x_left, x_right, params.N_x_training),
            })
            self.grids_validation[grid_name] = Grid({
                'voltage': voltages,
                'DeltaE': energies,
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
                    'DeltaE': energies,
                    'x': torch.tensor([x], dtype=params.si_real_dtype),
                })
                self.grids_validation[grid_name] = Grid({
                    'voltage': voltages,
                    'DeltaE': energies,
                    'x': torch.tensor([x], dtype=params.si_real_dtype),
                })

        quantities_requiring_grad = {}
        for grid_name in self.grids_training:
            quantities_requiring_grad[grid_name] = []
            if not params.fd_first_derivatives or not params.fd_second_derivatives:
                quantities_requiring_grad[grid_name].append('x')


        # Constant models

        layer_indep_const_models_dict = {}
        layer_indep_const_models_dict['E_L'] = FunctionModel(lambda q: q['DeltaE'])
        layer_indep_const_models_dict['E_R'] = FunctionModel(
            lambda q: q['DeltaE'] - physics.Q_E * q['voltage'],
        )

        # const_models_dict[i][name] = model
        const_models_dict = dict((i,{}) for i in range(0,N+2))

        ## Layers and contacts
        for i, models_dict in const_models_dict.items():
            x_left = self.boundaries[max(0,i-1)]
            x_right = self.boundaries[min(N,i)]

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

            for contact in ['L', 'R']:
                if contact == 'L':
                    x_in = x_left
                    x_out = x_right
                elif contact == 'R':
                    x_in = x_right
                    x_out = x_left
                else:
                    raise ValueError(f'Unknown contact: {contact}')

                models_dict[f'k{i}_{contact}'] = FunctionModel(
                    lambda q, i=i, contact=contact: k_function(q, i, contact),
                )
                models_dict[f'smooth_k{i}_{contact}'] = FunctionModel(
                    lambda q, i=i, contact=contact: \
                        q[f'k{i}_{contact}'] if i in (0,N+1) \
                        else smooth_k_function(q, i, contact),
                )
                # The shifts by x_out are important for
                # energies smaller than V, it keeps them from exploding.
                models_dict[f'a_phase{i}_{contact}'] = FunctionModel(
                    lambda q, i=i, x_out=x_out, contact=contact:
                        torch.exp(1j * q[f'smooth_k{i}_{contact}'] * (q['x'] - x_out)),
                )
                # b_phase explodes for large layers and imaginary smooth_k
                models_dict[f'b_phase{i}_{contact}'] = FunctionModel(
                    lambda q, i=i, x_out=x_out, contact=contact:
                        torch.exp(-1j * q[f'smooth_k{i}_{contact}'] * (q['x'] - x_out)),
                )
                models_dict[f'a_propagation_factor{i}_{contact}'] = FunctionModel(
                    lambda q, i=i, x_in=x_in, x_out=x_out, contact=contact:
                        torch.exp(1j * q[f'smooth_k{i}_{contact}'] * (x_in - x_out)),
                )
                models_dict[f'b_propagation_factor{i}_{contact}'] = FunctionModel(
                    lambda q, i=i, x_in=x_in, x_out=x_out, contact=contact:
                        torch.exp(-1j * q[f'smooth_k{i}_{contact}'] * (x_in - x_out)),
                )

        zero_model = ConstModel(0, model_dtype=params.si_real_dtype)
        one_model = ConstModel(1, model_dtype=params.si_real_dtype)

        ## Contacts
        for contact in ['L', 'R']:
            for i in (0,N+1):
                for c in ['a', 'b']:
                    const_models_dict[i][f'{c}_output{i}_{contact}'] = one_model
                    const_models_dict[i][f'{c}_output{i}_{contact}_dx'] = zero_model

        ## Output contacts: Initial conditions
        const_models_dict[N+1][f'a{N+1}_L'] = one_model
        const_models_dict[N+1][f'b{N+1}_L'] = zero_model
        const_models_dict[0]['a0_R'] = zero_model
        const_models_dict[0]['b0_R'] = one_model

        for (i, contact) in ((N+1, 'L'), (0, 'R')):
            for c in ('a', 'b'):
                const_models_dict[i][f'{c}{i}_{contact}_dx'] = zero_model
                const_models_dict[i][f'{c}{i}_propagated_{contact}'] = \
                    const_models_dict[i][f'{c}{i}_{contact}']
                const_models_dict[i][f'{c}{i}_propagated_{contact}_dx'] = \
                    const_models_dict[i][f'{c}{i}_{contact}_dx']

        const_models = []

        ## Layers
        for i in range(1,N+1):
            grid_name = f'bulk{i}'
            models_dict = dict(layer_indep_const_models_dict, **const_models_dict[i]) 
            for model_name, model in models_dict.items():
                const_models.append(get_multi_model(model, model_name, grid_name))

        ## Boundaries
        for i in range(0,N+1):
            for dx_string in dx_strings:
                grid_name = f'boundary{i}' + dx_string
                for model_name, model in layer_indep_const_models_dict.items():
                    const_models.append(get_multi_model(model, model_name, grid_name))
                for j in (i, i+1):
                    models_dict = const_models_dict[j]
                    for model_name, model in models_dict.items():
                        const_models.append(get_multi_model(model, model_name, grid_name))


        # Parameter-dependent models shared by multiple trainers

        shared_models = []
        self.used_losses = {}

        # Compose `shared_models` layer by layer
        for contact in ['L', 'R']:
            for i in layer_indices_dict[contact]:
                # The e- flow from in to out
                # We compute from out to in
                bulk = f'bulk{i}'
                boundary_left = f'boundary{i-1}'
                boundary_right = f'boundary{i}'
                is_left_contact = i == 0
                is_right_contact = i == N+1
                if contact == 'L':
                    boundary_in = boundary_left
                    boundary_out = boundary_right
                    is_in_contact = is_left_contact
                    is_out_contact = is_right_contact
                elif contact == 'R':
                    boundary_in = boundary_right
                    boundary_out = boundary_left
                    is_in_contact = is_right_contact
                    is_out_contact = is_left_contact
                else:
                    raise ValueError(f'Unknown contact: {contact}')

                is_contact = is_in_contact or is_out_contact

                boundaries_in = [] if is_in_contact else [boundary_in + dx_string
                                                          for dx_string in dx_strings]
                boundaries_out = [] if is_out_contact else [boundary_out + dx_string
                                                            for dx_string in dx_strings]
                bulks = [] if is_contact else [bulk]
                single_boundaries = []
                if not is_in_contact:
                    single_boundaries.append(boundary_in)
                if not is_out_contact:
                    single_boundaries.append(boundary_out)

                if not is_contact:
                    for c in ['a', 'b']:
                        shared_models.append(get_dx_model(
                            'multigrid' if params.fd_first_derivatives else 'exact',
                            f'{c}_output{i}_{contact}',
                            boundary_out,
                        ))

                if not is_out_contact:
                    for c in ['a', 'b']:
                        shared_models.append(MultiModel(
                            lambda qs, i=i, contact=contact:
                                factors_trafo(qs, i, contact),
                            f'factors{i}_{contact}',
                        ))

                    for grid_name in boundaries_in + bulks + boundaries_out:
                        shared_models.append(MultiModel(
                            lambda qs, contact=contact, grid_name=grid_name, i=i:
                                add_coeffs(qs, contact, grid_name, i, N),
                            f'coeffs{i}',
                        ))

                for grid_name in boundaries_in + bulks + boundaries_out:
                    shared_models.append(get_multi_model(
                        FunctionModel(
                            lambda q, i=i, contact=contact:
                                (q[f'a{i}_{contact}'] * q[f'a_phase{i}_{contact}'] 
                                 + q[f'b{i}_{contact}'] * q[f'b_phase{i}_{contact}'])
                        ),
                        f'phi{i}_{contact}',
                        grid_name,
                    ))

                for grid_name in bulks:
                    shared_models.append(get_dx_model(
                        'singlegrid' if params.fd_first_derivatives else 'exact',
                        f'phi{i}_{contact}',
                        grid_name,
                    ))

                # OPTIM: the following is only for cc loss
                for grid_name in single_boundaries:
                    shared_models.append(get_dx_model(
                        'multigrid' if params.fd_first_derivatives else 'exact',
                        f'phi{i}_{contact}',
                        grid_name,
                    ))

                if not is_contact:
                    for c in ['a', 'b']:
                        for grid_name in boundaries_in:
                            shared_models.append(get_multi_model(
                                FunctionModel(
                                    lambda q, c=c, i=i, contact=contact:
                                        q[f'{c}{i}_{contact}'] * q[f'{c}_propagation_factor{i}_{contact}']
                                ),
                                f'{c}{i}_propagated_{contact}',
                                grid_name,
                            ))

                        shared_models.append(get_dx_model(
                            'multigrid' if params.fd_first_derivatives else 'exact',
                            f'{c}{i}_propagated_{contact}',
                            boundary_in,
                        ))

        # Losses
        for i in range(1,N+1):
            bulk_name = f'bulk{i}'
            self.used_losses[bulk_name] = []

            for contact in ['L', 'R']:
                shared_models.append(MultiModel(
                    loss.SE_loss_trafo,
                    f'SE_loss{i}_{contact}',
                    kwargs = {
                        'qs_full': None, 'with_grad': True,
                        'i': i, 'N': N, 'contact': contact,
                    },
                ))
                self.used_losses[bulk_name].append(f'SE_loss{i}_{contact}')

                shared_models.append(MultiModel(
                    loss.j_loss_trafo,
                    f'j_loss{i}_{contact}',
                    kwargs = {'i': i, 'N': N, 'contact': contact},
                ))
                self.used_losses[bulk_name].append(f'j_loss{i}_{contact}')

        #for i in range(0,N+1):
        #    boundary_name = f'boundary{i}'
        #    self.used_losses[boundary_name] = []

        #    for contact in ['L', 'R']:
        #        shared_models.append(MultiModel(
        #            loss.wc_loss_trafo,
        #            f'wc_loss{i}_{contact}',
        #            kwargs = {'i': i, 'contact': contact},
        #        ))
        #        self.used_losses[boundary_name].append(f'wc_loss{i}_{contact}')

        #        shared_models.append(MultiModel(
        #            loss.cc_loss_trafo,
        #            f'cc_loss{i}_{contact}',
        #            kwargs = {'i': i, 'contact': contact},
        #        ))
        #        self.used_losses[boundary_name].append(f'cc_loss{i}_{contact}')


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
                subgrid_dict['DeltaE'] = lambda x, energy=energy: x == energy
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
                grids = ([f'boundary{i-1}' + dx_string  for dx_string in dx_strings]
                         + [f'bulk{i}']
                         + [f'boundary{i}' + dx_string  for dx_string in dx_strings])
                x_left = self.boundaries[i-1]
                x_right = self.boundaries[i]

                inputs_labels = []
                if params.continuous_voltage:
                    inputs_labels.append('voltage')
                if params.continuous_energy:
                    inputs_labels.append('DeltaE') # TODO: check whether E_L/E_R or DeltaE should be provided (required_quantities_labels would need to be changed as well)
                inputs_labels.append('x')

                model_transformations = {
                    'x': lambda x, q, x_left=x_left, x_right=x_right:
                            (x - x_left) / (x_right - x_left),
                    'DeltaE': lambda E, q: E / physics.EV,
                }

                for contact in ['L', 'R']:
                    for c in ['a', 'b']:
                        nn_model = SimpleNNModel(
                            inputs_labels,
                            params.activation_function,
                            n_neurons_per_hidden_layer = params.n_neurons_per_hidden_layer,
                            n_hidden_layers = params.n_hidden_layers,
                            model_dtype = params.model_dtype,
                            output_dtype = params.si_complex_dtype,
                            device = params.device,
                        )
                        c_model = TransformedModel(
                            nn_model,
                            input_transformations = model_transformations,
                        )
                        models.append(get_combined_multi_model(
                            c_model,
                            f'{c}_output{i}_{contact}',
                            grids,
                            combined_dimension_name = 'x',
                            required_quantities_labels = ['voltage', 'DeltaE', 'x'],
                        ))
                        trained_models_labels.append(f'{c}_output{i}_{contact}')

            models += shared_models


            self.trainers[trainer_name] = Trainer(
                models = models,
                batchers_training = batchers_training,
                batchers_validation = batchers_validation,
                used_losses = self.used_losses,
                trained_models_labels = trained_models_labels,
                Optimizer = params.Optimizer,
                optimizer_kwargs = params.optimizer_kwargs,
                optimizer_reset_tol=params.optimizer_reset_tol,
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