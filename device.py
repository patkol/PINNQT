import os
from pathlib import Path
import numpy as np
import torch

from kolpinn import mathematics
from kolpinn import io
from kolpinn import grid_quantities
from kolpinn.grid_quantities import Grid, get_fd_derivative
from kolpinn.batching import Batcher
from kolpinn.model import SimpleNNModel, ConstModel, FunctionModel, \
                          TransformedModel, get_model, get_extended_q_batchwise
from kolpinn.training import Trainer

import parameters as params
import physics
import loss



gaussian = lambda x, sigma: torch.exp(-x**2 / (2 * sigma**2))


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

        if params.loaded_parameters_index is not None:
            parameters_path = io.get_parameters_path(params.loaded_parameters_index)
            saved_energies = sorted([float(Path(s).stem[2:]) * physics.EV
                                     for s in os.listdir(parameters_path)])
            energies = torch.tensor(saved_energies, dtype=params.si_real_dtype)

        else:
            energies = torch.arange(
                physics.E_MIN,
                physics.E_MAX,
                physics.E_STEP,
                dtype=params.si_real_dtype,
            )


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
            self.models['phi_dx_fd' + str(i)] = FunctionModel(
                lambda q, i=i, **kwargs: get_fd_derivative('x', q['phi'+str(i)], q.grid),
            )
            self.models['phi_dx' + str(i)] = FunctionModel(
                lambda q, i=i, **kwargs: mathematics.grad(q['phi'+str(i)], q['x'], **kwargs),
                retain_graph = True,
                create_graph = True, # OPTIM: Set to false at boundary while validating (together with with_grad)
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
            if not i in (0,N):
                # The wave function is continuous on the left and right
                # by construction
                wc_loss_function = lambda q, with_grad, i=i: \
                    loss.get_wc_loss(q, with_grad=with_grad, i=i)
                self.loss_functions[name]['wc'+str(i)] = wc_loss_function
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

            smoother_function = lambda x: \
                x * (1 - gaussian(x, physics.smoothing_range))
            smooth_k_function = lambda q, i=i: \
                physics.k_function(
                    q['m_eff'+str(i)],
                    smoother_function(q['E']-q['V'+str(i)]),
                )
            # smooth_k: Fixing the non-smoothness of k in V at E=V
            self.models['smooth_k'+str(i)] = FunctionModel(smooth_k_function)


        # Trainers

        for energy in energies.cpu().numpy():
            energy_string = f'E={energy/physics.EV:.16e}'
            energy_subgrids_training = dict(
                (label, grid.get_subgrid({'E': lambda E: E == energy},
                                         copy_all = True))
                for label, grid in self.grids_training.items()
            )
            energy_subgrids_validation = dict(
                (label, grid.get_subgrid({'E': lambda E: E == energy},
                                         copy_all = True))
                for label, grid in self.grids_validation.items()
            )
            qs_training = physics.quantities_factory.get_quantities_dict(energy_subgrids_training)
            qs_validation = physics.quantities_factory.get_quantities_dict(energy_subgrids_validation)

            batchers_training = {}
            batchers_validation = {}
            trainer_models = {}
            models_dict = {}
            trained_models_labels = []

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

                if params.complex_polar:
                    q = get_extended_q_batchwise(
                        batchers_validation[name],
                        models = {
                            'V': self.models['V'+str(i)],
                            'm_eff': self.models['m_eff'+str(i)],
                        },
                        models_require_grad = False,
                    )
                    avg_V = torch.mean(q['V'])
                    avg_m_eff = torch.mean(q['m_eff'])
                    avg_k = physics.k_function(avg_m_eff, energy - avg_V)
                    estimated_phase_change = torch.real(avg_k * (x_right - x_left)).to(params.si_real_dtype)
                    print(f'Estimated phase change in Layer {i}: {estimated_phase_change}')
                    phase_multiplier = max(1, estimated_phase_change / (2 * np.pi))
                    phi_transformation = lambda x, phase_multiplier=phase_multiplier: \
                        x * phase_multiplier
                else:
                    phi_transformation = None

                nn_model = SimpleNNModel(
                    ['x'],
                    params.activation_function,
                    n_neurons_per_hidden_layer = params.n_neurons_per_hidden_layer,
                    n_hidden_layers = params.n_hidden_layers,
                    model_dtype = params.model_dtype,
                    output_dtype = params.si_complex_dtype,
                    device = params.device,
                    complex_polar = params.complex_polar,
                    phi_transformation = phi_transformation,
                )
                x_scaling_transformations = {
                    'x': lambda x, q, x_left=x_left, x_right=x_right:
                             (x - x_left) / (x_right - x_left),
                }
                if params.model_ab:
                    nn_model2 = SimpleNNModel(
                        ['x'],
                        params.activation_function,
                        n_neurons_per_hidden_layer = params.n_neurons_per_hidden_layer,
                        n_hidden_layers = params.n_hidden_layers,
                        model_dtype = params.model_dtype,
                        output_dtype = params.si_complex_dtype,
                        device = params.device,
                        complex_polar = params.complex_polar,
                        phi_transformation = phi_transformation,
                    )
                    trainer_models[f'a{i}'] = TransformedModel(
                        nn_model,
                        input_transformations = x_scaling_transformations,
                    )
                    trained_models_labels.append(f'a{i}')
                    trainer_models[f'b{i}'] = TransformedModel(
                        nn_model2,
                        input_transformations = x_scaling_transformations,
                    )
                    trained_models_labels.append(f'b{i}')
                    # The shifts by x_left/x_right are important for
                    # energies smaller than V, it keeps them from exploding.
                    trainer_models[f'phi{i}'] = FunctionModel(
                        lambda q, i=i, x_left=x_left, x_right=x_right: (
                            q[f'a{i}'] * torch.exp(1j * q[f'smooth_k{i}'] * (q['x'] - x_left))
                            + q[f'b{i}'] * torch.exp(-1j * q[f'smooth_k{i}'] * (q['x'] - x_right))
                        ),
                        output_dtype = params.si_complex_dtype,
                    )
                else:
                    trainer_models['phi' + str(i)] = TransformedModel(
                        nn_model,
                        input_transformations = x_scaling_transformations,
                    )
                    trained_models_labels.append('phi' + str(i))

                models_dict[name] = {
                    f'V{i}': self.models[f'V{i}'],
                    f'm_eff{i}': self.models[f'm_eff{i}'],
                }
                if params.model_ab:
                    models_dict[name][f'smooth_k{i}'] = self.models[f'smooth_k{i}']
                    models_dict[name][f'a{i}'] = trainer_models[f'a{i}']
                    models_dict[name][f'b{i}'] = trainer_models[f'b{i}']
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
                if i == 0:
                    models_dict[name]['k' + str(i)] = self.models['k' + str(i)]
                if i != 0:
                    if params.model_ab:
                        models_dict[name][f'a{i}'] = trainer_models[f'a{i}']
                        models_dict[name][f'b{i}'] = trainer_models[f'b{i}']
                        models_dict[name]['smooth_k' + str(i)] = self.models['smooth_k' + str(i)]
                    models_dict[name]['phi' + str(i)] = trainer_models['phi' + str(i)]
                    # Always using the exact derivatives on the boundaries
                    models_dict[name][f'phi_dx{i}'] = self.models[f'phi_dx{i}']
                if i == N:
                    models_dict[name]['k' + str(i+1)] = self.models['k' + str(i+1)]
                if i != N:
                    if params.model_ab:
                        models_dict[name][f'a{i+1}'] = trainer_models[f'a{i+1}']
                        models_dict[name][f'b{i+1}'] = trainer_models[f'b{i+1}']
                        models_dict[name]['smooth_k' + str(i+1)] = self.models['smooth_k' + str(i+1)]
                    models_dict[name]['phi' + str(i+1)] = trainer_models['phi' + str(i+1)]
                    models_dict[name]['phi_dx' + str(i+1)] = self.models['phi_dx' + str(i+1)]


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
