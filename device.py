import os
from pathlib import Path
import numpy as np
import torch

from kolpinn import io
from kolpinn import grid_quantities
from kolpinn.grid_quantities import Grid, Quantity, get_quantity
from kolpinn.batching import Batcher
from kolpinn.model import SimpleNNModel, ConstModel, FunctionModel, get_model, get_extended_q_batchwise
from kolpinn.training import Trainer

import parameters as params
import physics
import loss


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
            self.models['phi_dx' + str(i)] = FunctionModel(
                lambda q, i=i, **kwargs: q['phi'+str(i)].get_grad(q['x'], **kwargs),
                retain_graph = True,
                create_graph = True, # OPTIM: Set to false at boundary while validating (together with with_grad)
            )
            self.models['phi_dx_fd' + str(i)] = FunctionModel(
                lambda q, i=i, **kwargs: q['phi'+str(i)].get_fd_derivative('x'),
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
            self.models['V'+str(i)] = get_model(
                potentials[i],
                model_dtype = params.si_real_dtype,
                output_dtype = params.si_real_dtype,
            )
            self.models['m_eff'+str(i)] = get_model(
                m_effs[i],
                model_dtype = params.si_real_dtype,
                output_dtype = params.si_real_dtype,
            )

        ## Contacts
        for i in (0, N+1):
            k_function = lambda q, i=i: \
                physics.k_function(q['m_eff'+str(i)], q['E']-q['V'+str(i)])
            self.models['k'+str(i)] = FunctionModel(k_function)


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

                q = get_extended_q_batchwise(
                    batchers_validation[name],
                    models = {
                        'V': self.models['V'+str(i)],
                        'm_eff': self.models['m_eff'+str(i)],
                    },
                    models_require_grad = False,
                )
                mean = lambda x: torch.mean(x if torch.is_tensor(x) else x.values)
                avg_V = mean(q['V'])
                avg_m_eff = mean(q['m_eff'])
                avg_k = physics.k_function(avg_m_eff, energy - avg_V)
                estimated_phase_change = (avg_k * (x_right - x_left)).to(params.si_real_dtype)
                print(f'Estimated phase change in Layer {i}: {estimated_phase_change}')
                phase_multiplier = max(1, estimated_phase_change / (2 * np.pi))
                phi_transformation = lambda x, phase_multiplier=phase_multiplier: \
                    x * phase_multiplier
                trainer_models['phi' + str(i)] = SimpleNNModel(
                    ['x'],
                    {
                        'x': lambda x, q, x_left=x_left, x_right=x_right:
                                 (x - x_left) / (x_right - x_left),
                    },
                    lambda phi, q: phi,
                    params.activation_function,
                    n_neurons_per_hidden_layer = params.n_neurons_per_hidden_layer,
                    n_hidden_layers = params.n_hidden_layers,
                    model_dtype = params.model_dtype,
                    output_dtype = params.si_complex_dtype,
                    device = params.device,
                    complex_polar = params.complex_polar,
                    phi_transformation = phi_transformation,
                )
                trained_models_labels.append('phi' + str(i))

                fd = '_fd' if params.fd_first_derivatives else ''
                models_dict[name] = {
                    'phi' + str(i): trainer_models['phi' + str(i)],
                    'phi_dx' + str(i): self.models['phi_dx' + fd + str(i)],
                    'V' + str(i): self.models['V' + str(i)],
                    'm_eff' + str(i): self.models['m_eff' + str(i)],
                }

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
                    models_dict[name]['k' + str(0)] = self.models['k' + str(0)]
                else:
                    models_dict[name]['phi' + str(i)] = trainer_models['phi' + str(i)]
                    models_dict[name]['phi_dx' + str(i)] = self.models['phi_dx' + str(i)]
                if i == N:
                    models_dict[name]['k' + str(i+1)] = self.models['k' + str(i+1)]
                else:
                    models_dict[name]['phi' + str(i+1)] = trainer_models['phi' + str(i+1)]
                    models_dict[name]['phi_dx' + str(i+1)] = self.models['phi_dx' + str(i+1)]


            # Add the loss models

            used_losses = {}
            quantities_requiring_grad_dict = {}
            for batcher_name, loss_functions_dict in self.loss_functions.items():
                if (batcher_name.startswith('bulk')
                    and params.fd_first_derivatives
                    and params.fd_second_derivatives):
                    # The bulk needs no gradient if all derivatives are taken numerically
                    quantities_requiring_grad_dict[batcher_name] = []
                else:
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
