import torch

from kolpinn import grid_quantities
from kolpinn.grid_quantities import Grid, Quantity, get_quantity
from kolpinn.batching import Batcher
from kolpinn.model import QuantityModel, load_weights
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

        self.models = {}
        self.diffable_quantities = {}
        self.grids_training = {}
        self.grids_validation = {}
        self.batchers_training = {}
        self.batchers_validation = {}
        self.loss_functions = {}


        # Models & Grids

        ## Layers
        for i in range(1,N+1):
            x_left = self.boundaries[i-1]
            x_right = self.boundaries[i]
            self.models['phi' + str(i)] = QuantityModel(
                ['E','x'],
                {
                    'E': lambda E, q: E / physics.V_OOM,
                    'x': lambda x, q, x_left=x_left, x_right=x_right: # Capturing x_left/right
                             (x - x_left) / (x_right - x_left),
                },
                lambda phi, q: phi,
                params.activation_function,
                n_neurons_per_hidden_layer = params.n_neurons_per_hidden_layer,
                n_hidden_layers = params.n_hidden_layers,
                network_dtype = params.model_dtype,
                output_dtype = params.si_complex_dtype,
                device = params.device,
            )
            self.grids_training['bulk' + str(i)] = Grid({
                'E': torch.linspace(physics.E_MIN, physics.E_MAX, params.N_E_training),
                'x': torch.linspace(x_left, x_right, params.N_x_training),
            })
            self.grids_validation['bulk' + str(i)] = Grid({
                'E': torch.linspace(physics.E_MIN, physics.E_MAX, params.N_E_validation),
                'x': torch.linspace(x_left, x_right, params.N_x_validation),
            })

        ## Boundaries
        for i in range(0,N+1):
            self.grids_training['boundary' + str(i)] = Grid({
                'E': torch.linspace(
                         physics.E_MIN,
                         physics.E_MAX,
                         params.N_E_training,
                         dtype=params.si_real_dtype,
                     ),
                'x': torch.tensor(
                         [self.boundaries[i]],
                         dtype=params.si_real_dtype,
                     ),
            })
            self.grids_validation['boundary' + str(i)] = Grid({
                'E': torch.linspace(
                         physics.E_MIN,
                         physics.E_MAX,
                         params.N_E_validation,
                         dtype=params.si_real_dtype,
                     ),
                'x': torch.tensor(
                         [self.boundaries[i]],
                         dtype=params.si_real_dtype,
                     ),
            })

        load_weights(self.models, params.loaded_weights_index)


        # Quantities

        self.qs_training = physics.quantities_factory.get_quantities_dict(self.grids_training)
        self.qs_validation = physics.quantities_factory.get_quantities_dict(self.grids_validation)

        for i in range(0,N+2):
            potential = potentials[i]
            m_eff = m_effs[i]
            if not callable(potential):
                potential = lambda q, p=potential: Quantity(torch.tensor(p), q.grid)
            if not callable(m_eff):
                m_eff = lambda q, m=m_eff: Quantity(torch.tensor(m), q.grid)
            self.diffable_quantities['V'+str(i)] = potential
            self.diffable_quantities['m_eff'+str(i)] = m_eff

        for i in (0, N+1):
            k_function = lambda q, i=i: \
                (2 * q['m_eff'+str(i)] * (q['E']-q['V'+str(i)]) / physics.H_BAR**2).set_dtype(params.si_complex_dtype).transform(torch.sqrt)
            self.diffable_quantities['k'+str(i)] = k_function


        # Batchers & Losses

        ## Layers
        for i in range(1,N+1):
            name = 'bulk' + str(i)
            self.batchers_training[name] = Batcher(
                self.qs_training[name],
                self.grids_training[name],
                ['E','x'],
                [params.batch_size_E, params.batch_size_x],
            )
            self.batchers_validation[name] = Batcher(
                self.qs_validation[name],
                self.grids_validation[name],
                ['E','x'],
                [1, params.batch_size_x],
            )
            se_loss_function = lambda q, with_grad, i=i: \
                loss.get_SE_loss(q, with_grad=with_grad, i=i)
            const_j_loss_function = lambda q, with_grad, i=i: \
                loss.get_const_j_loss(q, with_grad=with_grad, i=i)
            self.loss_functions[name] = {
                'SE'+str(i): se_loss_function,
                'const_j'+str(i): const_j_loss_function,
            }

        ## Boundaries
        for i in range(0,N+1):
            name = 'boundary' + str(i)
            self.batchers_training[name] = Batcher(
                self.qs_training[name],
                self.grids_training[name],
                ['E'],
                [1],
            )
            self.batchers_validation[name] = Batcher(
                self.qs_validation[name],
                self.grids_validation[name],
                ['E'],
                [1],
            )

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

        self.quantities_requiring_grad_dict = dict((batcher_name, ['x'])
                                                   for batcher_name in self.batchers_training.keys())


        # Trainer

        self.trainer = Trainer(
            self.models,
            self.batchers_training,
            self.batchers_validation,
            self.loss_functions,
            self.quantities_requiring_grad_dict,
            params.Optimizer,
            params.learn_rate,
            diffable_quantities = self.diffable_quantities,
        )

        self.model_parameters = {}
