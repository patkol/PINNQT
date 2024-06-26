# Copyright (c) 2024 ETH Zurich, Patrice Kolb


import os
from pathlib import Path
from typing import Callable
import itertools
import numpy as np
import torch

from kolpinn import mathematics
from kolpinn.mathematics import complex_abs2
from kolpinn import storage
from kolpinn import grid_quantities
from kolpinn.grid_quantities import Grid, Subgrid, QuantityDict, \
                                    get_fd_derivative, combine_quantity, \
                                    squeeze_to, restrict
from kolpinn.batching import Batcher
from kolpinn.model import SimpleNNModel, ConstModel, FunctionModel, \
                          TransformedModel, get_model, \
                          MultiModel, get_multi_model, \
                          get_combined_multi_model, get_qs
from kolpinn.training import Trainer

import parameters as params
import physics
from classes import Contact
import loss


dx_strings = ['']
dx_shifts = [0.]
if params.fd_first_derivatives:
    dx_strings += ['_pdx', '_mdx']
    dx_shifts += [physics.dx, -physics.dx]

k_function = lambda q, i, contact: \
    physics.k_function(
        q[f'm_eff{i}'],
        q[f'E_{contact}'] - q[f'V_int{i}'] - q[f'V_el_approx{i}'],
    )

gaussian = lambda x, sigma: torch.exp(-x**2 / (2 * sigma**2))

smoother_function = lambda x, smoothing_range: \
    x * (1 - gaussian(x, smoothing_range))

# smooth_k: Fixing the non-smoothness of k in V at E=V
smooth_k_function = lambda q, i, contact: \
    physics.k_function(
        q[f'm_eff{i}'],
        smoother_function(
            q[f'E_{contact}'] - q[f'V_int{i}'] - q[f'V_el_approx{i}'],
            physics.energy_smoothing_range,
        ),
    )

def transition_function(a, b, transition_exp):
    """
    Smoothly transition from a at x=0 to b at x->inf.
    transition_exp(x) = torch.exp(-(x-x_left) / transition_distance)
    """
    return transition_exp * a + (1-transition_exp) * b

def get_V_voltage(q: QuantityDict, device_start, device_end):
    distance_factor = (q['x'] - device_start) / (device_end - device_start)
    # Cap the factor beyond the device limits, these regions correspond to
    # the contacts
    distance_factor[distance_factor < 0] = 0
    distance_factor[distance_factor > 1] = 1
    V_voltage = -q['voltage'] * physics.EV * distance_factor

    return V_voltage

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

def get_E_fermi(q: QuantityDict, *, i):
    dos = 8*np.pi / physics.H**3 * torch.sqrt(2 * q[f'm_eff{i}']**3 * q['DeltaE'])

    best_E_f = None
    best_abs_remaining_charge = float('inf')
    E_fs = np.arange(0*physics.EV, 0.8*physics.EV, 1e-3*physics.EV)
    for E_f in E_fs:
        fermi_dirac = 1 / (1 + torch.exp(physics.BETA * (q['DeltaE'] - E_f)))
        integrand = dos * fermi_dirac
        # Particle, not charge density
        n = (grid_quantities.sum_dimension('DeltaE', integrand, q.grid)
             * physics.E_STEP)
        abs_remaining_charge = torch.abs(n - q[f'doping{i}']).item()
        if abs_remaining_charge < best_abs_remaining_charge:
            best_abs_remaining_charge = abs_remaining_charge
            best_E_f = E_f

        E_f += 1e-3 * physics.EV

    assert best_E_f is not None
    assert best_E_f != E_fs[0] and best_E_f != E_fs[-1], E_f / physics.EV

    relative_remaining_charge = best_abs_remaining_charge / q[f'doping{i}']
    print(f'Best fermi energy: {best_E_f / physics.EV} eV with a relative remaining charge of {relative_remaining_charge}')

    #best_E_f = 0.258 * physics.EV # Fixed fermi level

    return best_E_f + q[f'V_int{i}'] + q[f'V_el{i}']

def factors_trafo(qs, i, contact):
    """
    Use the boundary conditions to find the factors to multiply a/b_output with
    given that a and b are known in layer `i_next`
    """

    i_next = contact.get_next_layer_index(i)
    i_boundary = contact.get_out_boundary_index(i)
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
        contact: Contact,
        grid_name: str,
        i: int,
    ):

    boundary_q = qs[f'boundary{contact.get_out_boundary_index(i)}']
    q = qs[grid_name]

    coeff = boundary_q[f'{c}_factor{i}_{contact}'] * q[f'{c}_output{i}_{contact}']
    q[f'{c}{i}_{contact}'] = coeff

    return qs

def add_coeffs(
        qs: dict[str,QuantityDict], contact: Contact, grid_name: str, i: int,
    ):
    for c in ('a', 'b'):
        add_coeff(c, qs, contact, grid_name, i)

    return qs

def to_full_trafo(
    qs: dict[str,QuantityDict],
    *,
    N: int,
    label_fn: Callable[[int],str],
    quantity_label: str,
):
    """
    Combine quantities in the 'bulk{i}' grids to the 'full' grid.
    They're called 'label_fn(i)' in the bulk grids and
    'quantity_label' in 'full'.
    """
    q = qs['full']
    quantity_list: list[torch.Tensor] = []
    subgrid_list: list[Subgrid] = []
    for i in range(1,N+1):
        q_layer = qs[f'bulk{i}']
        quantity_list.append(q_layer[label_fn(i)])
        assert isinstance(q_layer.grid, Subgrid)
        subgrid_list.append(q_layer.grid)

    q[quantity_label] = combine_quantity(quantity_list, subgrid_list, q.grid)

    return qs

def to_bulks_trafo(
    qs: dict[str,QuantityDict],
    *,
    N: int,
    label_fn: Callable[[int],str],
    quantity_label: str,
):
    """
    Inverse of 'to_full_trafo'
    """
    full_quantity = qs['full'][quantity_label]
    for i in range(1,N+1):
        q_layer = qs[f'bulk{i}']
        q_layer[label_fn(i)] = restrict(full_quantity, q_layer.grid)

    return qs

def dos_trafo(qs, *, contact):
    q = qs['full']
    q_in = qs[contact.grid_name]

    incoming_coeff_in = q_in[f'{contact.incoming_coeff_in_name}{contact.index}_{contact}']
    incoming_amplitude = complex_abs2(incoming_coeff_in)
    q[f'DOS_{contact}'] = (1 / (2*np.pi)
                           * complex_abs2(q[f'phi_{contact}'])
                           / q_in[f'dE_dk{contact.index}_{contact}']
                           / incoming_amplitude)
    return qs

def n_contact_trafo(qs, *, contact):
    """ Density from one contact only """
    q = qs['full']
    q_in = qs[contact.grid_name]

    integrand = q[f'DOS_{contact}'] * q_in[f'fermi_integral_{contact}']
    q[f'n_{contact}'] = (grid_quantities.sum_dimension('DeltaE', integrand, q.grid)
                            * physics.E_STEP)
    return qs

def n_trafo(qs, *, contacts):
    """ Full density """
    q = qs['full']
    q['n'] = sum(q[f'n_{contact}'] for contact in contacts)

    return qs

def V_electrostatic_trafo(qs):
    q = qs['full']

    assert q.grid.dimensions_labels[-1] == 'x'

    # Construct the discretized Laplace operator M assuming an
    # equispaced x grid
    # OPTIM: do this only once
    Nx = q.grid.dim_size['x']
    dx = q.grid.dimensions['x'][1] - q.grid.dimensions['x'][0]
    M = torch.zeros(Nx, Nx)
    permittivity = squeeze_to(['x'], q['permittivity'], q.grid)
    for i in range(1, Nx):
        M[i,i-1] = (permittivity[i] + permittivity[i-1]) / (2 * dx**2)
    for i in range(0, Nx-1):
        M[i,i+1] = (permittivity[i] + permittivity[i+1]) / (2 * dx**2)
    for i in range(1, Nx-1):
        M[i,i] = -M[i,i-1] - M[i,i+1]

    # Dirichlet BC
    # Assuming constant permittivity at the outer boundaries
    M[0,0] = -2 * permittivity[0] / dx**2
    M[-1,-1] = -2 * permittivity[-1] / dx**2

    # TODO: implementation that works if x is not the last coordinate,
    #       kolpinn function
    #       Then remove the assertion above
    rho = physics.Q_E * (q['doping'] - q['n'])
    rhs = -physics.Q_E * rho
    # Phi[-1] = 0, Phi[Nx] = -voltage * EV
    # S[0] = -epsilon[-1/2] / dx^2 * Phi[-1],
    # S[Nx-1] = -epsilon[Nx-1/2] / dx^2 * Phi[Nx]
    Phi_Nx = q['voltage'].squeeze(-1) * physics.EV
    rhs[:,:,Nx-1] += -permittivity[-1] / dx**2 * Phi_Nx
    # Permute: x must be the second last coordinate for torch.linalg.solve
    rhs_permuted = torch.permute(rhs, (0,2,1))
    Phi_permuted = torch.linalg.solve(M, rhs_permuted)
    Phi = torch.permute(Phi_permuted, (0,2,1))
    q['V_el'] = -physics.Q_E * Phi

    return qs

def TR_trafo(qs, *, contact):
    """ Transmission probability """
    q = qs['full']
    q_in = qs[contact.grid_name]
    q_out = qs[contact.out_boundary_name]
    incoming_coeff_in = q_in[f'{contact.incoming_coeff_in_name}{contact.index}_{contact}']
    incoming_coeff_out = q_in[f'{contact.incoming_coeff_out_name}{contact.index}_{contact}']
    outgoing_coeff_out = q_out[f'{contact.outgoing_coeff_out_name}{contact.out_index}_propagated_{contact}']
    real_v_in = torch.real(q_in[f'v{contact.index}_{contact}'])
    real_v_out = torch.real(q_out[f'v{contact.out_index}_{contact}'])
    q[f'T_{contact}'] = (complex_abs2(outgoing_coeff_out) / complex_abs2(incoming_coeff_in)
                         * real_v_out / real_v_in)
    q[f'R_{contact}'] = complex_abs2(incoming_coeff_out) / complex_abs2(incoming_coeff_in)

    return qs

def I_contact_trafo(qs, *, contact):
    q = qs['full']
    q_in = qs[contact.grid_name]
    integrand = q[f'T_{contact}'] * q_in[f'fermi_integral_{contact}']
    integral = (grid_quantities.sum_dimension('DeltaE', integrand, q.grid)
                * physics.E_STEP)
    sign = 1 if contact.name=='L' else -1
    prefactor = -physics.Q_E / physics.H_BAR / (2*np.pi) * sign
    q[f'I_spectrum_{contact}'] = prefactor * integrand
    q[f'I_{contact}'] =  prefactor * integral

    return qs

def I_trafo(qs, *, contacts):
    q = qs['full']
    q['I'] = sum(q[f'I_{contact}'] for contact in contacts)

    return qs

def j_exact_trafo(qs, *, contact):
    """
    Calculate the exact current at the output contact
    """
    q = qs[f'boundary{contact.out_boundary_index}']
    k = q[f'k{contact.out_index}_{contact}']
    m_eff = q[f'm_eff{contact.out_index}']
    sign = -1 if contact.name == 'L' else 1
    q[f'j_exact_{contact}'] = sign * physics.H_BAR * k / m_eff

    return qs



class Device:
    def __init__(
            self,
            *,
            boundaries,
            potentials,
            m_effs,
            dopings,
            permittivities,
            includes_contacts,
        ):
        """
        boundaries: [x_b0, ..., x_bN] with N the number of layers
        potentials: [V_0, ..., V_N+1] (including contacts),
                    constants or functions of q, grid
        m_effs: [m_0, ..., m_N+1], like potentials
        dopings & permittivities: Same
        Layer i in [1,N] has x_b(i-1) on the left and x_bi on the right.
        """

        N = len(potentials)-2
        assert len(m_effs) == N+2
        assert len(dopings) == N+2
        assert len(permittivities) == N+2
        assert len(boundaries) == N+1
        assert sorted(boundaries) == boundaries, boundaries

        self.n_layers = N
        self.boundaries = boundaries
        self.potentials = potentials
        self.m_effs = m_effs
        self.dopings = dopings
        self.permittivities = permittivities
        self.includes_contacts = includes_contacts

        self.device_start = self.boundaries[1 if self.includes_contacts else 0]
        self.device_end = self.boundaries[-2 if self.includes_contacts else -1]

        self.loss_functions = {}

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

        self.left_contact = Contact(
            name = 'L',
            index = 0,
            out_index = N+1,
            grid_name = f'boundary{0}',
            incoming_coeff_in_name = 'a',
            get_in_boundary_index = lambda i: max(0,i-1),
            get_out_boundary_index = lambda i: min(N,i),
            get_previous_layer_index = lambda i: i-1,
            get_next_layer_index = lambda i: i+1,
        )
        self.right_contact = Contact(
            name = 'R',
            index = N+1,
            out_index = 0,
            grid_name = f'boundary{N}',
            incoming_coeff_in_name = 'b',
            get_in_boundary_index = lambda i: min(N,i),
            get_out_boundary_index = lambda i: max(0,i-1),
            get_previous_layer_index = lambda i: i+1,
            get_next_layer_index = lambda i: i-1,
        )
        self.contacts = [self.left_contact, self.right_contact]

        layer_indices_dict = {'L': range(N+1,-1,-1), 'R': range(0,N+2)}


        # Grids

        self.grids = {}

        x_left = self.boundaries[0]
        x_right = self.boundaries[-1]
        # We're excluding the left- and rightmost points to avoid special cases
        x_step = (x_right - x_left) / (params.N_x + 2 - 1)
        self.grids['full'] = Grid({
            'voltage': voltages,
            'DeltaE': energies,
            'x': torch.linspace(
                # Making sure all points lie in a layer
                x_left + x_step,
                x_right - x_step,
                params.N_x,
            ),
        })

        self.n_dim = self.grids['full'].n_dim

        ## Layers
        for i in range(1,N+1):
            grid_name = f'bulk{i}'
            x_left = self.boundaries[i-1]
            x_right = self.boundaries[i]

            self.grids[grid_name] = self.grids['full'].get_subgrid(
                {'x': lambda x: torch.logical_and(x>=x_left, x<x_right)},
                copy_all=False,
            )

        ## Boundaries
        for i in range(0,N+1):
            for dx_string, dx_shift in zip(dx_strings, dx_shifts):
                grid_name = f'boundary{i}' + dx_string
                x = self.boundaries[i] + dx_shift
                self.grids[grid_name] = Grid({
                    'voltage': voltages,
                    'DeltaE': energies,
                    'x': torch.tensor([x], dtype=params.si_real_dtype),
                })

        quantities_requiring_grad = {}
        for grid_name in self.grids:
            quantities_requiring_grad[grid_name] = []
            if grid_name != 'full' and (not params.fd_first_derivatives
                                        or not params.fd_second_derivatives):
                # TODO: Is it fine that the grad is not propagated from full to bulk{i}?
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
            models_dict[f'V_int{i}'] = get_model(
                potentials[i],
                model_dtype = params.si_real_dtype,
                output_dtype = params.si_real_dtype,
            )
            models_dict[f'V_el_approx{i}'] = FunctionModel(
                lambda q, i=i: get_V_voltage(q, self.device_start, self.device_end),
            )
            models_dict[f'm_eff{i}'] = get_model(
                m_effs[i],
                model_dtype = params.si_real_dtype,
                output_dtype = params.si_real_dtype,
            )
            models_dict[f'doping{i}'] = get_model(
                dopings[i],
                model_dtype = params.si_real_dtype,
                output_dtype = params.si_real_dtype,
            )
            models_dict[f'permittivity{i}'] = get_model(
                permittivities[i],
                model_dtype = params.si_real_dtype,
                output_dtype = params.si_real_dtype,
            )

            for contact in self.contacts:
                x_in = self.boundaries[contact.get_in_boundary_index(i)]
                x_out = self.boundaries[contact.get_out_boundary_index(i)]

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

        ## Both contacts
        const_models_dict[0]['V_el0'] = ConstModel(
            0,
            model_dtype=params.si_real_dtype,
        )
        const_models_dict[N+1][f'V_el{N+1}'] = FunctionModel(
            lambda q: -q['voltage'] * physics.EV,
        )

        for contact in self.contacts:
            for i in (contact.index, contact.out_index):
                for c in ('a', 'b'):
                    const_models_dict[i][f'{c}_output{i}_{contact}'] = one_model
                    const_models_dict[i][f'{c}_output{i}_{contact}_dx'] = zero_model
                # V_el_approx is exact at the contacts
                const_models_dict[i][f'v{i}_{contact}'] = FunctionModel(
                    lambda q, i=i, contact=contact: torch.sqrt(
                        (2 * (q[f'E_{contact}'] - q[f'V_int{i}']
                              - q[f'V_el{i}'])
                         / q[f'm_eff{i}']).to(params.si_complex_dtype)
                    )
                )



        ## Output contact: Initial conditions
        const_models_dict[N+1][f'a{N+1}_L'] = one_model
        const_models_dict[N+1][f'b{N+1}_L'] = zero_model
        const_models_dict[0]['a0_R'] = zero_model
        const_models_dict[0]['b0_R'] = one_model
        for contact in self.contacts:
            i = contact.out_index
            for c in ('a', 'b'):
                const_models_dict[i][f'{c}{i}_{contact}_dx'] = zero_model
                const_models_dict[i][f'{c}{i}_propagated_{contact}'] = \
                    const_models_dict[i][f'{c}{i}_{contact}']
                const_models_dict[i][f'{c}{i}_propagated_{contact}_dx'] = \
                    const_models_dict[i][f'{c}{i}_{contact}_dx']

        ## Input contact
        for contact in self.contacts:
            # OPTIM: move to 'full' such that it does not have to be computed
            #        for the pdx & mdx grids separately
            i = contact.index
            const_models_dict[i][f'dE_dk{i}_{contact}'] = FunctionModel(
                lambda q, i=i, contact=contact:
                    torch.sqrt(2 * physics.H_BAR**2
                               * (q[f'E_{contact}'] - q[f'V_int{i}']
                                  - q[f'V_el{i}'])
                               / q[f'm_eff{i}']),
            )
            const_models_dict[i][f'E_fermi_{contact}'] = FunctionModel(
                lambda q, i=i: get_E_fermi(q, i=i),
            )
            const_models_dict[i][f'fermi_integral_{contact}'] = FunctionModel(
                lambda q, i=i, contact=contact:
                    (q[f'm_eff{i}'] / (np.pi * physics.H_BAR**2 * physics.BETA)
                     * torch.log(
                           1 + torch.exp(
                                   physics.BETA
                                   * (q[f'E_fermi_{contact}'] - q[f'E_{contact}'])
                               )
                       )
                    ),
            )

        const_models = []

        ## Full device
        grid_name = 'full'
        for model_name, model in layer_indep_const_models_dict.items():
            const_models.append(
                get_multi_model(model, model_name, grid_name),
            )

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

        # Constant MultiModels
        for contact in self.contacts:
            # OPTIM: unused
            const_models.append(MultiModel(
                j_exact_trafo,
                f'j_exact_{contact}',
                kwargs = {'contact': contact},
            ))

        const_models.append(MultiModel(
            to_full_trafo,
            f'm_eff',
            kwargs = {
                'N': N,
                'label_fn': lambda i, *, contact=contact: f'm_eff{i}',
                'quantity_label': 'm_eff',
            },
        ))
        const_models.append(MultiModel(
            to_full_trafo,
            'V_int',
            kwargs = {
                'N': N,
                'label_fn': lambda i, *, contact=contact: f'V_int{i}',
                'quantity_label': 'V_int',
            },
        ))
        const_models.append(MultiModel(
            to_full_trafo,
            f'doping',
            kwargs = {
                'N': N,
                'label_fn': lambda i, *, contact=contact: f'doping{i}',
                'quantity_label': 'doping',
            },
        ))
        const_models.append(MultiModel(
            to_full_trafo,
            f'permittivity',
            kwargs = {
                'N': N,
                'label_fn': lambda i, *, contact=contact: f'permittivity{i}',
                'quantity_label': 'permittivity',
            },
        ))


        # Parameter-dependent but trainer-independent models

        dependent_models = []
        self.used_losses = {}

        # Add the coeffs to `dependent_models` layer by layer
        for contact in self.contacts:
            for i in layer_indices_dict[contact.name]:
                bulk = f'bulk{i}'
                boundary_in = f'boundary{contact.get_in_boundary_index(i)}'
                boundary_out = f'boundary{contact.get_out_boundary_index(i)}'
                is_in_contact = i == contact.index
                is_out_contact = i == contact.out_index
                is_contact = is_in_contact or is_out_contact
                bulks = [] if is_contact else [bulk]
                boundaries_in = [] if is_in_contact else [boundary_in + dx_string
                                                          for dx_string in dx_strings]
                boundaries_out = [] if is_out_contact else [boundary_out + dx_string
                                                            for dx_string in dx_strings]
                single_boundaries = []
                if not is_in_contact:
                    single_boundaries.append(boundary_in)
                if not is_out_contact:
                    single_boundaries.append(boundary_out)

                if not is_contact:
                    for c in ('a', 'b'):
                        dependent_models.append(get_dx_model(
                            'multigrid' if params.fd_first_derivatives else 'exact',
                            f'{c}_output{i}_{contact}',
                            boundary_out,
                        ))

                if not is_out_contact:
                    dependent_models.append(MultiModel(
                        lambda qs, i=i, contact=contact:
                            factors_trafo(qs, i, contact),
                        f'factors{i}_{contact}',
                    ))

                    for grid_name in boundaries_in + bulks + boundaries_out:
                        dependent_models.append(MultiModel(
                            lambda qs, contact=contact, grid_name=grid_name, i=i:
                                add_coeffs(qs, contact, grid_name, i),
                            f'coeffs{i}',
                        ))

                if not is_contact:
                    for c in ('a', 'b'):
                        for grid_name in boundaries_in:
                            dependent_models.append(get_multi_model(
                                FunctionModel(
                                    lambda q, c=c, i=i, contact=contact:
                                        q[f'{c}{i}_{contact}'] * q[f'{c}_propagation_factor{i}_{contact}']
                                ),
                                f'{c}{i}_propagated_{contact}',
                                grid_name,
                            ))

                        dependent_models.append(get_dx_model(
                            'multigrid' if params.fd_first_derivatives else 'exact',
                            f'{c}{i}_propagated_{contact}',
                            boundary_in,
                        ))

        # Derived quantities
        ## Input contact (includes global quantities)
        for contact in self.contacts:
            dependent_models.append(MultiModel(
                TR_trafo,
                f'T/R_{contact}',
                kwargs = {'contact': contact},
            ))

            dependent_models.append(MultiModel(
                I_contact_trafo,
                f'I_{contact}',
                kwargs = {'contact': contact},
            ))

        dependent_models.append(MultiModel(
            I_trafo,
            f'I',
            kwargs = {'contacts': self.contacts},
        ))

        ## Bulk
        for i in range(1,N+1):
            bulk_name = f'bulk{i}'
            self.used_losses[bulk_name] = []

            for contact in self.contacts:
                dependent_models.append(get_multi_model(
                    FunctionModel(
                        lambda q, i=i, contact=contact:
                            (q[f'a{i}_{contact}'] * q[f'a_phase{i}_{contact}']
                             + q[f'b{i}_{contact}'] * q[f'b_phase{i}_{contact}'])
                    ),
                    f'phi{i}_{contact}',
                    bulk_name,
                ))

                dependent_models.append(get_dx_model(
                    'singlegrid' if params.fd_first_derivatives else 'exact',
                    f'phi{i}_{contact}',
                    bulk_name,
                ))

                dependent_models.append(get_multi_model(
                    FunctionModel(
                        lambda q, i=i, contact=contact:
                            torch.imag(physics.H_BAR * torch.conj(q[f'phi{i}_{contact}'])
                                        * q[f'phi{i}_{contact}_dx'] / q[f'm_eff{i}']),
                    ),
                    f'j{i}_{contact}',
                    bulk_name,
                ))

                dependent_models.append(MultiModel(
                    loss.j_loss_trafo,
                    f'j_loss{i}_{contact}',
                    kwargs = {'i': i, 'N': N, 'contact': contact},
                ))
                self.used_losses[bulk_name].append(f'j_loss{i}_{contact}')

        for contact in self.contacts:
            dependent_models.append(MultiModel(
                to_full_trafo,
                f'phi_{contact}',
                kwargs = {
                    'N': N,
                    'label_fn': lambda i, *, contact=contact: f'phi{i}_{contact}',
                    'quantity_label': f'phi_{contact}',
                },
            ))
            dependent_models.append(MultiModel(
                dos_trafo,
                f'DOS_{contact}',
                kwargs = {'contact': contact},
            ))
            dependent_models.append(MultiModel(
                n_contact_trafo,
                f'n_{contact}',
                kwargs = {'contact': contact},
            ))

        dependent_models.append(MultiModel(
            n_trafo,
            'n',
            kwargs = {'contacts': self.contacts},
        ))
        dependent_models.append(MultiModel(
            V_electrostatic_trafo,
            'V_el',
        ))
        dependent_models.append(MultiModel(
            to_bulks_trafo,
            f'V_el_distribution',
            kwargs = {
                'N': N,
                'label_fn': lambda i, *, contact=contact: f'V_el{i}',
                'quantity_label': f'V_el',
            },
        ))

        ## Bulk again
        for i in range(1,N+1):
            bulk_name = f'bulk{i}'
            for contact in self.contacts:
                dependent_models.append(MultiModel(
                    loss.SE_loss_trafo,
                    f'SE_loss{i}_{contact}',
                    kwargs = {
                        'qs_full': None, 'with_grad': True,
                        'i': i, 'N': N, 'contact': contact,
                    },
                ))
                self.used_losses[bulk_name].append(f'SE_loss{i}_{contact}')


        qs = get_qs(self.grids, const_models, quantities_requiring_grad)

        # Batchers

        batchers = {}

        ## Full device
        grid_name = 'full'
        batchers[grid_name] = Batcher(
            qs[grid_name],
            [],
            [],
        )

        ## Layers
        for i in range(1,N+1):
            grid_name = f'bulk{i}'
            #batch_size_x = (self.grids[grid_name].dim_size['x']
            #                if params.batch_size_x == -1
            #                else params.batch_size_x)
            batchers[grid_name] = Batcher(
                qs[grid_name],
                [], #['x'],
                [], #[batch_size_x],
            )

        ## Boundaries
        for i in range(0,N+1):
            for grid_name in [f'boundary{i}' + dx_string
                              for dx_string in dx_strings]:
                batchers[grid_name] = Batcher(
                    qs[grid_name],
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

            for contact in self.contacts:
                for c in ('a', 'b'):
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

        models += dependent_models


        self.trainer = Trainer(
            models = models,
            batchers_training = batchers,
            batchers_validation = batchers,
            used_losses = self.used_losses,
            trained_models_labels = trained_models_labels,
            Optimizer = params.Optimizer,
            optimizer_kwargs = params.optimizer_kwargs,
            optimizer_reset_tol = params.optimizer_reset_tol,
            Scheduler = params.Scheduler,
            scheduler_kwargs = params.scheduler_kwargs,
            saved_parameters_index = saved_parameters_index,
            save_optimizer = params.save_optimizer,
            loss_aggregate_function = params.loss_aggregate_function,
            name = 'all',
        )
        self.trainer.load(
            params.loaded_parameters_index,
            load_optimizer = params.load_optimizer,
            load_scheduler = params.load_scheduler,
        )


    def get_extended_qs(self):
        return self.trainer.get_extended_qs(for_training=False)
