# Copyright (c) 2025 ETH Zurich, Patrice Kolb

# OPTIM: don't return qs in trafos


import copy
import itertools
import numpy as np
import scipy.interpolate  # type: ignore
import torch

from kolpinn import mathematics
from kolpinn.grids import Grid
from kolpinn import quantities
from kolpinn.quantities import QuantityDict
from kolpinn import model

from classes import Contact
import physical_constants as consts
import parameters as params


def transition_f(x: torch.Tensor):
    """
    Smooth transition from 0 at x <= 0 to 1 at x -> inf
    https://en.wikipedia.org/wiki/Bump_function#Examples
    """
    # Assert that we don't ignore nonfinite x due to the nan_to_num
    assert torch.all(torch.isfinite(x)), x

    exp = torch.exp(-1 / x)
    clean_exp = torch.nan_to_num(exp)  # 0 at x=0

    return clean_exp * (x > 0)


def transition_g(x: torch.Tensor):
    """
    Smooth transition from 0 at x <= 0 to 1 at x >= 1
    https://en.wikipedia.org/wiki/Bump_function#Examples
    """
    f_x = transition_f(x)
    return f_x / (f_x + transition_f(1 - x))


def smooth_transition(x: torch.Tensor, x0: float, x1: float, y0: float, y1: float):
    """
    Smooth transition from y0 at x0 to y1 at x1
    """
    g = transition_g((x - x0) / (x1 - x0))
    return (y1 - y0) * g + y0


def smooth_rectangle(
    x: torch.Tensor,
    *,
    x_L: float,
    x_R: float,
    dx_smoothing: float,
    y0: float,
    y1: float,
):
    """
    The function with y0 outside and y1 inside of [x_L, x_R], smoothened
    """
    left_transition = smooth_transition(
        x, x0=x_L - dx_smoothing / 2, x1=x_L + dx_smoothing / 2, y0=y0, y1=y1
    )
    right_transition = smooth_transition(
        x, x0=x_R - dx_smoothing / 2, x1=x_R + dx_smoothing / 2, y0=0, y1=y0 - y1
    )

    return left_transition + right_transition


def smoothen(quantity: torch.Tensor, grid: Grid, dim_label: str):
    """
    Smoothen `quantity` on `grid` along `dim_label`.
    `grid` must be sorted along `dim_label`.
    """

    assert quantities.compatible(quantity, grid)

    other_dim_labels = copy.copy(grid.dimensions_labels)
    other_dim_labels.remove(dim_label)
    other_ranges = [
        range(quantity.size(grid.index[other_dim_label]))
        for other_dim_label in other_dim_labels
    ]

    smoothened_quantity = torch.zeros_like(quantity)

    # scipy.interpolate.UnivariateSpline only works with 1D data, so we have to
    # call it for all combinations of the other dimensions.
    for other_indices in itertools.product(*other_ranges):
        # Get current slice
        slices = [slice(None)] * grid.n_dim
        for other_dim_label, other_index in zip(other_dim_labels, other_indices):
            slices[grid.index[other_dim_label]] = other_index

        # Smoothen slice
        real_spline = scipy.interpolate.UnivariateSpline(
            grid[dim_label].cpu(),
            torch.real(quantity[slices]).cpu(),
            s=1,  # TODO: make adjustable
        )
        imag_spline = scipy.interpolate.UnivariateSpline(
            grid[dim_label].cpu(),
            torch.imag(quantity[slices]).cpu(),
            s=1,  # TODO: make adjustable
        )
        smoothened_quantity[slices] = torch.from_numpy(
            real_spline(grid[dim_label].cpu()) + 1j * imag_spline(grid[dim_label].cpu())
        )

    return smoothened_quantity


def get_k(q: QuantityDict, i: int, contact: Contact) -> torch.Tensor:
    k_squared = (
        2
        * q[f"m_eff{i}"]
        * (q[f"E_{contact}"] - q[f"V_int{i}"] - q[f"V_el{i}"])
        / consts.H_BAR**2
    )
    k_squared = k_squared.to(params.si_complex_dtype)

    return torch.sqrt(k_squared)


def get_V_voltage(q: QuantityDict, device_start, device_end):
    distance_factor = (q["x"] - device_start) / (device_end - device_start)
    # Cap the factor beyond the device limits, these regions correspond to
    # the contacts
    distance_factor[distance_factor < 0] = 0
    distance_factor[distance_factor > 1] = 1
    V_voltage = -q["voltage"] * consts.EV * distance_factor

    return V_voltage


def get_dx_model(mode: str, quantity_name: str, grid_name: str):
    name = quantity_name + "_dx"
    if mode == "exact":
        dx_model_single = model.FunctionModel(
            lambda q, *, q_full, with_grad: mathematics.grad(
                q[quantity_name],
                q_full["x"],
                retain_graph=True,  # OPTIM: not always necessary
                create_graph=True,  # OPTIM: not always necessary
            ),
            q_full=None,
            with_grad=True,
        )

        return model.get_multi_model(dx_model_single, name, grid_name)

    if mode == "multigrid":

        def dx_qs_trafo(qs):
            quantity_right = qs[grid_name + "_pdx"][quantity_name]
            quantity_left = qs[grid_name + "_mdx"][quantity_name]
            qs[grid_name][name] = (quantity_right - quantity_left) / (2 * params.dx)
            return qs

    elif mode == "singlegrid":

        def dx_qs_trafo(qs):
            q = qs[grid_name]
            q[name] = quantities.get_fd_derivative("x", q[quantity_name], q.grid)
            return qs

    else:
        raise ValueError("Unknown dx mode: ", mode)

    return model.MultiModel(dx_qs_trafo, name)


def get_E_fermi(q: QuantityDict, *, i: int):
    """The potential V still has to be added."""
    if params.CONSTANT_FERMI_LEVEL is not None:
        return params.CONSTANT_FERMI_LEVEL

    dos = 8 * np.pi / consts.H**3 * torch.sqrt(2 * q[f"m_eff{i}"] ** 3 * q["DeltaE"])

    best_E_f = None
    best_abs_remaining_charge = float("inf")
    E_fs = np.arange(0 * consts.EV, 0.8 * consts.EV, 1e-3 * consts.EV)
    for E_f in E_fs:
        fermi_dirac = 1 / (1 + torch.exp(params.BETA * (q["DeltaE"] - E_f)))
        integrand = dos * fermi_dirac
        # Particle, not charge density
        n = quantities.sum_dimension("DeltaE", integrand, q.grid) * params.E_STEP
        abs_remaining_charge = torch.abs(n - q[f"doping{i}"]).item()
        if abs_remaining_charge < best_abs_remaining_charge:
            best_abs_remaining_charge = abs_remaining_charge
            best_E_f = E_f

        E_f += 1e-3 * consts.EV

    assert best_E_f is not None
    assert best_E_f != E_fs[0] and best_E_f != E_fs[-1], E_f / consts.EV

    relative_remaining_charge = best_abs_remaining_charge / q[f"doping{i}"]
    print(
        f"Best fermi energy: {best_E_f / consts.EV} eV with a relative remaining charge of {relative_remaining_charge}"
    )

    return best_E_f


def get_phi_target(
    q: QuantityDict,
    *,
    i: int,
    contact: Contact,
    direction: int,
    get_derivative: bool = True,
):
    """
    Boundary conditions: Given phi{i_prev}, find phi{i} at the boundary between
    i and i_prev.
    i_prev is towards the input contact if direction == 1 and towards the output
    if direction == -1.
    """
    assert direction in (1, -1)

    i_prev = i - contact.direction * direction
    phi_target = torch.clone(q[f"phi{i_prev}_{contact}"])
    if get_derivative:
        phi_dx_target = (
            q[f"m_eff{i}"] / q[f"m_eff{i_prev}"] * q[f"phi{i_prev}_{contact}_dx"]
        )
    else:
        phi_dx_target = None

    return phi_target, phi_dx_target


def get_fermi_integral(*, m_eff, E_fermi, E):
    return (
        m_eff
        / (np.pi * consts.H_BAR**2 * params.BETA)
        * torch.log(1 + torch.exp(params.BETA * (E_fermi - E)))
    )


def get_n_contact(*, dos, fermi_integral, grid: Grid):
    integrand = dos * fermi_integral
    n_contact = quantities.sum_dimension("DeltaE", integrand, grid) * params.E_STEP
    return n_contact


def get_simple_phi(q: QuantityDict, *, i: int, contact: Contact):
    """Adapt get_simple_phi_dx when changing this!"""
    phi = torch.clone(q[f"phi_zero{i}_{contact}"])
    if params.use_phi_one:
        phi += q[f"phi_one{i}_{contact}"]

    return phi


def get_simple_phi_dx(q: QuantityDict, *, i: int, contact: Contact):
    """Dependent on get_simple_phi - not ideal, might be solved with lazy eval"""
    phi_dx = torch.clone(q[f"phi_zero{i}_{contact}_dx"])
    if params.use_phi_one:
        phi_dx += q[f"phi_one{i}_{contact}_dx"]

    return phi_dx


def a_output_transformation(
    a_output,
    *,
    q: QuantityDict,
    x_in,
    i: int,
    contact: Contact,
):
    if params.output_trafo == "none":
        return a_output

    if params.output_trafo == "polar":
        return torch.real(a_output) * torch.exp(1j * torch.imag(a_output))

    if params.output_trafo == "polar_times_x":
        return torch.real(a_output) * torch.exp(
            1j * torch.imag(a_output) * (q["x"] - x_in)
        )

    if params.output_trafo == "scaled_exp":
        a_output_real = torch.real(a_output)
        # k = params.K_OOM
        # k = 2 * np.pi / (10 * consts.NM)
        k = torch.sqrt(2 * q[f"m_eff{i}"] * q["DeltaE"]) / consts.H_BAR
        a_output_imag = torch.imag(a_output) * k * (q["x"] - x_in)
        a_output = a_output_real + 1j * a_output_imag
        return torch.exp(a_output)

    if params.output_trafo == "double_exp":
        k = torch.sqrt(2 * q[f"m_eff{i}"] * q["DeltaE"]) / consts.H_BAR
        return a_output * torch.exp(1j * a_output * k * (q["x"] - x_in))

    if params.output_trafo == "Veff":
        a_output_real = torch.real(a_output)
        Veff = torch.imag(a_output) * 0.1 * consts.EV
        k = torch.sqrt(2 * q[f"m_eff{i}"] * (q["DeltaE"] - Veff)) / consts.H_BAR
        return torch.exp(1j * k * (q["x"] - x_in))

    raise Exception(f"output_trafo {params.output_trafo} is unknown")


def b_output_transformation(b_output, **kwargs):
    return a_output_transformation(b_output, **kwargs)
