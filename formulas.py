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


def get_linear_transition(
    x: np.ndarray, *, x0: float, x1: float, y0: float, y1: float
) -> np.ndarray:
    distance_factor = (x - x0) / (x1 - x0)
    # Cap the factor beyond the limits
    distance_factor[distance_factor < 0] = 0
    distance_factor[distance_factor > 1] = 1
    y = y0 + distance_factor * (y1 - y0)

    return y


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


def smoothen_curve_univariate_spline(x: torch.Tensor, y: torch.Tensor):
    real_spline = scipy.interpolate.UnivariateSpline(
        x.cpu(),
        torch.real(y).cpu(),
        s=1,
    )
    imag_spline = scipy.interpolate.UnivariateSpline(
        x.cpu(),
        torch.imag(y).cpu(),
        s=1,
    )

    return torch.from_numpy(real_spline(x.cpu()) + 1j * imag_spline(x.cpu()))


def smoothen_curve_gaussian(
    x: torch.Tensor, y: torch.Tensor, *, sigma: float, cutoff_sigmas: float
):
    """
    x: 1D Tensor
    y[i, ...]: values at x[i]
    Similar to https://stackoverflow.com/a/24145141
    """

    N = len(x)
    assert y.shape[0] == N

    avg_dx = ((x[-1] - x[0]) / (N - 1)).item()
    cutoff = cutoff_sigmas * sigma
    N_cutoff = int(np.ceil(cutoff / avg_dx))
    x_expansion_left = torch.linspace(
        start=x[0] - cutoff, end=x[0] - avg_dx, steps=N_cutoff
    )
    x_expansion_right = torch.linspace(
        start=x[-1] + avg_dx, end=x[-1] + cutoff, steps=N_cutoff
    )
    x_expanded = torch.cat((x_expansion_left, x, x_expansion_right))
    expansion_shape = [1] * len(y.shape)
    expansion_shape[0] = -1
    y_expansion_left = (y[1, ...] - y[0, ...]) / (x[1] - x[0]) * (
        x_expansion_left.reshape(expansion_shape) - x[0]
    ) + y[0, ...]
    y_expansion_right = (y[-1, ...] - y[-2, ...]) / (x[-1] - x[-2]) * (
        x_expansion_right.reshape(expansion_shape) - x[-1]
    ) + y[-1, ...]
    y_expanded = torch.cat((y_expansion_left, y, y_expansion_right))

    assert len(x_expanded) == len(y_expanded)
    assert len(x_expanded) == N + 2 * N_cutoff

    smooth_y = torch.zeros_like(y)
    print("Smoothing...")
    # OPTIM: vectorize
    for i in range(N):
        i_expanded = i + N_cutoff
        window = slice(i_expanded - N_cutoff, i_expanded + N_cutoff + 1)
        delta_x = x_expanded[window] - x_expanded[i_expanded]
        weights = torch.exp(-(delta_x**2) / 2 / sigma**2)
        weights /= sum(weights)
        smooth_y[i, ...] = torch.sum(
            y_expanded[window, ...] * weights.reshape(expansion_shape), axis=0
        )
    print("Done")

    return smooth_y


smoothen_curve_functions = {
    "univariate_spline": smoothen_curve_univariate_spline,
    "gaussian": smoothen_curve_gaussian,
}


smoothen_curve_functions_is_vectorized = {
    "univariate_spline": False,
    "gaussian": True,
}


def smoothen(
    quantity: torch.Tensor, grid: Grid, dim_label: str, *, method: str, **kwargs
):
    """
    Smoothen `quantity` on `grid` along `dim_label`.
    `grid` must be sorted along `dim_label`.
    `method`:
        "gaussian": Convolve a gaussian (works for non-equispaced data)
        "univariate_spline": For backwards compatibility, does not behave smoothly wrt.
            changes in `quantity`.
    """

    assert quantities.compatible(quantity, grid)

    if smoothen_curve_functions_is_vectorized[method]:
        dim_index = grid.index[dim_label]
        quantity = torch.swapaxes(quantity, 0, dim_index)
        smooth_quantity = smoothen_curve_functions[method](
            grid[dim_label], quantity, **kwargs
        )
        smooth_quantity = torch.swapaxes(smooth_quantity, 0, dim_index)

        return smooth_quantity

    other_dim_labels = copy.copy(grid.dimensions_labels)
    other_dim_labels.remove(dim_label)
    other_ranges = [
        range(quantity.size(grid.index[other_dim_label]))
        for other_dim_label in other_dim_labels
    ]

    smoothened_quantity = torch.zeros_like(quantity)

    # The smoothing function only works with 1D data, so we have to
    # call it for all combinations of the other dimensions.
    for other_indices in itertools.product(*other_ranges):
        # Get current slice
        slices = [slice(None)] * grid.n_dim
        for other_dim_label, other_index in zip(other_dim_labels, other_indices):
            slices[grid.index[other_dim_label]] = other_index

        # Smoothen slice
        smoothened_quantity[slices] = smoothen_curve_functions[method](
            grid[dim_label], quantity[slices], **kwargs
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


def get_V_voltage(q: QuantityDict, x_L, x_R):
    """
    This does not take the permittivities into account, it's just meant as a first guess
    """
    distance_factor = (q["x"] - x_L) / (x_R - x_L)
    # Cap the factor beyond the device limits, these regions correspond to
    # the contacts
    distance_factor[distance_factor < 0] = 0
    distance_factor[distance_factor > 1] = 1
    V_voltage = -q["voltage"] * consts.EV * distance_factor

    return V_voltage


def get_smooth_V_voltage(q: QuantityDict, x_L, x_R):
    V_L = 0
    V_R = -q["voltage"] * consts.EV
    V_voltage = smooth_transition(q["x"], x0=x_L, x1=x_R, y0=V_L, y1=V_R)

    return V_voltage


def get_V_el_guess(q, guess_type: str, **kwargs):
    if guess_type == "zero":
        return torch.zeros_like(q["x"])

    if guess_type == "rtd":
        V_ext_range = (
            kwargs["x_L"] - kwargs["dx_smoothing"] / 2,
            kwargs["x_R"] + kwargs["dx_smoothing"] / 2,
        )
        V_drop = get_smooth_V_voltage(
            q,
            *V_ext_range,
        )
        V_mid = smooth_rectangle(q["x"], **kwargs)
        return V_drop + V_mid

    if guess_type == "transistor":
        x_ramp_L = kwargs["x_gate_L"] - kwargs["ramp_size"]
        x_ramp_R = kwargs["x_gate_R"] + kwargs["ramp_size"]
        V_channel = kwargs["V_channel"] - q["voltage2"] * consts.EV
        V_drain = -q["voltage"] * consts.EV
        V_ext = get_linear_transition(
            q["x"],
            x0=x_ramp_L,
            x1=kwargs["x_gate_L"],
            y0=0,
            y1=V_channel,
        )
        V_ext += get_linear_transition(
            q["x"],
            x0=kwargs["x_gate_R"],
            x1=x_ramp_R,
            y0=0,
            y1=V_drain - V_channel,
        )
        return V_ext

    if guess_type == "transistor_smooth":
        x_ramp_L = kwargs["x_gate_L"] - kwargs["ramp_size"]
        x_ramp_R = kwargs["x_gate_R"] + kwargs["ramp_size"]
        V_channel = kwargs["V_channel"] - q["voltage2"] * consts.EV
        V_drain = -q["voltage"] * consts.EV
        V_ext = smooth_transition(
            q["x"],
            x0=x_ramp_L,
            x1=kwargs["x_gate_L"],
            y0=0,
            y1=V_channel,
        )
        V_ext = V_ext + smooth_transition(
            q["x"],
            x0=kwargs["x_gate_R"],
            x1=x_ramp_R,
            y0=0,
            y1=V_drain - V_channel,
        )
        return V_ext

    raise Exception(f"Unknown V_el guess type: {guess_type}")


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

    def get_charge_mismatch(E_f: float):
        fermi_dirac = 1 / (1 + torch.exp(params.BETA * (q["DeltaE"] - E_f)))
        integrand = dos * fermi_dirac
        # Particle, not charge density
        n = quantities.sum_dimension("DeltaE", integrand, q.grid) * params.E_STEP
        abs_remaining_charge = torch.abs(n - q[f"doping{i}"]).item()

        return abs_remaining_charge

    res = scipy.optimize.minimize_scalar(
        get_charge_mismatch, bounds=params.E_fermi_search_range, options={"disp": 1}
    )
    assert res.success, res.message
    best_E_f = res.x
    relative_remaining_charge = res.fun / q[f"doping{i}"]
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
