import torch
import matplotlib.pyplot as plt

import formulas


def test_smoothen_curve_functions():
    kwargs_dict = {
        "univariate_spline": {},
        "gaussian": {
            "sigma": 0.2,
            "cutoff_sigmas": 6,
        },
    }

    # x = torch.linspace(start=1, end=3, steps=50)
    x = torch.sort(torch.rand((100,)))[0] * 2 + 1
    y = torch.exp(1j * x * 2.7)
    y += torch.abs(torch.real(y)) - torch.real(y)
    plt.scatter(x, torch.real(y), label="Initial Real")
    plt.scatter(x, torch.imag(y), label="Initial Imag")
    for type_, fn in formulas.smoothen_curve_functions.items():
        smooth_y = fn(x, y, **kwargs_dict[type_])
        plt.plot(x, torch.real(smooth_y), label=type_)
        plt.plot(x, torch.imag(smooth_y), label=type_, linestyle="dashed")
        plt.legend()

    plt.show()


def test_smoothen_curve_gaussian():
    """
    This is mainly to check whether the non-equispaced smoothing has a bad effect
    on the derivatives - whether they jump where the resolution is low for example.
    But seems fine!
    """

    kwargs = {
        "sigma": 1,
        "cutoff_sigmas": 6,
    }

    x = torch.concatenate(
        (torch.tensor([-0.01, 0, 0.01]), torch.arange(start=0.1, end=4, step=0.2))
    )
    y = (x + 0.7) ** 2 + 1 * x
    smooth_y = formulas.smoothen_curve_gaussian(x, y, **kwargs)

    dy_dx = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    dsmooth_y_dx = (smooth_y[1:] - smooth_y[:-1]) / (x[1:] - x[:-1])
    x_for_dx = (x[1:] + x[:-1]) / 2
    dy_dx_dx = (dy_dx[1:] - dy_dx[:-1]) / (x_for_dx[1:] - x_for_dx[:-1])
    dsmooth_y_dx_dx = (dsmooth_y_dx[1:] - dsmooth_y_dx[:-1]) / (
        x_for_dx[1:] - x_for_dx[:-1]
    )
    x_for_dx_dx = (x_for_dx[1:] + x_for_dx[:-1]) / 2

    fig, axs = plt.subplots(3, 1, squeeze=False, sharex=True)

    axs[0, 0].plot(x, y, label="Initial")
    axs[0, 0].plot(x, smooth_y, label="Smoothed")
    axs[0, 0].legend()

    axs[1, 0].set_title("dx")
    axs[1, 0].plot(x_for_dx, dy_dx, label="Initial")
    axs[1, 0].plot(x_for_dx, dsmooth_y_dx, label="Smoothed")

    axs[2, 0].set_title("dx^2")
    axs[2, 0].plot(x_for_dx_dx, dy_dx_dx, label="Initial")
    axs[2, 0].plot(x_for_dx_dx, dsmooth_y_dx_dx, label="Smoothed")

    plt.show()


if __name__ == "__main__":
    test_smoothen_curve_gaussian()
