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


if __name__ == "__main__":
    test_smoothen_curve_functions()
