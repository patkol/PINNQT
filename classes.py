# Copyright (c) 2025 ETH Zurich, Patrice Kolb


import dataclasses
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Callable


@dataclass
class Contact:
    name: str
    index: int
    out_index: int
    direction: int  # i +/- direction gives the next/previous layer index.
    grid_name: str
    get_in_boundary_index: Callable[[int], int]  # layer -> boundary
    get_out_boundary_index: Callable[[int], int]
    get_in_layer_index: Callable[[int], int]  # boundary -> layer
    get_out_layer_index: Callable[[int], int]

    in_boundary_index: int = dataclasses.field(init=False)
    out_boundary_index: int = dataclasses.field(init=False)
    in_layer_index: int = dataclasses.field(init=False)  # First layer of the device
    out_layer_index: int = dataclasses.field(init=False)  # Last layer of the device

    def __post_init__(self):
        self.in_boundary_index = self.get_out_boundary_index(self.index)
        self.out_boundary_index = self.get_in_boundary_index(self.out_index)
        self.in_layer_index = self.get_out_layer_index(self.in_boundary_index)
        self.out_layer_index = self.get_in_layer_index(self.out_boundary_index)

    def __repr__(self):
        return self.name


@dataclass
class Device:
    """
    boundaries: [x_b0, ..., x_bN] with N the number of layers
    potentials: [V_0, ..., V_N+1] (including contacts),
                constants or functions of q, grid
    m_effs: [m_0, ..., m_N+1], like potentials
    dopings & permittivities: Same
    Layer i in [1,N] has x_b(i-1) on the left and x_bi on the right.
    """

    boundaries: Sequence[float]
    potentials: Sequence[float]
    m_effs: Sequence[float]
    dopings: Sequence[float]
    permittivities: Sequence[float]
    includes_contacts: bool

    device_start: float = dataclasses.field(init=False)
    device_end: float = dataclasses.field(init=False)
    n_layers: int = dataclasses.field(init=False)
    contacts: Sequence[Contact] = dataclasses.field(init=False)

    def __post_init__(self):
        self.device_start = self.boundaries[1 if self.includes_contacts else 0]
        self.device_end = self.boundaries[-2 if self.includes_contacts else -1]

        self.n_layers = len(self.boundaries) - 1

        N = self.n_layers

        assert len(self.boundaries) == N + 1
        assert len(self.potentials) == N + 2
        assert len(self.m_effs) == N + 2
        assert len(self.dopings) == N + 2
        assert len(self.permittivities) == N + 2
        assert len(self.boundaries) == N + 1
        assert sorted(self.boundaries) == self.boundaries, self.boundaries

        """
        Layers:    0 | 1 | 2 | 3
        Boundaries:  0   1   2
        """
        left_contact = Contact(
            name="L",
            index=0,
            out_index=N + 1,
            direction=1,
            grid_name=f"boundary{0}",
            get_in_boundary_index=lambda i: max(0, i - 1),
            get_out_boundary_index=lambda i: min(N, i),
            get_in_layer_index=lambda i: i,
            get_out_layer_index=lambda i: i + 1,
        )
        right_contact = Contact(
            name="R",
            index=N + 1,
            out_index=0,
            direction=-1,
            grid_name=f"boundary{N}",
            get_in_boundary_index=lambda i: min(N, i),
            get_out_boundary_index=lambda i: max(0, i - 1),
            get_in_layer_index=lambda i: i + 1,
            get_out_layer_index=lambda i: i,
        )

        self.contacts = [left_contact, right_contact]
