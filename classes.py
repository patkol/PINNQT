# Copyright (c) 2024 ETH Zurich, Patrice Kolb


import dataclasses
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Callable


class Contact:
    def __init__(
        self,
        name: str,
        *,
        index: int,
        out_index: int,
        grid_name: str,
        incoming_coeff_in_name: str,
        get_in_boundary_index: Callable[[int], int],
        get_out_boundary_index: Callable[[int], int],
        get_previous_layer_index: Callable[[int], int],
        get_next_layer_index: Callable[[int], int],
    ):
        assert name in ('L', 'R')
        assert incoming_coeff_in_name in ('a', 'b')

        self.name = name
        self.index = index
        self.out_index = out_index
        self.grid_name = grid_name
        self.get_in_boundary_index = get_in_boundary_index
        self.get_out_boundary_index = get_out_boundary_index
        self.get_previous_layer_index = get_previous_layer_index
        self.get_next_layer_index = get_next_layer_index
        self.incoming_coeff_in_name = incoming_coeff_in_name
        self.incoming_coeff_out_name = 'b' if incoming_coeff_in_name == 'a' else 'a'
        self.outgoing_coeff_in_name = self.incoming_coeff_out_name
        self.outgoing_coeff_out_name = self.incoming_coeff_in_name

        self.in_boundary_index = self.get_out_boundary_index(self.index)
        self.out_boundary_index = self.get_in_boundary_index(self.out_index)
        self.out_boundary_name = f'boundary{self.out_boundary_index}'

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
    Contacts: Sequence[Contact] = dataclasses.field(init=False)

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

        left_contact = Contact(
            name='L',
            index=0,
            out_index=N + 1,
            grid_name=f'boundary{0}',
            incoming_coeff_in_name='a',
            get_in_boundary_index=lambda i: max(0, i - 1),
            get_out_boundary_index=lambda i: min(N, i),
            get_previous_layer_index=lambda i: i - 1,
            get_next_layer_index=lambda i: i + 1,
        )
        right_contact = Contact(
            name='R',
            index=N + 1,
            out_index=0,
            grid_name=f'boundary{N}',
            incoming_coeff_in_name='b',
            get_in_boundary_index=lambda i: min(N, i),
            get_out_boundary_index=lambda i: max(0, i - 1),
            get_previous_layer_index=lambda i: i + 1,
            get_next_layer_index=lambda i: i - 1,
        )

        self.contacts = [left_contact, right_contact]
