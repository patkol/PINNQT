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

    def __repr__(self):
        return self.name