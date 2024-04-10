import torch

from kolpinn import mathematics


# General
simulated_device_name = 'rtd1'
seed = 0
device = 'cuda'
si_real_dtype = torch.float64
si_complex_dtype = torch.complex128

# Models
loaded_parameters_index = None
# Whether to use the state of the saved optimizer (possibly overwriting optimizer_kwargs)
load_optimizer = False
load_scheduler = False
save_optimizer = False
n_hidden_layers = 9
n_neurons_per_hidden_layer = 100
activation_function = torch.nn.Tanh()
model_dtype = torch.float32

# Training
max_n_training_steps = 50
max_time = None
min_loss = None
report_each = 1
Optimizer = torch.optim.LBFGS
optimizer_kwargs = {'lr': 1, 'tolerance_grad': 0, 'tolerance_change': 0}
Scheduler = None # torch.optim.lr_scheduler.ReduceLROnPlateau
scheduler_kwargs = None # {'factor': 0.5, 'patience': 5, 'cooldown': 5, 'min_lr': 1e-6, 'eps': 0}
#loss_function = lambda x: mathematics.complex_abs2(x)**2
#loss_aggregate_function = lambda losses: torch.sqrt(sum(losses))
loss_function = lambda x: mathematics.complex_abs2(x)
loss_aggregate_function = lambda losses: sum(losses)
fd_first_derivatives = True
fd_second_derivatives = True
# Whether the voltage/energy is an input to the NN
continuous_voltage = True
continuous_energy = True
# Whether to use the weights from the previous energy step
continuous_training = loaded_parameters_index is None

# Resolution
N_x = 100
N_x_training = N_x
N_x_validation = N_x
batch_size_x = N_x
