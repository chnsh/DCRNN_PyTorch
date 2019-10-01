import torch
import torch.nn as nn
from abc import ABC, abstractmethod

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DCRNNModel(metaclass=ABC):
    def __init__(self, is_training, scale_factor, adj_mx, **model_kwargs):
        super().__init__()
        self.adj_mx = adj_mx
        self.is_training = is_training
        self.scale_factor = scale_factor
        self.max_diffusion_step = int(model_kwargs.get('max_diffusion_step', 2))
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.filter_type = model_kwargs.get('filter_type', 'laplacian')
        # self.max_grad_norm = float(model_kwargs.get('max_grad_norm', 5.0))
        self.num_nodes = int(model_kwargs.get('num_nodes', 1))
        self.num_rnn_layers = int(model_kwargs.get('num_rnn_layers', 1))
        self.rnn_units = int(model_kwargs.get('rnn_units'))
        self.input_dim = int(model_kwargs.get('input_dim', 1))
        self.hidden_state_size = self.num_nodes * self.rnn_units

    @abstractmethod
    @property
    def dcgru_layers(self):
        pass

    def _forward_cell(self, cell_input, prev_hidden_states):
        """
        Runs for 1 time step through all layers.
        :param cell_input: shape (batch_size, input feature size)
        :param prev_hidden_states: (num_layers, batch_size, hidden size)
        :return: output of cell: shape(batch_size, hidden size)
                all hidden states from layer: shape(num_layers, batch_size, hidden size)
        """
        hidden_states = []
        output = cell_input
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            hidden_state = dcgru_layer(output, prev_hidden_states[layer_num])
            hidden_states.append(hidden_state)
            output = hidden_state

        return output, torch.cat(hidden_states, dim=1)  # runs in O(num_layers) so not too slow #todo: check dim

    def _forward_impl(self, inputs, hidden_state):
        """
        forward pass.

        :param inputs: shape (batch_size, timesteps, input_size)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size) -> optional, zeros if not provided
        :return: output: # shape (timesteps, batch_size, self.hidden_state_size)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size) (lower indices mean lower layers)
        """
        batch_size, timesteps, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size),
                                       device=device)
        output = torch.empty((timesteps, batch_size, self.hidden_state_size))
        for t in range(timesteps):
            hidden_state = self.t_step_forward_pass(hidden_state, inputs, output, t)

        output = output.permute(1, 0, 2)
        return output, hidden_state

    @abstractmethod
    def t_step_forward_pass(self, hidden_state, inputs, output, t):
        """
        Implements the forward pass for timestep t.
        """
        # this is to accommodate curriculum learning
        pass


class EncoderModel(nn.Module, DCRNNModel):
    def __init__(self, is_training, scaler, adj_mx, **model_kwargs):
        super().__init__(is_training, scaler, adj_mx, **model_kwargs)
        # https://pytorch.org/docs/stable/nn.html#gru
        self.seq_len = int(model_kwargs.get('seq_len'))  # for the encoder

    def t_step_forward_pass(self, hidden_state, inputs, output, t):
        cell_input = inputs[:, t, :]  # (batch_size, input_size)
        cell_output, hidden_state = self._forward_cell(cell_input, hidden_state)
        output[t] = cell_output
        return hidden_state

    def dcgru_layers(self):
        # input shape is supposed to be Input (batch_size, timesteps, num_sensor*input_dim)
        # first layer takes input shape and subsequent layer take input from the first layer
        return [nn.GRUCell(input_size=self.num_nodes * self.input_dim,
                           hidden_size=self.hidden_state_size,
                           bias=True)] + [nn.GRUCell(input_size=self.hidden_state_size,
                                                     hidden_size=self.hidden_state_size,
                                                     bias=True) for _ in
                                          range(self.num_rnn_layers - 1)]

    def forward(self, inputs, hidden_state=None):
        """
        Encoder forward pass.

        :param inputs: shape (batch_size, timesteps, num_nodes * input_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size) -> optional, zeros if not provided
        :return: output: # shape (timesteps, batch_size, self.hidden_state_size)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size) (lower indices mean lower layers)
        """
        return self._forward_impl(inputs, hidden_state)


class DecoderModel(nn.Module, DCRNNModel):
    def __init__(self, is_training, scale_factor, adj_mx, **model_kwargs):
        super().__init__(is_training, scale_factor, adj_mx, **model_kwargs)
        self.output_dim = int(model_kwargs.get('output_dim', 1))
        self.use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))
        self.horizon = int(model_kwargs.get('horizon', 1))  # for the decoder
        self.projection_layer = nn.Linear(self.hidden_state_size, self.num_nodes * self.output_dim)

    def dcgru_layers(self):
        return [nn.GRUCell(input_size=self.num_nodes * self.output_dim,
                           hidden_size=self.hidden_state_size,
                           bias=True)] + [nn.GRUCell(input_size=self.hidden_state_size,
                                                     hidden_size=self.hidden_state_size,
                                                     bias=True) for _ in
                                          range(self.num_rnn_layers - 1)]

    def forward(self, inputs, hidden_state=None):
        """
        Decoder forward pass.

        :param inputs: shape (batch_size, timesteps, num_nodes * input_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size) -> optional, zeros if not provided
        :return: output: # shape (timesteps, batch_size, self.num_nodes * self.output_dim)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size) (lower indices mean lower layers)
        """
        output, hidden = self._forward_impl(inputs, hidden_state)
        return self.projection_layer(output), hidden_state
