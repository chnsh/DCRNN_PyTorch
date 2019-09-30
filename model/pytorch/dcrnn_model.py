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
    def dcgru_layers(self):
        pass

    @staticmethod
    def _forward_layer(inputs, dcgru_layer, hidden_state):
        # inputs shape = (timesteps, batch_size, input_size)
        outputs = []
        for cell_input in inputs[:, ]:
            hidden_state = dcgru_layer(cell_input, hidden_state)
            outputs.append(hidden_state)

        return torch.cat(outputs, dim=1)  # runs in O(timesteps) not too slow

    def _forward_impl(self, inputs, hidden_state):
        """
        forward pass.

        :param inputs: shape (batch_size, timesteps, num_nodes * input_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size) -> optional, zeros if not provided
        :return: output: # shape (timesteps, batch_size, self.hidden_state_size)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size) (lower indices mean lower layers)
        """
        layer_input = inputs.permute(1, 0, 2)  # first axis is now timesteps
        if hidden_state is None:
            batch_size = inputs.size()[0]
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size),
                                       device=device)
        hidden = torch.empty_like(hidden_state)
        # noinspection PyTypeChecker
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            layer_states = self._forward_layer(layer_input, dcgru_layer, hidden_state[layer_num])
            # append last time step's hidden state
            hidden[layer_num] = layer_states[-1]
            layer_input = layer_states

        output = layer_input  # last layer's output
        return output, hidden


class EncoderModel(nn.Module, DCRNNModel):
    def __init__(self, is_training, scaler, adj_mx, **model_kwargs):
        super().__init__(is_training, scaler, adj_mx, **model_kwargs)
        # https://pytorch.org/docs/stable/nn.html#gru
        self.seq_len = int(model_kwargs.get('seq_len'))  # for the encoder

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
