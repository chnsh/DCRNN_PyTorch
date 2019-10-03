import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DCRNNModel:
    def __init__(self, is_training, adj_mx, **model_kwargs):
        self.adj_mx = adj_mx
        self.is_training = is_training
        self.max_diffusion_step = int(model_kwargs.get('max_diffusion_step', 2))
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.filter_type = model_kwargs.get('filter_type', 'laplacian')
        # self.max_grad_norm = float(model_kwargs.get('max_grad_norm', 5.0))
        self.num_nodes = int(model_kwargs.get('num_nodes', 1))
        self.num_rnn_layers = int(model_kwargs.get('num_rnn_layers', 1))
        self.rnn_units = int(model_kwargs.get('rnn_units'))
        self.hidden_state_size = self.num_nodes * self.rnn_units


class EncoderModel(nn.Module, DCRNNModel):
    def __init__(self, is_training, adj_mx, **model_kwargs):
        # super().__init__(is_training, adj_mx, **model_kwargs)
        # https://pytorch.org/docs/stable/nn.html#gru
        nn.Module.__init__(self)
        DCRNNModel.__init__(self, is_training, adj_mx, **model_kwargs)
        self.input_dim = int(model_kwargs.get('input_dim', 1))
        self.seq_len = int(model_kwargs.get('seq_len'))  # for the encoder
        self.dcgru_layers = nn.ModuleList([nn.GRUCell(input_size=self.num_nodes * self.input_dim,
                                                      hidden_size=self.hidden_state_size,
                                                      bias=True)] + [
                                              nn.GRUCell(input_size=self.hidden_state_size,
                                                         hidden_size=self.hidden_state_size,
                                                         bias=True) for _ in
                                              range(self.num_rnn_layers - 1)])

    def forward(self, inputs, hidden_state=None):
        """
        Encoder forward pass.

        :param inputs: shape (batch_size, self.num_nodes * self.input_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.hidden_state_size)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size),
                                       device=device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        return output, torch.stack(hidden_states)  # runs in O(num_layers) so not too slow


class DecoderModel(nn.Module, DCRNNModel):
    def __init__(self, is_training, adj_mx, **model_kwargs):
        # super().__init__(is_training, adj_mx, **model_kwargs)
        nn.Module.__init__(self)
        DCRNNModel.__init__(self, is_training, adj_mx, **model_kwargs)
        self.adj_mx = adj_mx
        self.is_training = is_training
        self.max_diffusion_step = int(model_kwargs.get('max_diffusion_step', 2))
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.filter_type = model_kwargs.get('filter_type', 'laplacian')
        # self.max_grad_norm = float(model_kwargs.get('max_grad_norm', 5.0))
        self.num_nodes = int(model_kwargs.get('num_nodes', 1))
        self.num_rnn_layers = int(model_kwargs.get('num_rnn_layers', 1))
        self.rnn_units = int(model_kwargs.get('rnn_units'))
        self.hidden_state_size = self.num_nodes * self.rnn_units
        self.output_dim = int(model_kwargs.get('output_dim', 1))
        self.use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))
        self.horizon = int(model_kwargs.get('horizon', 1))  # for the decoder
        self.projection_layer = nn.Linear(self.hidden_state_size, self.num_nodes * self.output_dim)
        self.dcgru_layers = nn.ModuleList([nn.GRUCell(input_size=self.num_nodes * self.output_dim,
                                                      hidden_size=self.hidden_state_size,
                                                      bias=True)] + [
                                              nn.GRUCell(input_size=self.hidden_state_size,
                                                         hidden_size=self.hidden_state_size,
                                                         bias=True) for _ in
                                              range(self.num_rnn_layers - 1)])

    def forward(self, inputs, hidden_state=None):
        """
        Decoder forward pass.

        :param inputs: shape (batch_size, self.num_nodes * self.output_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.num_nodes * self.output_dim)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        return self.projection_layer(output), torch.stack(hidden_states)
