import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DCRNNModel(nn.Module):
    def __init__(self, is_training, scale_factor, adj_mx, **model_kwargs):
        super().__init__()
        self.adj_mx = adj_mx
        self.is_training = is_training
        self.scale_factor = scale_factor
        self.max_diffusion_step = int(model_kwargs.get('max_diffusion_step', 2))
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.filter_type = model_kwargs.get('filter_type', 'laplacian')
        self.horizon = int(model_kwargs.get('horizon', 1))  # for the decoder
        # self.max_grad_norm = float(model_kwargs.get('max_grad_norm', 5.0))
        self.num_nodes = int(model_kwargs.get('num_nodes', 1))
        self.num_rnn_layers = int(model_kwargs.get('num_rnn_layers', 1))
        self.rnn_units = int(model_kwargs.get('rnn_units'))
        self.seq_len = int(model_kwargs.get('seq_len'))  # for the encoder
        self.use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))
        self.input_dim = int(model_kwargs.get('input_dim', 1))
        self.output_dim = int(model_kwargs.get('output_dim', 1))


class EncoderModel(DCRNNModel):
    def __init__(self, is_training, scaler, adj_mx, **model_kwargs):
        super().__init__(is_training, scaler, adj_mx, **model_kwargs)

        # https://pytorch.org/docs/stable/nn.html#gru

        # since input shape is Input (batch_size, timesteps, num_sensor*input_dim),batch_first=True
        self.dcgru = nn.GRU(input_size=self.num_nodes * self.input_dim,
                            hidden_size=self.rnn_units,
                            num_layers=self.num_rnn_layers,
                            batch_first=True)

    def forward(self, inputs, hidden_state=None):
        # is None okay?
        return self.dcgru(inputs, hidden_state)
