from typing import Optional

import torch
from torch import Tensor

from lib import utils


class FCLayerParams:
    def __init__(self, rnn_network: torch.nn.RNN):
        self._rnn_network = rnn_network
        self._params_dict = {}
        self._biases_dict = {}

    def get_weights(self, shape):
        if shape not in self._params_dict:
            nn_param = torch.nn.init.xavier_normal(torch.empty(*shape))
            self._params_dict[shape] = nn_param
            self._rnn_network.register_parameter('fc_weight_{}'.format(str(shape)), nn_param)
        return self._params_dict[shape]

    def get_biases(self, length, bias_start=0.0):
        if length not in self._biases_dict:
            biases = torch.nn.init.constant(torch.empty(length), bias_start)
            self._biases_dict[length] = biases
            self._rnn_network.register_parameter('fc_biases_{}'.format(str(length)), biases)

        return self._biases_dict[length]


class DCGRUCell(torch.nn.RNN):
    def __init__(self, num_units, adj_mx, max_diffusion_step, num_nodes, input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 num_proj=None,
                 nonlinearity='tanh', filter_type="laplacian", use_gc_for_ru=True):
        """

        :param num_units:
        :param adj_mx:
        :param max_diffusion_step:
        :param num_nodes:
        :param input_size:
        :param num_proj:
        :param nonlinearity:
        :param filter_type: "laplacian", "random_walk", "dual_random_walk".
        :param use_gc_for_ru: whether to use Graph convolution to calculate the reset and update gates.
        """
        super(DCGRUCell, self).__init__(input_size, hidden_size, bias=True,
                                        # bias param does not exist in tf code?
                                        num_layers=num_layers,
                                        nonlinearity=nonlinearity)
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        # support other nonlinearities up here?
        self._num_nodes = num_nodes
        self._num_proj = num_proj
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._supports = []
        self._use_gc_for_ru = use_gc_for_ru
        supports = []
        if filter_type == "laplacian":
            supports.append(utils.calculate_scaled_laplacian(adj_mx, lambda_max=None))
        elif filter_type == "random_walk":
            supports.append(utils.calculate_random_walk_matrix(adj_mx).T)
        elif filter_type == "dual_random_walk":
            supports.append(utils.calculate_random_walk_matrix(adj_mx).T)
            supports.append(utils.calculate_random_walk_matrix(adj_mx.T).T)
        else:
            supports.append(utils.calculate_scaled_laplacian(adj_mx))
        for support in supports:
            self._supports.append(self._build_sparse_matrix(support))

        self._proj_weights = torch.nn.Parameter(torch.randn(self._num_units, self._num_proj))
        self._fc_params = FCLayerParams(self)

    @property
    def state_size(self):
        return self._num_nodes * self._num_units

    @property
    def output_size(self):
        output_size = self._num_nodes * self._num_units
        if self._num_proj is not None:
            output_size = self._num_nodes * self._num_proj
        return output_size

    def forward(self, input: Tensor, hx: Optional[Tensor] = ...):
        """Gated recurrent unit (GRU) with Graph Convolution.
        :param input: (B, num_nodes * input_dim)

        :return
        - Output: A `2-D` tensor with shape `[batch_size x self.output_size]`.
        - New state: Either a single `2-D` tensor, or a tuple of tensors matching
            the arity and shapes of `state`
        """
        output_size = 2 * self._num_units
        # We start with bias of 1.0 to not reset and not update.
        if self._use_gc_for_ru:
            fn = self._gconv
        else:
            fn = self._fc
        value = torch.sigmoid(fn(input, hx, output_size, bias_start=1.0))
        value = torch.reshape(value, (-1, self._num_nodes, output_size))
        r, u = torch.split(tensor=value, split_size_or_sections=2, dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))

        c = self._gconv(input, r * hx, self._num_units)
        if self._activation is not None:
            c = self._activation(c)

        output = new_state = u * hx + (1 - u) * c
        if self._num_proj is not None:
            batch_size = input.shape[0]
            output = torch.reshape(new_state, shape=(-1, self._num_units))
            output = torch.reshape(torch.matmul(output, self._proj_weights),
                                   shape=(batch_size, self.output_size))
        return output, new_state

    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def _fc(self, inputs, state, output_size, bias_start=0.0):
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = torch.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=-1)
        input_size = inputs_and_state.shape[-1]
        weights = self._fc_params.get_weights((input_size, output_size))
        value = torch.sigmoid(torch.matmul(inputs_and_state, weights))
        biases = self._fc_params.get_biases(output_size, bias_start)
        value += biases
        return value