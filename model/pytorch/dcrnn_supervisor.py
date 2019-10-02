import numpy as np
import torch

from model.pytorch.dcrnn_model import EncoderModel, DecoderModel


class DCRNNSupervisor:
    def __init__(self, adj_mx, **kwargs):
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')

        self.num_nodes = int(self._model_kwargs.get('num_nodes', 1))
        self.input_dim = int(self._model_kwargs.get('input_dim', 1))
        self.seq_len = int(self._model_kwargs.get('seq_len'))  # for the encoder
        self.output_dim = int(self._model_kwargs.get('output_dim', 1))
        self.use_curriculum_learning = bool(
            self._model_kwargs.get('use_curriculum_learning', False))
        self.horizon = int(self._model_kwargs.get('horizon', 1))  # for the decoder

    def train(self, encoder_model: EncoderModel, decoder_model: DecoderModel, **kwargs):
        kwargs.update(self._train_kwargs)
        return self._train(**kwargs)

    def _train_one_batch(self, inputs, labels, encoder_model: EncoderModel,
                         decoder_model: DecoderModel, encoder_optimizer,
                         decoder_optimizer, criterion):
        """

        :param inputs: shape (seq_len, batch_size, num_sensor, input_dim)
        :param labels: shape (horizon, batch_size, num_sensor, input_dim)
        :param encoder_model:
        :param decoder_model:
        :param encoder_optimizer:
        :param decoder_optimizer:
        :param criterion: minimize this criterion
        :return: loss?
        """

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        batch_size = inputs.size(1)

        inputs = inputs.view(self.seq_len, batch_size, self.num_nodes * self.input_dim)
        labels = labels.view(self.horizon, batch_size, self.num_nodes * self.output_dim)

        loss = 0

        encoder_hidden_state = None
        for t in range(self.seq_len):
            _, encoder_hidden_state = encoder_model.forward(inputs[t], encoder_hidden_state)

        go_symbol = torch.zeros((batch_size, self.num_nodes * self.output_dim))

        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        for t in range(self.horizon):
            decoder_output, decoder_hidden_state = decoder_model.forward(decoder_input,
                                                                         decoder_hidden_state)
            decoder_input = decoder_output

            if self.use_curriculum_learning:  # todo check for is_training (pytorch way?)
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold():
                    decoder_input = labels[t]

            loss += criterion(decoder_output, labels[t])

        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        return loss.item()

    def _train(self, encoder_model: EncoderModel, decoder_model: DecoderModel, base_lr, epoch,
               steps, patience=50, epochs=100,
               min_learning_rate=2e-6, lr_decay_ratio=0.1, save_model=1,
               test_every_n_epochs=10):
        pass

    def _compute_sampling_threshold(self):
        return 1.0  # todo
