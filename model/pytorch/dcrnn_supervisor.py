import os
import time

import numpy as np
import torch

from lib import utils
from model.pytorch.dcrnn_model import EncoderModel, DecoderModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DCRNNSupervisor:
    def __init__(self, adj_mx, **kwargs):
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')

        self.max_grad_norm = self._train_kwargs.get('max_grad_norm', 1.)

        # logging.
        self._log_dir = self._get_log_dir(kwargs)
        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = utils.get_logger(self._log_dir, __name__, 'info.log', level=log_level)

        # data set
        self._data = utils.load_dataset(**self._data_kwargs)
        self.standard_scaler = self._data['scaler']

        self.num_nodes = int(self._model_kwargs.get('num_nodes', 1))
        self.input_dim = int(self._model_kwargs.get('input_dim', 1))
        self.seq_len = int(self._model_kwargs.get('seq_len'))  # for the encoder
        self.output_dim = int(self._model_kwargs.get('output_dim', 1))
        self.cl_decay_steps = int(self._model_kwargs.get('cl_decay_steps', 1000))
        self.use_curriculum_learning = bool(
            self._model_kwargs.get('use_curriculum_learning', False))
        self.horizon = int(self._model_kwargs.get('horizon', 1))  # for the decoder

        # setup model
        self.encoder_model = EncoderModel(True, adj_mx, **self._model_kwargs)
        self.decoder_model = DecoderModel(True, adj_mx, **self._model_kwargs)

    @staticmethod
    def _get_log_dir(kwargs):
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            batch_size = kwargs['data'].get('batch_size')
            learning_rate = kwargs['train'].get('base_lr')
            max_diffusion_step = kwargs['model'].get('max_diffusion_step')
            num_rnn_layers = kwargs['model'].get('num_rnn_layers')
            rnn_units = kwargs['model'].get('rnn_units')
            structure = '-'.join(
                ['%d' % rnn_units for _ in range(num_rnn_layers)])
            horizon = kwargs['model'].get('horizon')
            filter_type = kwargs['model'].get('filter_type')
            filter_type_abbr = 'L'
            if filter_type == 'random_walk':
                filter_type_abbr = 'R'
            elif filter_type == 'dual_random_walk':
                filter_type_abbr = 'DR'
            run_id = 'dcrnn_%s_%d_h_%d_%s_lr_%g_bs_%d_%s/' % (
                filter_type_abbr, max_diffusion_step, horizon,
                structure, learning_rate, batch_size,
                time.strftime('%m%d%H%M%S'))
            base_dir = kwargs.get('base_dir')
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def train(self, **kwargs):
        kwargs.update(self._train_kwargs)
        return self._train(**kwargs)

    def _train_one_batch(self, inputs, labels, batches_seen, encoder_optimizer,
                         decoder_optimizer, criterion):
        """

        :param inputs: shape (seq_len, batch_size, num_sensor, input_dim)
        :param labels: shape (horizon, batch_size, num_sensor, input_dim)
        :param encoder_optimizer:
        :param decoder_optimizer:
        :param criterion: minimize this criterion
        :return: loss?
        """

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        batch_size = inputs.size(1)

        inputs = inputs.view(self.seq_len, batch_size, self.num_nodes * self.input_dim)
        labels = labels[..., :self.output_dim].view(self.horizon, batch_size,
                                                    self.num_nodes * self.output_dim)

        loss = 0

        encoder_hidden_state = None
        for t in range(self.seq_len):
            _, encoder_hidden_state = self.encoder_model.forward(inputs[t], encoder_hidden_state)

        self._logger.info("Encoder complete, starting decoder")
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.output_dim))

        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []

        for t in range(self.horizon):
            decoder_output, decoder_hidden_state = self.decoder_model.forward(decoder_input,
                                                                              decoder_hidden_state)
            decoder_input = decoder_output

            outputs.append(decoder_output)

            if self.use_curriculum_learning:  # todo check for is_training (pytorch way?)
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]

            loss += criterion(self.standard_scaler.inverse_transform(decoder_output),
                              self.standard_scaler.inverse_transform(labels[t]))

        self._logger.info("Decoder complete, starting backprop")
        loss.backward()

        # gradient clipping - this does it in place
        torch.nn.utils.clip_grad_norm_(self.encoder_model.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.decoder_model.parameters(), self.max_grad_norm)

        encoder_optimizer.step()
        decoder_optimizer.step()

        outputs = torch.stack(outputs)
        return outputs.view(self.horizon, batch_size, self.num_nodes, self.output_dim), loss.item()

    def _train(self, base_lr,
               steps, patience=50, epochs=100,
               min_learning_rate=2e-6, lr_decay_ratio=0.1, log_every=10, save_model=1,
               test_every_n_epochs=10, **kwargs):
        # steps is used in learning rate - will see if need to use it?
        encoder_optimizer = torch.optim.Adam(self.encoder_model.parameters(), lr=base_lr)
        decoder_optimizer = torch.optim.Adam(self.encoder_model.parameters(), lr=base_lr)
        criterion = torch.nn.L1Loss()  # mae loss

        batches_seen = 0
        self._logger.info('Start training ...')
        for epoch_num in range(epochs):
            train_iterator = self._data['train_loader'].get_iterator()
            losses = []

            start_time = time.time()

            for _, (x, y) in enumerate(train_iterator):
                x = torch.from_numpy(x).float()
                y = torch.from_numpy(y).float()
                self._logger.debug("X: {}".format(x.size()))
                self._logger.debug("y: {}".format(y.size()))
                x = x.permute(1, 0, 2, 3)
                y = y.permute(1, 0, 2, 3)
                output, loss = self._train_one_batch(x, y, batches_seen, encoder_optimizer,
                                                     decoder_optimizer, criterion)
                losses.append(loss)
                batches_seen += 1

            end_time = time.time()
            if epoch_num % log_every == 0:
                message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f} ' \
                          'lr:{:.6f} {:.1f}s'.format(epoch_num, epochs, batches_seen,
                                                     np.mean(losses), 0.0,
                                                     0.0, (end_time - start_time))
                self._logger.info(message)

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))
