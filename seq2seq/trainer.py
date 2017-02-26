# -*- coding: utf-8 -*-

import logging

from utils.data_utils import read_data, sentence2id, load_vocab
from symbol import sym_gen
import mxnet as mx
import os
from metric.metric import Perplexity
import sys
from utils.data_utils import load_vocab, get_enc_dec_text_id
from common.constant import special_words
from seq2seq_bucket_io import EncoderDecoderIter, DummyIter
from masked_bucket_io import MaskedBucketSentenceIter
import time


class trainer(object):
    def __init__(self, train_source_path, train_target_path, vocabulary_path):
        self.train_source_path = train_source_path
        self.train_target_path = train_target_path
        self.vocabulary_path = vocabulary_path

    def set_mxnet_parameter(self, **kwargs):
        mxnet_parameter_defaults = {
            "kv_store": "local",
            "monitor_interval": 2,
            "log_level": logging.ERROR,
            "log_path": './logs',
            "save_checkpoint_freq": 100,
            "enable_evaluation": False
        }
        for (parameter, default) in mxnet_parameter_defaults.iteritems():
            setattr(self, parameter, kwargs.get(parameter, default))
        return self

    def set_model_parameter(self, **kwargs):
        model_parameter_defaults = {
            "s_layer_num": 1,
            "s_hidden_unit": 512,
            "s_embed_size": 512,
            "t_layer_num": 1,
            "t_hidden_unit": 512,
            "t_embed_size": 512,
            "buckets": [(3, 10), (3, 20), (5, 20), (7, 30)]
        }
        for (parameter, default) in model_parameter_defaults.iteritems():
            setattr(self, parameter, kwargs.get(parameter, default))
        return self

    def set_train_parameter(self, **kwargs):
        train_parameter_defaults = {
            "s_dropout": 0.5,
            "t_dropout": 0.5,
            "load_epoch": 0,
            "model_prefix": "query2vec",
            "device_mode": "cpu",
            "devices": 0,
            "lr_factor": None,
            "lr": 0.05,
            "wd": 0.0005,
            "train_max_samples": sys.maxsize,
            "momentum": 0.1,
            "show_every_x_batch": 10,
            "optimizer": 'sgd',
            "batch_size": 128,
            "num_epoch": 65535,
            "num_examples": 65535
        }
        for (parameter, default) in train_parameter_defaults.iteritems():
            setattr(self, parameter, kwargs.get(parameter, default))
        return self

    def _get_lr_scheduler(self, kv):
        if self.lr_factor is None or self.lr_factor >= 1:
            return (self.lr, None)
        epoch_size = self.num_examples / self.batch_size
        if 'dist' in self.kv_store:
            epoch_size /= kv.num_workers
        begin_epoch = self.load_epoch if self.load_epoch else 0
        step_epochs = [int(l) for l in self.lr_step_epochs.split(',')]
        lr = self.lr
        for s in step_epochs:
            if begin_epoch >= s:
                lr *= self.lr_factor
        if lr != self.lr:
            logging.info('Adjust learning rate to %e for epoch %d' % (lr, begin_epoch))

        steps = [epoch_size * (x - begin_epoch) for x in step_epochs if x - begin_epoch > 0]
        return (lr, mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=self.lr_factor))

    def _load_model(self, rank=0):
        if self.load_epoch is None:
            return (None, None, None)
        assert self.model_prefix is not None
        model_prefix = self.model_prefix
        if rank > 0 and os.path.exists("%s-%d-symbol.json" % (model_prefix, rank)):
            model_prefix += "-%d" % (rank)
        sym, arg_params, aux_params = mx.model.load_checkpoint(
            model_prefix, self.load_epoch)
        logging.info('Loaded model %s_%04d.params', model_prefix, self.load_epoch)
        return (sym, arg_params, aux_params)

    def _save_model(self, rank=0, period=10):
        if self.model_prefix is None:
            return None
        dst_dir = os.path.dirname(self.model_prefix)
        if not os.path.isdir(dst_dir):
            os.mkdir(dst_dir)
        return mx.callback.do_checkpoint(self.model_prefix if rank == 0 else "%s-%d" % (
            self.model_prefix, rank), period)

    def _initial_log(self, kv, log_level):
        # logging
        logging.basicConfig(format='Node[' + str(kv.rank) + '] %(asctime)s %(levelname)s:%(name)s:%(message)s',
                            level=log_level,
                            datefmt='%H:%M:%S')
        file_handler = logging.FileHandler(os.path.join(self.log_path, time.strftime("%Y%m%d-%H%M%S") + '.logs'))
        file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)-5.5s:%(name)s] %(message)s'))
        logging.root.addHandler(file_handler)

    def get_LSTM_shape(self):
        # initalize states for LSTM

        forward_source_init_c = [('forward_source_l%d_init_c' % l, (self.batch_size, self.s_hidden_unit)) for l in
                                 range(self.s_layer_num)]
        forward_source_init_h = [('forward_source_l%d_init_h' % l, (self.batch_size, self.s_hidden_unit)) for l in
                                 range(self.s_layer_num)]
        backward_source_init_c = [('backward_source_l%d_init_c' % l, (self.batch_size, self.s_hidden_unit)) for l in
                                  range(self.s_layer_num)]
        backward_source_init_h = [('backward_source_l%d_init_h' % l, (self.batch_size, self.s_hidden_unit)) for l in
                                  range(self.s_layer_num)]
        source_init_states = forward_source_init_c + forward_source_init_h + backward_source_init_c + backward_source_init_h

        target_init_c = [('target_l%d_init_c' % l, (self.batch_size, self.t_hidden_unit)) for l in
                         range(self.t_layer_num)]
        # target_init_h = [('target_l%d_init_h' % l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
        target_init_states = target_init_c
        return source_init_states, target_init_states

    def print_all_variable(self):
        for arg, value in self.__dict__.iteritems():
            logging.info("%s: %s" % (arg, value))

    def train(self):
        # kvstore
        self.kv = mx.kvstore.create(self.kv_store)
        self._initial_log(self.kv, self.log_level)

        self.print_all_variable()

        # load vocabulary
        vocab = load_vocab(self.vocabulary_path)

        vocab_size = len(vocab)
        self.vocab_size = vocab_size
        logging.info('vocab size: {0}'.format(vocab_size))

        # get states shapes
        source_init_states, target_init_states = self.get_LSTM_shape()

        # build data iterator

        #enc = get_enc_dec_text_id(self.train_source_path, vocab)
        #dec = get_enc_dec_text_id(self.train_source_path, vocab)
        #data_loader = EncoderDecoderIter(enc, dec, special_words.get('pad_word'), special_words.get('eos_word'), target_init_states, batch_size=self.batch_size)
        # build data iterator
        data_loader = MaskedBucketSentenceIter(self.train_source_path, self.train_target_path, vocab,
                                               vocab,
                                               self.buckets, self.batch_size,
                                               source_init_states, target_init_states, seperate_char='\n',
                                               text2id=sentence2id, read_data=read_data,
                                               max_read_sample=self.train_max_samples)
        network = sym_gen(s_vocab_size=vocab_size, s_layer_num=self.s_layer_num,
                          s_hidden_unit=self.s_hidden_unit, s_embed_size=self.s_embed_size, s_dropout=self.s_dropout,
                          t_vocab_size=vocab_size, t_layer_num=self.t_layer_num,
                          t_hidden_unit=self.t_hidden_unit, t_embed_size=self.t_embed_size, t_dropout=self.t_dropout,
                          t_label_num=vocab_size,
                          data_names=data_loader.get_data_names(), label_names=data_loader.get_label_names(),
                          batch_size=self.batch_size)

        self._fit(network, data_loader)

    def _fit(self, network, data_loader):
        """
        train a model
        network : the symbol definition of the nerual network
        data_loader : function that returns the train and val data iterators
        """

        # load model
        sym, arg_params, aux_params = self._load_model(self.kv.rank)
        if sym is not None:
            assert sym.tojson() == network.tojson()

        # save model
        checkpoint = self._save_model(self.kv.rank, self.save_checkpoint_freq)

        # devices for training
        if self.device_mode is None or self.device_mode == 'cpu':
            devs = mx.cpu() if self.devices is None or self.devices is '' else [
                mx.cpu(int(i)) for i in self.devices.split(',')]
        else:
            devs = mx.gpu() if self.devices is None or self.devices is '' else [
                mx.gpu(int(i)) for i in self.devices.split(',')]

        # devs = [mx.cpu(0), mx.cpu(1), mx.cpu(2)]
        # learning rate
        lr, lr_scheduler = self._get_lr_scheduler(self.kv)

        # create model
        model = mx.mod.BucketingModule(network, default_bucket_key=data_loader.default_bucket_key, context=devs)
        optimizer_params = {
            'learning_rate': lr,
            'momentum': self.momentum,
            'wd': self.wd,
            'lr_scheduler': lr_scheduler}

        monitor = mx.mon.Monitor(self.monitor_interval, pattern=".*") if self.monitor_interval > 0 else None

        initializer = mx.init.Xavier(
            rnd_type='gaussian', factor_type="in", magnitude=2)
        # initializer   = mx.init.Xavier(factor_type="in", magnitude=2.34),

        # evaluation metrices
        # eval_metrics = ['accuracy']
        eval_metrics = []
        # if args.top_k > 0:
        #   eval_metrics.append(metric.TopKAccuracy('top_k_accuracy', top_k=args.top_k))

        # callbacks that run after each batch
        batch_end_callbacks = [mx.callback.Speedometer(self.batch_size, self.show_every_x_batch)]
        # run
        model.fit(data_loader,
                  begin_epoch=self.load_epoch if self.load_epoch else 0,
                  num_epoch=self.num_epoch,
                  eval_data=data_loader if self.enable_evaluation else None,
                  eval_metric=mx.metric.np(Perplexity),
                  kvstore=self.kv,
                  optimizer=self.optimizer,
                  optimizer_params=optimizer_params,
                  initializer=initializer,
                  arg_params=arg_params,
                  aux_params=aux_params,
                  batch_end_callback=batch_end_callbacks,
                  epoch_end_callback=checkpoint,
                  allow_missing=True,
                  monitor=monitor
                  )
