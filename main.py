# -*- coding: utf-8 -*-
"""
Sequence 2 sequence for Query2Vec

"""

import logging
import os
import time
from conf.customArgType import IntegerType, LoggerLevelType, DirectoryType, FileType
from conf.customArgAction import AppendTupleWIthoutDefault
import argparse
from common.constant import special_words


def parse_args():
    parser = argparse.ArgumentParser(description='Train Seq2seq model for query2vec')

    parser.add_argument('-lp', '--log-path', default=os.path.join(os.getcwd(), 'data', 'logs'),
                              type=DirectoryType, help='Log directory (default: __DEFAULT__).')
    parser.add_argument('-ll', '--log-level', choices=['debug', 'info', 'warn', 'error'], default='info',
                              type=LoggerLevelType,
                              help='Log level on console (default: __DEFAULT__).')
    subparsers = parser.add_subparsers(help='train vocabulary')

    train_parser = subparsers.add_parser("train", help='train model', add_help=False)
    train_parser.set_defaults(action='train')
    vocab_parser = subparsers.add_parser("vocab")
    vocab_parser.set_defaults(action='vocab')

    # model parameter
    train_parser.add_argument('-sny', '--source-num-layers', default=3, type=int,
                              help='number of layers for the source LSTM recurrent neural network')
    train_parser.add_argument('-snh', '--source-num-hidden', default=512, type=int,
                              help='number of hidden units in the source neural network')
    train_parser.add_argument('-es', '--embed-size', default=150, type=int,
                              help='embedding size ')

    train_parser.add_argument('-tny', '--target-num-layers', default=3, type=int,
                              help='number of layers for the target LSTM recurrent neural network')
    train_parser.add_argument('-tnh', '--target-num-hidden', default=512, type=int,
                              help='number of hidden units in the target neural network')

    train_parser.add_argument('-b', '--buckets', nargs=2, action=AppendTupleWIthoutDefault, type=int,
                              default=[(3, 10), (3, 20), (5, 20), (7, 30)])

    # train parameter

    train_parser.add_argument('-do', '--dropout', default=0.0, type=float,
                              help='dropout is the probability to ignore the neuron outputs')
    train_parser.add_argument('-le', '--load-epoch', dest='load_epoch', help='epoch of pretrained model',
                               type=int)
    train_parser.add_argument('-mp', '--model-prefix', default='query2vec',
                              type=str,
                              help='the experiment name, this is also the prefix for the parameters file')
    train_parser.add_argument('-pd', '--model-path', default=os.path.join(os.getcwd(), 'data', 'model'), type=DirectoryType,
                              help='the directory to store the parameters of the training')
    train_parser.add_argument('-lr', '--learning-rate', dest='lr', default=0.01, type=float,
                              help='learning rate of the stochastic gradient descent')
    train_parser.add_argument('-lrf', '--lr-factor', default=1, type=float,
                              help='the ratio to reduce lr on each step')
    train_parser.add_argument('-wd', '--weight-decay', type=float, default=0.0005, help='weight decay for sgd')
    train_parser.add_argument('-opt', '--optimizer', type=str, default='sgd',
                              help='the optimizer type')
    train_parser.add_argument('-tms', '--train-max-samples', default=20000, type=int,
                              help='the max sample num for training')
    train_parser.add_argument('-mom', '--momentum', type=float, default=0.9, help='momentum for sgd')
    train_parser.add_argument('-sexb', '--show-every-x-batch', dest='show_every_x_batch',
                              help='show progress for every x batches',
                              default=1, type=int)
    train_parser.add_argument('-ne', '--num-epoch', dest='num_epoch', help='end epoch of query2vec training',
                              default=100000, type=int)

    train_parser.add_argument('-bs', '--batch-size', default=128, type=int,
                              help='batch size for each databatch')

    # mxnet parameter
    train_parser.add_argument('-dm', '--device-mode', choices=['cpu', 'gpu'],
                              help='define define mode, (default: %(default)s)',
                              default='cpu')
    train_parser.add_argument('-d', '--devices', type=str, default='0',
                              help='the devices will be used, e.g "0,1,2,3"')

    train_parser.add_argument('-lf', '--log-freq', default=1000, type=int,
                              help='the frequency to printout the training verbose information')

    train_parser.add_argument('-scf', '--save-checkpoint-freq', default=100, type=int,
                              help='the frequency to save checkpoint')

    train_parser.add_argument('-kv', '--kv-store', dest='kv_store', help='the kv-store type',
                              default='device', type=str)
    train_parser.add_argument('-mi', '--monitor-interval', default=0, type=int,
                              help='log network parameters every N iters if larger than 0')
    train_parser.add_argument('-eval', '--enable-evaluation', action='store_true', help='enable evaluation')

    train_parser.add_argument('--iterations', default=1, type=int,
                              help='number of iterations (epoch)')


    train_parser.add_argument('--gpus', type=str,
                              help='the gpus will be used, e.g "0,1,2,3"')

    train_parser.add_argument('-eti', '--enc-test-input', type=str,
                              help='the file name of the encoder input for testing')
    train_parser.add_argument('-dti', '--dec-test-input', type=str,
                              help='the file name of the decoder input for testing')



    train_parser.add_argument('-wll', '--work-load_-ist', dest='work_load_list', help='work load for different devices',
                              default=None, type=list)


    train_parser.add_argument('--factor-step', type=int, default=50000, help='the step used for lr factor')
    train_parser.add_argument('--monitor', action='store_true', default=False,
                              help='if true, then will use monitor debug')

    #data parameter
    train_parser.add_argument('train_source_path', type=str,
                              help='the file name of the encoder input for training')
    train_parser.add_argument('train_target_path', type=str,
                              help='the file name of the decoder input for training')
    train_parser.add_argument('vocabulary_path', default=os.path.join(os.getcwd(), 'data', 'vocabulary', 'vocab.pkl'),
                              type=str,
                              help='vocabulary with he most common words')

    #vocabulary parameter
    vocab_parser.add_argument('-tw', '--top-words', default=40000, type=int,
                              help='the words with the top frequency to retain in the vocabulary')

    vocab_parser.add_argument('--overwrite', action='store_true', help='overwrite earlier created files, also forces the\
                        program not to reuse count files')
    vocab_parser.add_argument('files', nargs='+',
                              help='the corpus input files')
    vocab_parser.add_argument('vocab_file',
                              type=FileType,
                              help='the file with the words which are the most command words in the corpus')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    print(args)
    if args.action == 'train':
        from seq2seq.trainer import trainer

        trainer(train_source_path=args.train_source_path, train_target_path=args.train_target_path,
                vocabulary_path=args.vocabulary_path) \
            .set_model_parameter(s_layer_num=args.source_num_layers, s_hidden_unit=args.source_num_hidden,
                                 s_embed_size=args.embed_size, t_layer_num=args.target_num_layers,
                                 t_hidden_unit=args.target_num_hidden, t_embed_size=args.embed_size,
                                 buckets=args.buckets) \
            .set_train_parameter(s_dropout=args.dropout, t_dropout=args.dropout, load_epoch=args.load_epoch,
                                 model_prefix=os.path.join(args.model_path, args.model_prefix), device_mode=args.device_mode, devices=args.devices,
                                 lr_factor=args.lr_factor, wd=args.weight_decay,\
                                 lr=args.lr, train_max_samples=args.train_max_samples, momentum=args.momentum,
                                 show_every_x_batch=args.show_every_x_batch, num_epoch=args.num_epoch,
                                 optimizer=args.optimizer, batch_size=args.batch_size) \
            .set_mxnet_parameter(log_path=args.log_path, log_level=args.log_level, kv_store=args.kv_store, enable_evaluation=args.enable_evaluation,
                                 monitor_interval=args.monitor_interval, save_checkpoint_freq=args.save_checkpoint_freq) \
            .train()
    elif args.action == 'vocab':
        from vocabulary.vocab_gen import vocab

        vocab(args.files, args.vocab_file, top_words=args.top_words, special_words=special_words,
              log_path=args.log_path, log_level=args.log_level, overwrite=args.overwrite) \
            .create_dictionary()
