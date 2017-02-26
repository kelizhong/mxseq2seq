import mxnet as mx
from network.rnn.LSTM import lstm, LSTMModel, LSTMParam, LSTMState
from network.rnn.GRU import gru, GRUModel, GRUParam, GRUState

class LstmEncoder(object):
    def __init__(self, seq_len,
                 num_hidden,
                 input_dim, output_dim,
                 vocab_size, embedding_size,
                 dropout=0.0, num_layers=1):
        self.seq_len = seq_len
        self.num_hidden = num_hidden
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.dropout = dropout
        self.num_layers = num_layers

    def encode(self):
        data = mx.sym.Variable('enc_data')
        embed_weight = mx.sym.Variable("source_embed_weight")
        # embedding layer
        embed = mx.sym.Embedding(data=data, input_dim=self.vocab_size + 1,
                                 weight=embed_weight, output_dim=self.embedding_size, name='source_embed')

        stack = mx.rnn.SequentialRNNCell()
        for i in range(self.num_layers):
            stack.add(mx.rnn.LSTMCell(num_hidden=self.num_hidden, prefix='lstm_l%d_' % i))
        outputs, states = stack.unroll(self.seq_len, inputs=embed, merge_outputs=True)

        return states[-1][0]

class BiDirectionalLstmEncoder(object):
    def __init__(self, seq_len, use_masking,
                 state_dim,
                 input_dim, output_dim,
                 vocab_size, embed_dim,
                 dropout=0.0, num_of_layer=1, embed_weight=None):
        self.seq_len = seq_len
        self.use_masking = use_masking
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.num_of_layer = num_of_layer
        self.embed_weight = embed_weight

    def encode(self):
        data = mx.sym.Variable('source')  # input data, source
        # declare variables
        if not self.embed_weight:
            self.embed_weight = mx.sym.Variable("source_embed_weight")
        forward_param_cells = []
        forward_last_states = []
        for i in range(self.num_of_layer):
            forward_param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("forward_source_l%d_i2h_weight" % i),
                                                 i2h_bias=mx.sym.Variable("forward_source_l%d_i2h_bias" % i),
                                                 h2h_weight=mx.sym.Variable("forward_source_l%d_h2h_weight" % i),
                                                 h2h_bias=mx.sym.Variable("forward_source_l%d_h2h_bias" % i)))
            forward_state = LSTMState(c=mx.sym.Variable("forward_source_l%d_init_c" % i),
                                      h=mx.sym.Variable("forward_source_l%d_init_h" % i))
            forward_last_states.append(forward_state)
        assert (len(forward_last_states) == self.num_of_layer)
        backward_param_cells = []
        backward_last_states = []
        for i in range(self.num_of_layer):
            backward_param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("backward_source_l%d_i2h_weight" % i),
                                                  i2h_bias=mx.sym.Variable("backward_source_l%d_i2h_bias" % i),
                                                  h2h_weight=mx.sym.Variable("backward_source_l%d_h2h_weight" % i),
                                                  h2h_bias=mx.sym.Variable("backward_source_l%d_h2h_bias" % i)))
            backward_state = LSTMState(c=mx.sym.Variable("backward_source_l%d_init_c" % i),
                                       h=mx.sym.Variable("backward_source_l%d_init_h" % i))
            backward_last_states.append(backward_state)
        assert (len(backward_last_states) == self.num_of_layer)

        # embedding layer
        embed = mx.sym.Embedding(data=data, input_dim=self.vocab_size + 1,
                                 weight=self.embed_weight, output_dim=self.embed_dim, name='embed')
        wordvec = mx.sym.SliceChannel(data=embed, num_outputs=self.seq_len, squeeze_axis=1)

        # split mask
        if self.use_masking:
            input_mask = mx.sym.Variable('source_mask')
            masks = mx.sym.SliceChannel(data=input_mask, num_outputs=self.seq_len, name='sliced_source_mask')

        forward_hidden_all = []
        backward_hidden_all = []
        for seq_idx in range(self.seq_len):
            forward_hidden = wordvec[seq_idx]
            backward_hidden = wordvec[self.seq_len - 1 - seq_idx]
            if self.use_masking:
                forward_mask = masks[seq_idx]
                backward_mask = masks[self.seq_len - 1 - seq_idx]

            # stack LSTM
            for i in range(self.num_of_layer):
                if i == 0:
                    dp_ratio = 0.
                else:
                    dp_ratio = self.dropout
                forward_next_state = lstm(self.state_dim, indata=forward_hidden,
                                          prev_state=forward_last_states[i],
                                          param=forward_param_cells[i],
                                          seqidx=seq_idx, layeridx=i, dropout=dp_ratio)
                backward_next_state = lstm(self.state_dim, indata=backward_hidden,
                                           prev_state=backward_last_states[i],
                                           param=backward_param_cells[i],
                                           seqidx=seq_idx, layeridx=i, dropout=dp_ratio)

                # process masking
                if self.use_masking:
                    forward_prev_state_h = forward_last_states[i].h
                    forward_prev_state_c = forward_last_states[i].c
                    forward_new_h = mx.sym.broadcast_mul(1.0 - forward_mask,
                                                         forward_prev_state_h) + mx.sym.broadcast_mul(
                        forward_mask,
                        forward_next_state.h)
                    forward_new_c = mx.sym.broadcast_mul(1.0 - forward_mask,
                                                         forward_prev_state_c) + mx.sym.broadcast_mul(
                        forward_mask,
                        forward_next_state.c)
                    forward_next_state = LSTMState(c=forward_new_c, h=forward_new_h)

                    backward_prev_state_h = backward_last_states[i].h
                    backward_prev_state_c = backward_last_states[i].c
                    backward_new_h = mx.sym.broadcast_mul(1.0 - backward_mask,
                                                          backward_prev_state_h) + mx.sym.broadcast_mul(
                        backward_mask,
                        backward_next_state.h)
                    backward_new_c = mx.sym.broadcast_mul(1.0 - backward_mask,
                                                          backward_prev_state_c) + mx.sym.broadcast_mul(
                        backward_mask,
                        backward_next_state.c)
                    backward_next_state = LSTMState(c=backward_new_c, h=backward_new_h)

                forward_hidden = forward_next_state.h
                forward_last_states[i] = forward_next_state
                backward_hidden = backward_next_state.h
                backward_last_states[i] = backward_next_state

            if self.dropout > 0.:
                forward_hidden = mx.sym.Dropout(data=forward_hidden, p=self.dropout)
                backward_hidden = mx.sym.Dropout(data=backward_hidden, p=self.dropout)

            # bi_hidden = mx.sym.Concat(forward_hidden, backward_hidden)
            # hidden_all.append(bi_hidden)
            forward_hidden_all.append(forward_hidden)
            backward_hidden_all.insert(0, backward_hidden)

        bi_hidden_all = []
        # for seq_idx in range(self.seq_len):
        for f, b in zip(forward_hidden_all, backward_hidden_all):
            bi = mx.sym.Concat(f, b, dim=1)
            bi_hidden_all.append(bi)

        if self.use_masking:
            return forward_hidden_all, backward_hidden_all, bi_hidden_all, masks
        else:
            return forward_hidden_all, backward_hidden_all, bi_hidden_all
