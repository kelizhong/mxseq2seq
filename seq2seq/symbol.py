from seq2seq.encoder import LstmEncoder, BiDirectionalLstmEncoder
from network.seq2seq.decoder import LstmAttentionDecoder, LstmDecoder
import mxnet as mx
def s2s_unroll(s_layer_num, s_seq_len, s_vocab_size, s_hidden_unit, s_embed_size, s_dropout,
               t_layer_num, t_seq_len, t_vocab_size, t_hidden_unit, t_embed_size, t_label_num, t_dropout,
               **kwargs):
    #embedding
    embed_weight = mx.sym.Variable("embed_weight")

    encoder = LstmEncoder(seq_len=s_seq_len, num_hidden=s_hidden_unit,
                                       input_dim=s_vocab_size, output_dim=0,
                                       vocab_size=s_vocab_size, embedding_size=s_embed_size,
                                       dropout=s_dropout, num_layers=s_layer_num)

    #encoder = BiDirectionalLstmEncoder(seq_len=s_seq_len, use_masking=True, state_dim=s_hidden_unit,
    #                                   input_dim=s_vocab_size, output_dim=0,
    #                                   vocab_size=s_vocab_size, embed_dim=s_embed_size,
    #                                   dropout=s_dropout, num_of_layer=s_layer_num, embed_weight=embed_weight)

    decoder = LstmDecoder(seq_len=t_seq_len, use_masking=True, state_dim=t_hidden_unit,
                          input_dim=t_vocab_size, output_dim=t_label_num,
                          vocab_size=t_vocab_size, embed_dim=t_embed_size, dropout=t_dropout,
                          num_of_layer=t_layer_num, embed_weight=embed_weight, **kwargs)
    encoded_for_init_state= encoder.encode()
    #forward_hidden_all, backward_hidden_all, source_representations, source_mask_sliced = encoder.encode()

    #encoded_for_init_state = mx.sym.Concat(forward_hidden_all[-1], backward_hidden_all[0], dim=1,
    #                                       name='encoded_for_init_state')
    target_representation = decoder.decode(encoded_for_init_state)
    return target_representation

def sym_gen(s_vocab_size=None, s_layer_num=None, s_hidden_unit=None, s_embed_size=None, s_dropout=None,
            t_vocab_size=None, t_layer_num=None, t_hidden_unit=None, t_embed_size=None, t_dropout=None,
            t_label_num=None,
            batch_size=None, data_names=None, label_names=None):
    def _sym_gen(s_t_len):
        sym = s2s_unroll(s_layer_num=s_layer_num, s_seq_len=s_t_len[0],
                         s_vocab_size=s_vocab_size + 1,
                         s_hidden_unit=s_hidden_unit, s_embed_size=s_embed_size, s_dropout=s_dropout,
                         t_layer_num=t_layer_num, t_seq_len=s_t_len[1],
                         t_vocab_size=t_vocab_size + 1,
                         t_hidden_unit=t_hidden_unit, t_embed_size=t_embed_size,
                         t_label_num=t_label_num + 1, t_dropout=t_dropout, batch_size=batch_size)
        print(sym.list_arguments())
        return sym, data_names, label_names

    return _sym_gen