from seq2seq_bucket_io import EncoderDecoderIter, DummyIter
from common.constant import special_words
from utils.data_utils import load_vocab, get_enc_dec_text_id
from vocab import vocab

data_files = ['../data/ptb.train.txt']
vocab_file = '../data/vocabulary/seq2seq.pkl'
vocab(data_files, vocab_file, top_words=40000, special_words=special_words, log_path='../data/logs').create_dictionary()
# load vocabulary
vocab = load_vocab(vocab_file)

vocab_size = len(vocab)
print('vocab size: {0}'.format(vocab_size))

enc = get_enc_dec_text_id('../data/ptb.train.txt', vocab)

enc_dec = EncoderDecoderIter(enc, enc, special_words.get('pad_word'), special_words.get('eos_word'))

for x in enc_dec:
    print(x.data_names)
    print(x.bucket_key)