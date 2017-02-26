import os
import pickle
import re
import sys

from common import constant as config

import codecs

# Read from doc
def read_data(path, max_read_line=sys.maxsize):
    data  = []
    count = 0
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            count += 1
            if count > max_read_line:
                break
            line = line.strip()
            data.append(line.split(' '))
    return data


def load_vocab(path):
    """
    Load vocab from file, the 0, 1, 2, 3 should be reserved for pad, <unk>, <s>, </s>
    :param path: the vocab
    :param special:
    :return:
    """
    with open(path, 'rb') as f:
        vocab = pickle.load(f)

    return vocab


def sentence2id(sentence, the_vocab):
    words = list(sentence)
    words = [the_vocab[w] if w in the_vocab else the_vocab[config.unk_word] for w in words if len(w) > 0]
    return words


def word2id(word, the_vocab):
    return the_vocab[word] if word in the_vocab else the_vocab[config.unk_word]

def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.

  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  if os.path.isfile(vocabulary_path):
    rev_vocab = []
    with open(vocabulary_path, "r+") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)

def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size=40000):
  """
  Create vocabulary file (if it does not exist yet) from data file.
  Data file should have one sentence per line.
  Each sentence will be tokenized.
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
  """
  print (vocabulary_path)
  if not os.path.isfile(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with open(data_path, 'r+') as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 100000 == 0:
          print("  processing line %d" % counter)
        try:
          tokens = tokenize(line)
        except:
          print("Tokenize failure: " + line)
          continue
        for w in tokens:
          if vocab.has_key(w):
            vocab[w] += 1
          else:
            vocab[w] = 1
      vocab_list = ["pad", "<unk>", "<s>", "</s>"] + sorted(vocab, key=vocab.get, reverse=True)
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      with open(vocabulary_path, 'w+') as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + b"\n")
    print('Vocabulary file created')

def tokenize(sentence):
  """Tokenizer: split the sentence into a list of tokens."""
  sentence = clean_html(sentence)
  words = sentence.split()
  return [w for w in words if w.strip()]

def clean_html(html):
  """
  Copied from NLTK package.
  Remove HTML markup from the given string.

  :param html: the HTML string to be cleaned
  :type html: str
  :rtype: str
  """

  # First we remove inline JavaScript/CSS:
  cleaned = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", "", html.strip())
  # Then we remove html comments. This has to be done before removing regular
  # tags since comments can contain '>' characters.
  cleaned = re.sub(r"(?s)<!--(.*?)-->[\n]?", "", cleaned)
  # Next we can remove the remaining tags:
  cleaned = re.sub(r"(?s)<.*?>", " ", cleaned)
  # Finally, we deal with whitespace
  cleaned = re.sub(r"&nbsp;", " ", cleaned)
  cleaned = re.sub(r"  ", " ", cleaned)
  cleaned = re.sub(r"  ", " ", cleaned)
  return cleaned.strip()

def get_enc_dec_text_id(path, vocab):
    enc_data = []
    with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as fid:
        for line in fid:
            enc_list = line.strip().split()
            enc = [vocab.get(word) if vocab.get(word) != None else vocab.get('<UNK>') for word in enc_list]
            enc_data.append(enc)
    return enc_data