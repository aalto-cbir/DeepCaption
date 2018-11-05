import pickle
import re
import data_loader as dl
import sys
import os
from collections import Counter
import nltk
import pickle

try:
    from tqdm import tqdm
except ImportError as e:
    print('WARNING: tqdm module not found. Install it if you want a fancy progress bar :-)')

    def tqdm(x, disable=False): return x


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if isinstance(word, str):
            if word not in self.word2idx:
                return self.word2idx['<unk>']
            return self.word2idx[word]
        else:
            if word not in self.idx2word:
                return -1
            return self.idx2word[word]

    def __len__(self):
        return len(self.word2idx)

    def add_special_tokens(self):
        self.add_word('<pad>')
        self.add_word('<start>')
        self.add_word('<end>')
        self.add_word('<unk>')

    def save(self, vocab_path):
        is_txt = re.search('\\.txt$', vocab_path)
        if is_txt:
            ll = self.get_list()
            with open(vocab_path, 'w') as f:
                for l in ll:
                    f.write(l + '\n')
        else:
            with open(vocab_path, 'wb') as f:
                pickle.dump(self, f)

    def get_list(self):
        return [self.idx2word[i] for i in range(self.__len__())]


def get_vocab(ext_args, dataset_params=None):
    """Load vocabulary based on vocab_directive:
    :param ext_args ArgumentParser arguments coming from the caller script
    :param ext_args.vocab directive has following values
        AUTO - load vocabulary from the model file if we are continuing
               training the existing model, and if the loaded model contains
               vocabulary, otherwise fetch vocabulary Pickle from
               `vocabs/train1+train2+trainN.pkl` if it exists,
               or create and load it if it doesn’t.
               Train1, train2, trainN are the training datasets used.
        REGEN - create a new vocabulary file from the training datasets used
                for training, even if the cached vocab file exists
        file.{pkl,txt} - force load vocabulary from a specified path
    :param dataset_params
    """
    # Check if we are dealing with a directive:
    if ext_args.vocab.isalpha() and ext_args.vocab.isupper():
        if ext_args.vocab == 'AUTO' or ext_args.vocab == 'REGEN':
            vocab_path = '{}/{}.pkl'.format(ext_args.vocab_root, ext_args.datasets)
            if ext_args.vocab == 'AUTO' and os.path.isfile(vocab_path):
                return get_vocab_from_pickle(vocab_path)
            else:
                if dataset_params is None:
                    print("Dataset parameters need to be specified when"
                          " building a vocabulary")
                    sys.exit(1)

                build_vocab(vocab_path, dataset_params, ext_args)
        else:
            print("Invalid vocabulary directive")
            sys.exit(1)
    elif ext_args.vocab.endswith('.pkl'):
        return get_vocab_from_pickle(ext_args.vocab)
    elif ext_args.vocab.endswith('.txt'):
        return get_vocab_from_txt(ext_args.vocab)


def get_vocab_from_pickle(vocab_path):
    # Load vocabulary wrapper
    with open(vocab_path, 'rb') as f:
        print("Extracting vocabulary from {}".format(vocab_path))
        vocab = pickle.load(f)

    return vocab


def get_vocab_from_txt(vocab_path):
    l = []
    with open(vocab_path) as f:
        print("Extracting vocabulary from {} text file".format(vocab_path))
        for a in f:
            b = a.split()
            l.extend(b)

    return get_vocab_from_list(l, True)


def get_vocab_from_list(l, add_specials):
    vocab = Vocabulary()
    if add_specials:
        vocab.add_special_tokens()
    for i in l:
        vocab.add_word(i)

    return vocab


def build_vocab(vocab_output_path, dataset_params, ext_args):
    """Generate vocabulary pickle file
    :param vocab_output_path target path where to save the file
    :param dataset_params dataset configuration parameters supplied to data_loader
    :paramn ext_args exteranl ArgumentParser arguments supplied by calling script
    """

    # Check that we are not overwriting anything
    if os.path.exists(vocab_output_path):
        print('ERROR: {} exists, please remove it first if you really want to replace it.'.
              format(vocab_output_path))
        sys.exit(1)

    # Get data loader
    data_loader, _ = dl.get_loader(dataset_params, None, None, 128, shuffle=False,
                                   num_workers=ext_args.num_workers, skip_images=True)

    if len(data_loader) == 0:
        print('ERROR: No captions found, please specify a dataset that has captions defined.')
    if ext_args.dataset == 'coco':
        print('HINT: instead of "coco" use "coco:train2014"')
    sys.exit(1)

    # Start counting words...
    counter = Counter()
    show_progress = sys.stderr.isatty()
    for _, captions, _, _, _ in tqdm(data_loader, disable=not show_progress):
        for caption in captions:
            if ext_args.no_tokenize:
                words = caption.split()
            else:
                words = nltk.tokenize.word_tokenize(caption.lower())
            if ext_args.show_tokens:
                joined = ' '.join(words)
                diff_same = "DIFF" if caption != joined else "SAME"
                print(diff_same, caption, '=>', joined)
            counter.update(words)

    if ext_args.show_vocab_stats:
        for k in counter.keys():
            print(k, counter[k])

    # If the word frequency is less than 'threshold', then the word is discarded
    words = [word for word, cnt in counter.items() if cnt >= ext_args.vocab_threshold]

    if ext_args.verbose:
        print(words)

    # Create a vocab wrapper and add some special tokens
    vocab = Vocabulary()
    vocab.add_special_tokens()

    # Add the words to the vocabulary
    for word in words:
        vocab.add_word(word)

    # Save it
    vocab.save(vocab_output_path)

    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary to '{}'".format(vocab_output_path))
