import dataset as dl
import sys
import os
from collections import Counter
import nltk
import pickle
import re

try:
    from tqdm import tqdm
except ImportError as e:
    print('WARNING: tqdm module not found. Install it if you want a fancy progress bar :-)')

    def tqdm(x, disable=False): return x


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    special_tokens = ['<pad>', '<start>', '<end>', '<unk>']

    def __init__(self, start_token=True):
        """Initialize vocabulary object
        :param start_token - determine whether the vocabulary should contain <start> token
        """
        self.word2idx = {}
        self.idx2word = {}
        self.freq = {}
        self.freq_sum = -1
        self.idx = 0

        self.metadata = {'includes_start_token': start_token}

        # Do not add <start> token:
        if not start_token:
            self.special_tokens.remove('<start>')

    def add_word(self, word, frequency=None):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            if frequency is not None:
                self.freq[self.idx] = frequency

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

    def __contains__(self, item):
        if isinstance(item, str):
            return item in self.word2idx
        elif isinstance(item, int):
            return item in self.idx2word
        else:
            raise ValueError

    def __len__(self):
        return len(self.word2idx)

    def frequency(self, word):
        assert hasattr(self, 'metadata') and self.metadata['save_frequencies'], \
            'No frequencies were saved for the vocabulary, please rebuild it again with frequencies enabled.'

        if isinstance(word, str):
            return self.freq[self.word2idx[word]]
        elif isinstance(word, int):
            return self.freq[word]
        elif isinstance(word, list):
            return [self.freq[x] for x in word]
        else:
            raise ValueError('Unrecognised parameter type')

    def frequency_sum(self):
        assert hasattr(self, 'metadata') and self.metadata['save_frequencies'], \
            'No frequencies were saved for the vocabulary, please rebuild it again with frequencies enabled.'

        return self.freq_sum

    def add_special_tokens(self):
        for t in self.special_tokens:
            self.add_word(t, 0)

    def save(self, vocab_path):
        vocab_dir = os.path.dirname(vocab_path)
        if vocab_dir is not None and vocab_dir is not '':
            os.makedirs(vocab_dir, exist_ok=True)

        if vocab_path.endswith('.txt'):
            ll = self.get_list()
            with open(vocab_path, 'w') as f:
                for l in ll:
                    f.write(l + '\n')
        else:
            with open(vocab_path, 'wb') as f:
                pickle.dump(self, f)

    def get_list(self):
        return [self.idx2word[i] for i in range(self.__len__())]

    def update_metadata(self, updated_metadata):
        """Store vocabulary metadata
        :param updated_metadata dict vocabulary metadata info"""

        # Merge existing metadata with supplied:
        self.metadata = {**self.metadata, **updated_metadata}

    def __str__(self):
        return 'Vocabulary with {} words.'.format(len(self))


def get_vocab(ext_args, dataset_params=None):
    """Load vocabulary based on vocab_directive:
    :param ext_args ArgumentParser arguments coming from the caller script
    :param ext_args.vocab directive has following values
        AUTO - load vocabulary from the model file if we are continuing
               training the existing model, and if the loaded model contains
               vocabulary, otherwise fetch vocabulary Pickle from
               `vocabs/train1+train2+trainN.pkl` if it exists,
               or create and load it if it doesn't.
               Train1, train2, trainN are the training datasets used.
        REGEN - create a new vocabulary file from the training datasets used
                for training, even if the cached vocab file exists
        file.{pkl,txt} - force load vocabulary from a specified path
    :param dataset_params
    """
    # Check if we are dealing with a directive:
    if ext_args.vocab.isalpha() and ext_args.vocab.isupper():
        if ext_args.vocab == 'AUTO' or ext_args.vocab == 'REGEN':
            vocab_path = '{}/vocab-{}.pkl'.format(ext_args.vocab_root, ext_args.dataset)
            if ext_args.vocab == 'AUTO' and os.path.isfile(vocab_path):
                return get_vocab_from_pickle(vocab_path)
            else:
                if dataset_params is None:
                    print("Dataset parameters need to be specified when"
                          " building a vocabulary")
                    sys.exit(1)

                return build_vocab(vocab_path, dataset_params, ext_args)
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
        print("Loading existing vocabulary pickle: {}".format(vocab_path))
        vocab = pickle.load(f)

    return vocab


def get_vocab_from_txt(vocab_path):
    lst = []
    with open(vocab_path) as f:
        print("Extracting vocabulary from {} text file".format(vocab_path))
        for a in f:
            b = a.split()
            lst.extend(b)

    return get_vocab_from_list(lst, True)


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
    :param ext_args external ArgumentParser arguments supplied by calling script
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
    print('Building vocabulary with threshold {} (inclusive) ...'.
          format(ext_args.vocab_threshold))
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

    # If the word frequency is less than 'threshold', then the word is discarded
    words, leftovers = [], []
    for word, cnt in counter.items():
        if cnt >= ext_args.vocab_threshold:
            words.append((word, cnt))
        elif ext_args.create_leftovers_file:
            leftovers.append((word, cnt))

    if ext_args.verbose:
        print(words)

    # Create a vocab wrapper and add some special tokens
    vocab = Vocabulary()
    vocab.add_special_tokens()

    # Add the words to the vocabulary
    for word, cnt in words:
        vocab.add_word(word, cnt if ext_args.keep_frequency else None)

    if ext_args.keep_frequency:
        vocab.freq_sum = sum(vocab.freq.values())

    # Store vocabulary metadata:
    metadata = {
        'file_path': os.path.abspath(vocab_output_path),
        'dataset': ext_args.dataset,
        'vocab_threshold': ext_args.vocab_threshold,
        'no_tokenize': ext_args.no_tokenize,
        'keep_frequency': ext_args.keep_frequency
    }

    vocab.update_metadata(metadata)

    # Save it
    if not ext_args.dont_create_vocab:
        vocab.save(vocab_output_path)

        print("Total vocabulary size: {}".format(len(vocab)))
        print("Saved the vocabulary to '{}'".format(vocab_output_path))

    if ext_args.create_leftovers_file:
        dirn = os.path.dirname(vocab_output_path)
        namef = os.path.basename(vocab_output_path).split('.')[0]
        leftovers_name = (dirn + '/') if dirn else '' + namef + '-leftovers.txt'
        with open(leftovers_name, 'w') as f:
            for word, count in sorted(leftovers, key=lambda x: (-x[1], x[0])):
                f.write('{} {}\n'.format(word, count))

        print("Leftover words saved to '{}'".format(leftovers_name))

    if ext_args.create_vocab_stats_file:
        dirn = os.path.dirname(vocab_output_path)
        namef = os.path.basename(vocab_output_path).split('.')[0]
        stats_name = (dirn + '/') if dirn else '' + namef + '-vocab_stats.txt'
        with open(stats_name, 'w') as f:
            for word, count in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
                f.write('{} {}\n'.format(word, count))

        print("Leftover words saved to '{}'".format(stats_name))

    return vocab


# sentence functions now

def fix_caption(caption, skip_start_token=False, keep_tokens=False, capitalize=True):
    if False:
        print('fix_caption() skip_start_token={} keep_tokens={} capitalize={}'.
              format(skip_start_token, keep_tokens, capitalize))
        print('fix_caption() input   [{}]'.format(caption))
    if keep_tokens:
        if skip_start_token:
            m = re.match(r'(.*?)( <end>)', caption)
        else:
            m = re.match(r'(<start> )(.*?)( <end>)', caption)
        if m is None:
            print('ERROR: fix_caption(A) unexpected caption format: "{}"'.format(caption))
            return caption.capitalize() if capitalize else caption

        ret = ''.join(m.groups())
    else:
        if skip_start_token:
            m = re.match(r'^(.*?)( <end>)?$', caption)
        else:
            m = re.match(r'^<start> (.*?)( <end>)?$', caption)
        if m is None:
            print('ERROR: fix_caption(B) unexpected caption format: "{}"'.format(caption))
            return caption.capitalize() if capitalize else caption

        ret = m.group(1)

    ret = re.sub(r'\s([.,])(\s|$)', r'\1\2', ret)
    if capitalize:
        ret = ret.capitalize()
    if False:
        print('fix_caption() returns [{}]'.format(ret))
    return ret


def caption_ids_to_words(sampled_ids, vocab, skip_start_token=False,
                         keep_tokens=False, capitalize=True):
    """
    Converts output tensor of ids to sentences.
    :param sampled_ids: tensor of ids
    :param vocab: vocabulary object
    :param keep_tokens: Will keep <start> and <end> if True.
    :return: Resulting sentence.
    """
    sampled_caption = []
    for word_id in sampled_ids.cpu().numpy():
        word = vocab.idx2word[word_id]
        if word == '<end>':
            if keep_tokens:
                sampled_caption.append(word)
            break
        sampled_caption.append(word)
        
    return fix_caption(' '.join(sampled_caption),
                       skip_start_token=skip_start_token,
                       keep_tokens=keep_tokens, capitalize=capitalize)


def caption_ids_ext_to_words(sampled_ids, vocab, skip_start_token=False,
                             keep_tokens=False, capitalize=True):
    """
    Converts output struct of ids and probs with alternatives to sentences.
    :param sampled_ids: struct of ids
    :param vocab: vocabulary object
    :param keep_tokens: Will keep <start> and <end> if True.
    :return: Resulting sentence.
    """
    if False:
        print(('caption_ids_ext_to_words() skip_start_token={} keep_tokens={}'
               ' capitalize={}').format(skip_start_token, keep_tokens,
                                        capitalize))
    sampled_caption = []
    end_found = False
    for word_id in sampled_ids:
        wl = []
        for word_id_alt in word_id:
            word = vocab.idx2word[word_id_alt[0]]
            if word=='<end>' and len(wl)==0 and not keep_tokens:
                end_found = True
                break
            if len(word_id_alt)==2:
                word += '='+str(word_id_alt[1])
            wl.append(word)
        if len(wl):
            sampled_caption.append('/'.join(wl))
        if (end_found):
            break
            
    return fix_caption(' '.join(sampled_caption), skip_start_token=skip_start_token,
                       keep_tokens=keep_tokens, capitalize=capitalize)


def paragraph_ids_to_words(sampled_ids, vocab, skip_start_token=False, keep_tokens=False):
    paragraph = ''
    for sentence in sampled_ids:
        if sentence[0] == vocab("<pad>"):
            break
        paragraph += caption_ids_to_words(sentence, vocab,
                                          skip_start_token=skip_start_token,
                                          keep_tokens=keep_tokens) + '. '

    paragraph = paragraph.replace(" .", ".")

    return paragraph


def remove_duplicate_sentences(caption):
    """Removes consecutively repeating sentences from the caption"""
    sentences = caption.split('.')

    no_dupes = [sentences[0].strip()]

    for i, _ in enumerate(sentences):
        if sentences[i].strip() != no_dupes[-1].strip():
            no_dupes.append(sentences[i].strip())

    return '. '.join(no_dupes)


def remove_incomplete_sentences(caption):
    """Removes sentences that don't end with a period (truncated or incomplete)"""
    sentences = caption.split('.')
    if sentences[-1] != '':
        sentences[-1] = ''
        return '.'.join(sentences)
    else:
        return caption


def word_ids_to_words(sample, vocab, is_hierarchical=False, skip_start_token=False, keep_tokens=False):
    """
    Converts a tensor matrix of ids (model outputs) into a list of sentences.
    :param sample: Tensor matrix with rows of ids to be converted to sentences.
    :param vocab: vocabulary object
    :param is_hierarchical: if the output comes from a hierarchical model
    :param keep_tokens: Will keep <start> and <end> if True.
    :return: Dictionary with sentences addressed by the position in which they were placed in the tensor, by shape[0].
    """
    ids_to_words_fn = paragraph_ids_to_words if is_hierarchical else caption_ids_to_words
    return {i: [ids_to_words_fn(sample[i], vocab, skip_start_token=skip_start_token, keep_tokens=keep_tokens).lower()]
            for i in range(sample.shape[0])}


def clean_word_ids(sample, vocab):
    end_idx = [(sample[i] == vocab('<end>')).nonzero() for i in range(sample.size(0))]  # because no dim=0 parameter
    end_idx = [i[0].item() if i.size(0) != 0 else None for i in end_idx]
    return [sample[i][:ei + 1 if ei is not None else ei] for i, ei in enumerate(end_idx)]  # end_idx included
