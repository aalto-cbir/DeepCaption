import pickle


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
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def add_special_tokens(self):
        self.add_word('<pad>')
        self.add_word('<start>')
        self.add_word('<end>')
        self.add_word('<unk>')

    def save(self, vocab_path):
        with open(vocab_path, 'wb') as f:
            pickle.dump(self, f)

    def get_list(self):
        return [ self.idx2word[i] for i in range(self.__len__()) ]
    
            
def get_vocab(vocab_path):
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

