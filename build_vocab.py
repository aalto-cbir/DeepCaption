import nltk
import pickle
import argparse
import sys
import json
from collections import Counter


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


def build_vocab(dataset, json_file, threshold):
    """Build a simple vocabulary wrapper."""

    counter = Counter()

    if dataset == 'coco':
        from pycocotools.coco import COCO
        coco = COCO(json_file)
        ids = coco.anns.keys()

        for i, id in enumerate(ids):
            caption = str(coco.anns[id]['caption'])
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)

            if (i + 1) % 1000 == 0:
                print("[{}/{}] Tokenized the captions.".format(i + 1, len(ids)))
    elif dataset == 'vist':
        with open(json_file) as raw_data:
            json_data = json.load(raw_data)
            annotations = json_data['annotations']

        for i, annotation in enumerate(annotations):
            caption = str(annotation[0]['text'])
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)

            if (i + 1) % 1000 == 0:
                print("[{}/{}] Tokenized the captions.".format(i + 1, len(annotations)))
    elif dataset == 'msrvtt':
        with open(json_file, 'r') as fp:
            j = json.load(fp)
            sentences = j['sentences']
            for i, s in enumerate(sentences):
                caption = str(s['caption'])
                tokens = nltk.tokenize.word_tokenize(caption.lower())
                counter.update(tokens)
                if (i + 1) % 1000 == 0:
                    print("[{}/{}] Tokenized the captions.".format(i + 1, len(sentences)))
    else:
        print("Invalid dataset specified...")
        sys.exit(1)

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


def main(args):
    vocab = build_vocab(dataset=args.dataset,
                        json_file=args.caption_path,
                        threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco',
                        help='which dataset to use')
    parser.add_argument('--caption_path', type=str,
                        default='datasets/data/COCO/annotations/captions_train2014.json',
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str,
                        default='datasets/processed/COCO/vocab.pkl',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4,
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)
