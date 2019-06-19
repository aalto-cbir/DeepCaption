"""
Preprocess a raw json dataset into hdf5/json files for use in data_loader.lua

Input: json file that has the form
[{ file_path: 'path/img.jpg', captions: ['a caption', ...] }, ...]
example element in this list would look like
{'captions': [u'A man with a red helmet on a small moped on a dirt road. ', u'Man riding a motor bike on a dirt road on the countryside.', u'A man riding on the back of a motorcycle.', u'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ', u'A man in a red shirt and a red hat is on a motorcycle on a hill side.'], 'file_path': u'val2014/COCO_val2014_000000391895.jpg', 'id': 391895}

This script reads this json, does some basic preprocessing on the captions
(e.g. lowercase, etc.), creates a special UNK token, and encodes everything to arrays

Output: a json file and an hdf5 file
The hdf5 file contains several fields:
/images is (N,3,256,256) uint8 array of raw image data in RGB format
/labels is (M,max_length) uint32 array of encoded labels, zero padded
/label_start_ix and /label_end_ix are (N,) uint32 arrays of pointers to the
  first and last indices (in range 1..M) of labels for each image
/label_length stores the length of the sequence for each of the M sequences

The json file has a dict that contains:
- an 'ix_to_word' field storing the vocab in form {ix:'word'}, where ix is 1-indexed
- an 'images' field that is a list holding auxiliary information for each image,
  such as in particular the 'split' it was assigned to.
"""

import argparse
from collections import defaultdict
from tqdm import tqdm
import pickle

from build_vocab import check_dataset
import dataset as dl
from utils import get_ground_truth_captions
from vocabulary import get_vocab


def precook(s, n=4, out=False):
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    words = s.split()
    counts = defaultdict(int)
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i:i + k])
            counts[ngram] += 1
    return counts


def cook_refs(refs, n=4):  # lhuang: oracle will call with "average"
    """
    Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    """
    return [precook(ref, n) for ref in refs]


def create_crefs(refs):
    crefs = []
    for ref in tqdm(refs, total=len(refs), desc='2/3'):
        # ref is a list of 5 captions
        crefs.append(cook_refs(ref))
    return crefs


def compute_doc_freq(crefs):
    """
    Compute term frequency for reference data.
    This will be used to compute idf (inverse document frequency later)
    The term frequency is stored in the object
    :return: None
    """
    document_frequency = defaultdict(float)
    for refs in tqdm(crefs, total=len(crefs), desc='3/3'):
        # refs, k ref captions of one image
        for ngram in set([ngram for ref in refs for ngram in ref.keys()]):
            document_frequency[ngram] += 1
            # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)
    return document_frequency


def build_dict(gts, vocab):
    count_imgs = 0
    refs_words = []
    for id, gt in tqdm(gts.items(), total=len(gts), desc='1/3'):
        ref_words = []
        for caption in gt:
            tmp_words = [w if w in vocab else '<unk>' for w in caption.split()]
            ref_words.append(' '.join(tmp_words))
            # ref_idxs.append(' '.join([str(word2tok[_]) for _ in tmp_tokens]))

        refs_words.append(ref_words)
        count_imgs += 1

    ngram_words = compute_doc_freq(create_crefs(refs_words))

    return ngram_words, count_imgs


def get_data_loader(args):
    check_dataset(args)
    dataset_configs = dl.DatasetParams(args.dataset_config_file)
    dataset_params = dataset_configs.get_params(args.dataset)

    # Get data loader
    data_loader, _ = dl.get_loader(dataset_params, None, None, 128, shuffle=False,
                                   num_workers=args.num_workers, skip_images=True,
                                   unique_ids=True)

    assert len(data_loader) != 0, 'ERROR: No captions found, please specify a dataset that has captions defined.'
    assert args.dataset != 'coco', 'HINT: instead of "coco" use "coco:train2014"'

    return data_loader


def main(args):
    data_loader = get_data_loader(args)
    gts = get_ground_truth_captions(data_loader.dataset)
    vocab = get_vocab(args)

    ngram_words, ref_len = build_dict(gts, vocab)

    with open(args.output, 'wb') as f:
        pickle.dump({'document_frequency': ngram_words, 'ref_len': ref_len}, f, pickle.HIGHEST_PROTOCOL)

    print('Output file saved in', args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, help='which dataset to use')
    parser.add_argument('--dataset_config_file', type=str,
                        default='datasets/datasets.conf',
                        help='location of dataset configuration file')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--vocab', type=str, default=None,
                        help='Vocabulary directive or path. '
                             'Directives are all-caps, no special characters. '
                             'Vocabulary file formats supported - *.{pkl,txt}.\n'
                             'AUTO: If vocabulary corresponding to current training set '
                             'combination exits in the vocab/ folder load it. '
                             'If not, generate a new vocabulary file\n'
                             'REGEN: Regenerate a new vocabulary file and place it in '
                             'vocab/ folder\n'
                             'path/to/vocab.\{pkl,txt\}: path to Pickled or plain-text '
                             'vocabulary file\n')
    parser.add_argument('--output', default='ngrams_output.pkl', help='output pickle file')
    args = parser.parse_args()

    main(args)
