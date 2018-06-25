import pickle
import argparse


def main(args):
    if args.dataset == 'coco':
        from datasets import coco_vocab
        vocab = coco_vocab.build_vocab(
            json=args.caption_path,
            threshold=args.threshold)
        vocab_path = args.vocab_path
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab, f)
        print("Total vocabulary size: {}".format(len(vocab)))
        print("Saved the vocabulary wrapper to '{}'".format(vocab_path))
    else:
        print("Invalid dataset. Exiting")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco',
                        help='which dataset to use')
    parser.add_argument('--caption_path', type=str,
                        default='data/annotations/captions_train2014.json',
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4,
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)
