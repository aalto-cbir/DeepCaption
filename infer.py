#!/usr/bin/env python3

import argparse
import glob
import json
import os
import re
import sys
import numpy as np

from datetime import datetime
from PIL import Image

import torch
from torchvision import transforms

from vocabulary import Vocabulary, get_vocab  # (Needed to handle Vocabulary pickle)
from data_loader import get_loader, ExternalFeature, DatasetConfig, DatasetParams
from model import ModelParams, EncoderDecoder
from model import SoftAttentionEncoderDecoder, SpatialAttentionEncoderDecoder

try:
    from tqdm import tqdm
except ImportError as e:
    print('WARNING: tqdm module not found. Install it if you want a fancy progress bar :-)')

    def tqdm(x, disable=False): return x

# Device configuration now in infer()
device = None


def basename(fname):
    return os.path.splitext(os.path.basename(fname))[0]


def fix_caption(caption, skip_start_token=False):
    if skip_start_token:
        m = re.match(r'^(.*?)( <end>)?$', caption)
    else:
        m = re.match(r'^<start> (.*?)( <end>)?$', caption)
    if m is None:
        print('ERROR: unexpected caption format: "{}"'.format(caption))
        return caption.capitalize()

    ret = m.group(1)
    ret = re.sub(r'\s([.,])(\s|$)', r'\1\2', ret)
    return ret.capitalize()


def caption_ids_to_words(sampled_ids, vocab):
    sampled_caption = []
    for word_id in sampled_ids.cpu().numpy():
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    return fix_caption(' '.join(sampled_caption))


def paragraph_ids_to_words(sampled_ids, vocab):
    paragraph = ''
    for sentence in sampled_ids:
        if sentence[0] == vocab("<pad>"):
            break
        paragraph += caption_ids_to_words(sentence, vocab) + '. '

    paragraph = paragraph.replace(" .", ".")

    return paragraph


def path_from_id(image_dir, image_id):
    """Return image path based on image directory, image id and
    glob matching for extension"""
    return glob.glob(os.path.join(image_dir, image_id) + '.*')[0]


def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    if image.mode != 'RGB':
        print('WARNING: converting {} from {} to RGB'.
              format(image_path, image.mode))
        image = image.convert('RGB')

    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image


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


def infer(ext_args=None):
    args = parse_args(ext_args)

    global device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # Create model directory
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    scorers = {}
    if args.scoring is not None:
        for s in args.scoring.split(','):
            s = s.lower().strip()
            if s == 'cider':
                from eval.cider import Cider
                scorers['CIDEr'] = Cider(df='corpus')

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((args.resize, args.resize)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Get dataset parameters:
    dataset_configs = DatasetParams(args.dataset_config_file)
    dataset_params = dataset_configs.get_params(args.dataset,
                                                args.image_dir,
                                                args.image_files)

    # Build models
    print('Bulding models.')

    if device.type == 'cpu':
        state = torch.load(args.model, map_location=lambda storage, loc: storage)
    else:
        state = torch.load(args.model)
    params = ModelParams(state)
    if args.ext_features:
        params.update_ext_features(args.ext_features)
    if args.ext_persist_features:
        params.update_ext_persist_features(args.ext_persist_features)

    print("Loaded model parameters:")
    print(params)

    # Load the vocabulary:
    if args.vocab is not None:
        # Loading vocabulary from file path supplied by the user:
        vocab = get_vocab(args)
    elif params.vocab is not None:
        print('Loading vocabulary stored in the model file.')
        vocab = params.vocab
    else:
        print('ERROR: you must either load a model that contains vocabulary or '
              'specify a vocabulary with the --vocab option!')
        sys.exit(1)

    print('Size of the vocabulary is {}'.format(len(vocab)))

    if params.has_external_features() and any(dc.name == 'generic' for dc in dataset_params):
        print('WARNING: you cannot use external features without specifying all datasets in '
              'datasets.conf.')
        print('Hint: take a look at datasets/datasets.conf.default.')

    # Build data loader
    print("Loading dataset: {}".format(args.dataset))

    # Update dataset params with needed model params:
    for i in dataset_params:
        i.config_dict['skip_start_token'] = params.skip_start_token
        # For visualizing attention we need file names instead of IDs in our output:
        if args.store_image_paths:
            i.config_dict['return_image_file_name'] = True

    ext_feature_sets = [params.features.external, params.persist_features.external]

    # We ask it to iterate over images instead of all (image, caption) pairs
    data_loader, ef_dims = get_loader(dataset_params, vocab=None, transform=transform,
                                      batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers,
                                      ext_feature_sets=ext_feature_sets,
                                      skip_images=not params.has_internal_features(),
                                      iter_over_images=True)

    # Build the models
    if params.attention is None:
        _Model = EncoderDecoder
    elif params.attention == 'spatial':
        _Model = SpatialAttentionEncoderDecoder
    elif params.attention == 'soft':
        _Model = SoftAttentionEncoderDecoder
    else:
        print("ERROR: Invalid attention model specified")
        sys.exit(1)

    model = _Model(params, device, len(vocab), state, ef_dims).eval()

    # Store captions here:
    output_data = []

    if params.attention is not None:
        # Store attention weights here:
        # DIM: ( #images, words/sentence, flattened convolutional image grid)
        output_alphas = np.zeros((len(data_loader.dataset),
                                  args.max_seq_length,
                                  model.decoder.num_attention_locs))

    gts = {}
    res = {}

    print('Starting inference, max captions/sentence length: {}'.format(args.max_seq_length))
    show_progress = sys.stderr.isatty() and not args.verbose
    for i, (images, ref_captions, lengths, image_ids,
            features) in enumerate(tqdm(data_loader, disable=not show_progress)):

        if len(scorers) > 0:
            for j in range(len(ref_captions)):
                jid = image_ids[j]
                if jid not in gts:
                    gts[jid] = []
                rcs = ref_captions[j]
                if type(rcs) is str:
                    rcs = [rcs]
                for rc in rcs:
                    gts[jid].append(rc.lower())

        images = images.to(device)

        init_features = features[0].to(device)if len(features) > 0 and \
            features[0] is not None else None
        persist_features = features[1].to(device) if len(features) > 1 and \
            features[1] is not None else None

        # Generate a caption from the image
        sampled_batch = model.sample(images, init_features, persist_features,
                                     max_seq_length=args.max_seq_length,
                                     start_token_id=vocab('<start>'),
                                     end_token_id=vocab('<end>'))

        if params.attention is None:
            sampled_ids_batch = sampled_batch
        else:
            # When sampling from attention models we also get an attention
            # distribution stored in alphas (can be used for visualization):
            sampled_ids_batch, alphas = sampled_batch
            if params.attention is not None:
                alphas_numpy = alphas.detach().cpu().numpy()
                offset_begin = i * args.batch_size
                offset_end = offset_begin + alphas_numpy.shape[0]
                output_alphas[offset_begin: offset_end, :] = alphas_numpy

        for i in range(sampled_ids_batch.shape[0]):
            sampled_ids = sampled_ids_batch[i]

            # Convert word_ids to words
            if params.hierarchical_model:
                caption = paragraph_ids_to_words(sampled_ids, vocab)
            else:
                caption = caption_ids_to_words(sampled_ids, vocab)

            if args.no_repeat_sentences:
                caption = remove_duplicate_sentences(caption)

            if args.only_complete_sentences:
                caption = remove_incomplete_sentences(caption)

            if args.verbose:
                print('=>', caption)

            output_data.append({'caption': caption, 'image_id': image_ids[i]})
            res[image_ids[i]] = [caption.lower()]

    for score_name, scorer in scorers.items():
        score = scorer.compute_score(gts, res)[0]
        print('Test', score_name, score)

    # Decide output format, fall back to txt
    if args.output_format is not None:
        output_format = args.output_format
    elif args.output_file and args.output_file.endswith('.json'):
        output_format = 'json'
    else:
        output_format = 'txt'

    # Create a sensible default output path for results:
    output_file = None
    if not args.output_file and not args.print_results:
        model_name = args.model.split(os.sep)[-2]
        model_epoch = basename(args.model)
        output_file = '{}-{}.{}'.format(model_name, model_epoch, output_format)
    else:
        output_file = args.output_file

    if output_file:
        output_path = os.path.join(args.results_path, output_file)
        if output_format == 'json':
            json.dump(output_data, open(output_path, 'w'))
        else:
            with open(output_path, 'w') as fp:
                for data in output_data:
                    print(data['image_id'], data['caption'], file=fp)

        print('Wrote generated captions to {} as {}'.format(output_path, args.output_format))

        if params.attention is not None:
            # Store attention weights:
            output_path_alphas = os.path.splitext(output_path)[0] + '_alphas.npy'
            np.save(output_path_alphas, output_alphas)
            print("Wrote attention weights to {}".format(output_path_alphas))

    if args.print_results:
        for d in output_data:
            print('{}: {}'.format(d['image_id'], d['caption']))

    return output_data


def parse_args(ext_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='generic',
                        help='which dataset to use')
    parser.add_argument('--dataset_config_file', type=str,
                        default='datasets/datasets.conf',
                        help='location of dataset configuration file')
    parser.add_argument('--resize', type=int, default=224,
                        help='resize input image to this size')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('image_files', type=str, nargs='*')
    parser.add_argument('--image_dir', type=str,
                        help='input image dir for generating captions')
    parser.add_argument('--model', type=str, required=True,
                        help='path to existing model')
    parser.add_argument('--vocab', type=str, help='path for vocabulary wrapper')
    parser.add_argument('--ext_features', type=str,
                        help='paths for the external features, overrides the '
                        'paths in the model ckpt file (which are the ones '
                        'used for training), comma separated')
    parser.add_argument('--ext_persist_features', type=str,
                        help='paths for external persist features')
    parser.add_argument('--store_image_paths', action='store_true',
                        help='Save paths to images in the output file')
    parser.add_argument('--output_file', type=str,
                        help='path for output file, default: model_name.txt')
    parser.add_argument('--output_format', type=str, help='format of the output file')
    parser.add_argument('--verbose', help='verbose output',
                        action='store_true')
    parser.add_argument('--results_path', type=str, default='results/',
                        help='path for saving results')
    parser.add_argument('--print_results', action='store_true')
    parser.add_argument('--scoring', type=str)
    parser.add_argument('--max_seq_length', type=int, default=20,
                        help='maximum allowed length of the decoded sequence')
    parser.add_argument('--no_repeat_sentences', action='store_true',
                        help='allow repeating sentences inside a paragraph')
    parser.add_argument('--only_complete_sentences', action='store_true')
    parser.add_argument('--cpu', action="store_true",
                        help="Use CPU even when GPU is available")

    return parser.parse_args(ext_args)


if __name__ == '__main__':
    begin = datetime.now()
    print('Started inference at {}.'.format(begin))

    infer()

    end = datetime.now()
    print('Inference ended at {}. Total time: {}.'.format(end, end - begin))
