#!/usr/bin/env python3

import argparse
import json
import os
import sys

from datetime import datetime

import torch
from torchvision import transforms

from vocabulary import paragraph_ids_to_words, caption_ids_to_words, remove_duplicate_sentences, \
    remove_incomplete_sentences, get_vocab  # (Needed to handle Vocabulary pickle)
from data_loader import get_loader, DatasetParams
from model.encoder_decoder import ModelParams, EncoderDecoder
from utils import basename

try:
    from tqdm import tqdm
except ImportError as e:
    print('WARNING: tqdm module not found. Install it if you want a fancy progress bar :-)')

    def tqdm(x, disable=False): return x


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


# Device configuration now in infer()
device = None

class infer_object:
    def __init__(self, args):
        # print('__init__() :', args)

        if 'cpu' not in args:
            args['cpu'] = False
        if 'vocab' not in args:
            args['vocab'] = None
        if 'ext_features' not in args:
            args['ext_features'] = None
        if 'ext_persist_features' not in args:
            args['ext_persist_features'] = None
        
        global device
        device = torch.device('cuda' if torch.cuda.is_available() and
                              not args['cpu'] else 'cpu')

        # Build models
        print('Bulding models.')

        if device.type == 'cpu':
            self.state = torch.load(args['model'], map_location=lambda storage, loc: storage)
        else:
            self.state = torch.load(args['model'])
            
        self.params = ModelParams(self.state)
        if args['ext_features']:
            self.params.update_ext_features(args['ext_features'])
        if args['ext_persist_features']:
            self.params.update_ext_persist_features(args['ext_persist_features'])

        print('Loaded model parameters from <{}>:'.format(args['model']))
        print(self.params)

        # Load the vocabulary:
        if args['vocab'] is not None:
            # Loading vocabulary from file path supplied by the user:
            args_attr = AttributeDict(args)
            self.vocab = get_vocab(args_attr)
        elif self.params.vocab is not None:
            print('Loading vocabulary stored in the model file.')
            self.vocab = self.params.vocab
        else:
            print('ERROR: you must either load a model that contains vocabulary or '
                  'specify a vocabulary with the --vocab option!')
            sys.exit(1)

            print('Size of the vocabulary is {}'.format(len(self.vocab)))

        ef_dims = None
        self.model = EncoderDecoder(self.params, device, len(self.vocab), self.state, ef_dims).eval()
        # print(self.params.ext_features_dim)


    def external_features(self):
        ef_dims = self.params.ext_features_dim
        ret = [(self.state['features'].external,         ef_dims[0]),
               (self.state['persist_features'].external, ef_dims[1])]
        # print(ret)
        return ret


    def infer(self, args):
        # print('infer() :', args)

        if 'image_features' not in args:
            args['image_features'] = None
    
        # Image preprocessing
        transform = transforms.Compose([
            transforms.Resize((args['resize'], args['resize'])),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])

        # Get dataset parameters:
        dataset_configs = DatasetParams(args['dataset_config_file'])
        dataset_params = dataset_configs.get_params(args['dataset'],
                                                    args['image_dir'],
                                                    args['image_files'],
                                                    args['image_features'])

        if self.params.has_external_features() and \
           any(dc.name == 'generic' for dc in dataset_params):
            print('WARNING: you cannot use external features without specifying all datasets in '
                  'datasets.conf.')
            print('Hint: take a look at datasets/datasets.conf.default.')

        # Build data loader
        print("Loading dataset: {}".format(args['dataset']))

        # Update dataset params with needed model params:
        for i in dataset_params:
            i.config_dict['skip_start_token'] = self.params.skip_start_token
            # For visualizing attention we need file names instead of IDs in our output:
            if args['store_image_paths']:
                i.config_dict['return_image_file_name'] = True

        ext_feature_sets = [self.params.features.external, self.params.persist_features.external]
        if args['dataset']=='incore':
            ext_feature_sets = None
        
        # We ask it to iterate over images instead of all (image, caption) pairs
        data_loader, ef_dims = get_loader(dataset_params, vocab=None, transform=transform,
                                          batch_size=args['batch_size'], shuffle=False,
                                          num_workers=args['num_workers'],
                                          ext_feature_sets=ext_feature_sets,
                                          skip_images=not self.params.has_internal_features(),
                                          iter_over_images=True)

        self.data_loader = data_loader
        # Create model directory
        if not os.path.exists(args['results_path']):
            os.makedirs(args['results_path'])

        scorers = {}
        if args['scoring'] is not None:
            for s in args['scoring'].split(','):
                s = s.lower().strip()
                if s == 'cider':
                    from eval.cider import Cider
                    scorers['CIDEr'] = Cider(df='corpus')

        # Store captions here:
        output_data = []

        gts = {}
        res = {}

        print('Starting inference, max captions/sentence length: {}'.format(args['max_seq_length']))
        show_progress = sys.stderr.isatty() and not args['verbose']
        for i, (images, ref_captions, lengths, image_ids,
                features) in enumerate(tqdm(self.data_loader, disable=not show_progress)):

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

            init_features    = features[0].to(device) if len(features) > 0 and \
                               features[0] is not None else None
            persist_features = features[1].to(device) if len(features) > 1 and \
                               features[1] is not None else None

            # Generate a caption from the image
            sampled_batch = self.model.sample(images, init_features, persist_features,
                                              max_seq_length=args['max_seq_length'],
                                              start_token_id=self.vocab('<start>'),
                                              end_token_id=self.vocab('<end>'))

            sampled_ids_batch = sampled_batch

            for i in range(sampled_ids_batch.shape[0]):
                sampled_ids = sampled_ids_batch[i]

                # Convert word_ids to words
                if self.params.hierarchical_model:
                    caption = paragraph_ids_to_words(sampled_ids, self.vocab)
                else:
                    caption = caption_ids_to_words(sampled_ids, self.vocab)

                if args['no_repeat_sentences']:
                    caption = remove_duplicate_sentences(caption)

                if args['only_complete_sentences']:
                    caption = remove_incomplete_sentences(caption)

                if args['verbose']:
                    print('=>', caption)

                output_data.append({'caption': caption, 'image_id': image_ids[i]})
                res[image_ids[i]] = [caption.lower()]

        for score_name, scorer in scorers.items():
            score = scorer.compute_score(gts, res)[0]
            print('Test', score_name, score)

        # Decide output format, fall back to txt
        if args['output_format'] is not None:
            output_format = args['output_format']
        elif args['output_file'] and args['output_file'].endswith('.json'):
            output_format = 'json'
        else:
            output_format = 'txt'

        # Create a sensible default output path for results:
        output_file = None
        if not args['output_file'] and not args['print_results']:
            model_name = args['model'].split(os.sep)[-2]
            model_epoch = basename(args['model'])
            output_file = '{}-{}.{}'.format(model_name, model_epoch, output_format)
        else:
            output_file = args['output_file']

        if output_file:
            output_path = os.path.join(args['results_path'], output_file)
            if output_format == 'json':
                json.dump(output_data, open(output_path, 'w'))
            else:
                with open(output_path, 'w') as fp:
                    for data in output_data:
                        print(data['image_id'], data['caption'], file=fp)

            print('Wrote generated captions to {} as {}'.format(output_path, args['output_format']))

        if args['print_results']:
            for d in output_data:
                print('{}: {}'.format(d['image_id'], d['caption']))

        return output_data


def inferx(a):
    args = argparse.Namespace()
    args.image_files             = []
    args.dataset                 = 'incore'
    args.resize                  =  224
    args.batch_size              = 128
    args.num_workers             = 2
    args.max_seq_length          = 20
    args.results_path            = '/tmp'
    args.ext_features            = []
    args.ext_persist_features    = []
    args.store_image_paths       = ''
    args.dataset_config_file     = ''
    args.scoring                 = ''
    args.image_dir               = ''
    args.vocab                   = None
    args.output_format           = None
    args.output_file             = False
    args.verbose                 = False
    args.no_repeat_sentences     = False
    args.only_complete_sentences = False
    args.print_results           = True
    d = vars(args)
    for k, v in a.items():
        d[k] = v;
    return main(args)


def main(args):
    a = vars(args)
    infobj = infer_object(a)
    infobj.infer(a)

    
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
    parser.add_argument('--results_path', type=str, default='output/results/',
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
    args = parse_args()

    begin = datetime.now()
    print('Started inference at {}.'.format(begin))

    main(args=args)

    end = datetime.now()
    print('Inference ended at {}. Total time: {}.'.format(end, end - begin))
