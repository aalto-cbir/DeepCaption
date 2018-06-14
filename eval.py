import argparse
import glob
import json
# import matplotlib.pyplot as plt
import numpy as np 
import pickle
import re
import sys
import torch

from PIL import Image
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from torchvision import transforms 
from tqdm import tqdm
from os.path import basename, splitext

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def fix_caption(caption):
    m = re.match(r'^<start> (.*?)( <end>)?$', caption)
    if m is None:
        print('ERROR: unexpected caption format: "{}"'.format(caption))
        sys.exit(1)              

    ret = m.group(1)
    ret = re.sub(r'\s([.,])(\s|$)',r'\1\2', ret)
    return ret.capitalize()

def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    if image.mode != 'RGB':
        print('WARNING: converting {} to RGB'.format(image_path))
        image = image.convert('RGB')
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    print('Reading vocabulary from {}.'.format(args.vocab_path))
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    print('Bulding models.')
    encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    print('Reading trained encoder model from {}'.format(args.encoder_path))
    encoder.load_state_dict(torch.load(args.encoder_path))

    print('Reading trained decoder model from {}'.format(args.decoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    output_data = []
    file_list = glob.glob(args.image_dir + '/*.jpg')
    N = len(file_list)
    print('Directory {} contains {} files.'.format(args.image_dir, N))
    for i, image_file in tqdm(enumerate(file_list), disable=args.verbose):
        m = re.search(r'0*(\d+)$', splitext(basename(image_file))[0])
        if m is None:
            print('Unable to parse COCO image_id from filename {}!'.format(image_file))
            sys.exit(1)
        image_id = int(m.group(1))
        assert image_id > 0

        if args.verbose:
            print('[{:.2%}] Reading [{}] as {} ...'.format(i / N, image_id, image_file))
        # Prepare an image
        image = load_image(image_file, transform)
        image_tensor = image.to(device)

        # Generate an caption from the image
        feature = encoder(image_tensor)
        sampled_ids = decoder.sample(feature)
        sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)

        # Convert word_ids to words
        sampled_caption = []
        for word_id in sampled_ids:
            word = vocab.idx2word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break
        sentence = fix_caption(' '.join(sampled_caption))

        if args.verbose:
            print('=>', sentence)

        output_data.append({'caption': sentence, 'image_id': image_id})

    json.dump(output_data, open(args.output_file, 'w'))
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='data/val2014', help='input image dir for generating captions')
    parser.add_argument('--encoder_path', type=str, default='models/encoder-5-3000.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/decoder-5-3000.ckpt', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--output_file', type=str, default='output.json', help='path for output JSON')
    parser.add_argument('--verbose', help='verbose output', action='store_true')
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    args = parser.parse_args()
    main(args)
