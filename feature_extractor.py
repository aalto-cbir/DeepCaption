import argparse
import os
import pickle
from glob import glob

import torch
from PIL import Image
from torchvision import transforms

from model import ModelParams, EncoderCNN

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    # transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])
image_types = ('*.jpg', '*.png')


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def torchify_image(batch):
    final_tensor = torch.tensor([])
    for image_path in batch:
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        image = image.unsqueeze(0)
        final_tensor = torch.cat([final_tensor, image])

    return final_tensor.to(device)


def extract_features(image_paths, encoder, output_dir, batch_size=4):
    for idx in range(0, len(image_paths), batch_size):
        batch = image_paths[idx:idx + batch_size]
        images = torchify_image(batch)
        features = encoder(images).data.cpu().numpy()
        for _idx in range(len(batch)):
            image_path = batch[_idx]
            feature_file = output_dir + os.path.basename(image_path).split('.')[0] + '.pkl'
            with open(feature_file, 'wb') as file:
                pickle.dump(features[_idx], file)


def main(args):
    if args.extractor == 'resnet-152':
        params = ModelParams.fromargs(args)
        encoder = EncoderCNN(params).to(device)
    else:
        print('yet to implement other extractors')
        return

    image_paths = []
    for image_type in image_types:
        image_paths.extend(glob(args.image_dir + image_type))

    extract_features(image_paths, encoder, args.output_dir)


if __name__ == '__main__':
    print('\nextracting image features')
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str,
                        default='./resized_temp/',
                        help='images for which features are needed')
    parser.add_argument('--class_filter', type=str, default=None,
                        help='path to (optional) new-line separated file '
                             'listing ids of images to include')
    parser.add_argument('--output_dir', type=str,
                        default='./features/',
                        help='directory for saving image features')
    parser.add_argument('--extractor', type=str,
                        default='resnet-152',
                        help='name of the extractor, ex: InceptionV3, resnet-152')

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=256,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in lstm')
    parser.add_argument('--dropout', type=float, default=0,
                        help='dropout for the LSTM')

    arguments = parser.parse_args()
    main(args=arguments)
