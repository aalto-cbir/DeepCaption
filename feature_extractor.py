import argparse
import os
import pickle
from glob import glob

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

from model import ModelParams, EncoderCNN

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])
image_types = ('*.jpg', '*.png')


class Extractor(nn.Module):
    def __init__(self, model_name):
        """Load the pretrained model and replaces top fc layer."""
        super(Extractor, self).__init__()

        if model_name == 'alexnet':
            print('using AlexNet ... features shape (256 * 6 * 6, ) ')
            model = models.alexnet(pretrained=True)
        elif model_name == 'densenet201':
            print('using DenseNet 201 ... features shape (1920, 7, 7, ) ')
            model = models.densenet201(pretrained=True)
        else:
            print('using resnet 152 ... features shape (2048, ) ')
            model = models.resnet152(pretrained=True)

        modules = list(model.children())[:-1]
        self.extractor = nn.Sequential(*modules)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.extractor(images)
        features = features.reshape(features.size(0), -1)
        return features


def torchify_image(batch):
    final_tensor = torch.tensor([])
    for image_path in batch:
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        image = image.unsqueeze(0)
        final_tensor = torch.cat([final_tensor, image])

    return final_tensor.to(device)


def extract_features(image_paths, extractor, output_dir, batch_size=4):
    for idx in range(0, len(image_paths), batch_size):
        batch = image_paths[idx:idx + batch_size]
        images = torchify_image(batch)
        features = extractor(images).data.cpu().numpy()
        for _idx in range(len(batch)):
            image_path = batch[_idx]
            feature_file = output_dir + os.path.basename(image_path).split('.')[0] + '.pkl'
            with open(feature_file, 'wb') as file:
                pickle.dump(features[_idx], file)


def main(args):
    extractor = Extractor(args.extractor).to(device)

    image_paths = []
    for image_type in image_types:
        image_paths.extend(glob(args.image_dir + image_type))

    extract_features(image_paths, extractor, args.output_dir)


if __name__ == '__main__':
    print('\nextracting image features')
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str,
                        default='./resources/images/train/resized/',
                        help='images for which features are needed')
    parser.add_argument('--output_dir', type=str,
                        default='./features/',
                        help='directory for saving image features')
    parser.add_argument('--extractor', type=str,
                        default='densenet201',
                        help='name of the extractor, ex: alexnet, resnet152')

    arguments = parser.parse_args()
    main(args=arguments)
