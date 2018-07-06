import argparse
import pickle

from torchvision import transforms

import build_vocab
import resize
from data_loader import get_loader

if __name__ == '__main__':
    print('main')

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, default='vist-seq',
    #                     help='which dataset to use')
    # parser.add_argument('--caption_path', type=str,
    #                     default='resources/sis/train.story-in-sequence.json',
    #                     help='path for train annotation file')
    # parser.add_argument('--vocab_path', type=str,
    #                     default='resources/vocab_train.pkl',
    #                     help='path for saving vocabulary wrapper')
    # parser.add_argument('--threshold', type=int, default=4,
    #                     help='minimum word count threshold')
    # args = parser.parse_args()
    # build_vocab.main(args)
    print('building vocab complete')

    parser = argparse.ArgumentParser()
    # parser.add_argument('--image_dir', type=str,
    #                     default='resources/images/train',
    #                     help='directory for train images')
    # parser.add_argument('--class_filter', type=str, default=None,
    #                     help='path to (optional) new-line separated file '
    #                          'listing ids of images to include')
    # parser.add_argument('--output_dir', type=str,
    #                     default='resources/images/train/resized',
    #                     help='directory for saving resized images')
    # parser.add_argument('--create_zip', action="store_true",
    #                     help='save ZIP file as "\{output_dir\}.zip"')
    # parser.add_argument('--image_size', type=int, default=256,
    #                     help='size for image after processing')
    # args = parser.parse_args()
    # resize.main(args=args)
    print('resizing images complete')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='vist-seq',
                        help='which dataset to use')
    # parser.add_argument('--load_model', type=str,
    #                     help='existing model, for continuing training')
    # parser.add_argument('--model_basename', type=str, default='model',
    #                     help='base name for model snapshot filenames')
    # parser.add_argument('--model_path', type=str, default='models/',
    #                     help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str,
                        default='resources/vocab_train.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str,
                        default='resources/images/train/resized',
                        help='directory for resized images'
                             'if "image_dir" points to zip archive - extract '
                             'to /tmp/ , use the extracted images to train')
    # parser.add_argument('--tmp_dir_prefix', type=str, default='image_captioning',
    #                     help='where in /tmp folder to store project data')
    parser.add_argument('--caption_path', type=str,
                        default='resources/sis/train.story-in-sequence.json',
                        help='path for train annotation json file')
    # parser.add_argument('--log_step', type=int, default=10,
    #                     help='step size for prining log info')

    # Model parameters
    # parser.add_argument('--embed_size', type=int, default=256,
    #                     help='dimension of word embedding vectors')
    # parser.add_argument('--hidden_size', type=int, default=512,
    #                     help='dimension of lstm hidden states')
    # parser.add_argument('--num_layers', type=int, default=1,
    #                     help='number of layers in lstm')
    # parser.add_argument('--dropout', type=float, default=0,
    #                     help='dropout for the LSTM')

    # Training parameters
    # parser.add_argument('--force_epoch', type=int, default=0,
    #                     help='Force start epoch (for broken model files...)')
    # parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    # parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()

    transform = transforms.Compose([
        # transforms.Resize((256, 256)),
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        print("Extracting vocabulary from {}".format(args.vocab_path))
        vocab = pickle.load(f)

    # Build data loader
    print("Loading dataset: {}".format(args.dataset))
    data_loader = get_loader(args.dataset, args.image_dir, args.caption_path,
                             vocab, transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers)

    print('loading data complete, verifying')
    total_step = len(data_loader)
    for batch in data_loader:
        print('sequence size: ', len(batch[0]))
        print('sequence image shape: ', batch[0][0].shape)
        print('story shape: ', batch[1].shape)
        break
    print('Go design your model!')
