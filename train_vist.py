import argparse
import os
import pickle
import sys
import torch
import numpy as np

from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

from data_loader import get_loader, collate_fn_vist
from utils import torchify_sequence

# Device configuration
from model_vist import DecoderRNN, ModelParams, EncoderCNN, EncoderRNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_models(args, params, encoder_cnn, encoder_rnn, decoder, optimizer, epoch):
    bn = args.model_basename
    file_name = '{}-ep{}.ckpt'.format(bn, epoch + 1)

    state = {
        'epoch': epoch + 1,
        'encoder_cnn': encoder_cnn.state_dict(),
        'encoder_rnn': encoder_rnn.state_dict(),
        'decoder': decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'embed_size': params.embed_size,
        'hidden_size': params.hidden_size,
        'num_layers': params.num_layers,
        'dropout': params.dropout
    }

    torch.save(state, os.path.join(args.model_path, file_name))


def main(args):
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
    data_loader = get_loader(args.dataset, args.image_dir, args.caption_path,
                             vocab, transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers, _collate_fn=collate_fn_vist)

    print('data load complete, verifying size', len(data_loader))

    state = None
    params = ModelParams.fromargs(args)
    start_epoch = 0

    if args.load_model:
        state = torch.load(args.load_model)
        params = ModelParams(state)
        start_epoch = state['epoch']
        print('Loading model {} at epoch {}.'.format(args.load_model,
                                                     start_epoch))
    if args.force_epoch:
        start_epoch = args.force_epoch - 1

    # Build the models
    print('Using device: {}'.format(device.type))
    print('Initializing model...')
    encoder_cnn = EncoderCNN(params).to(device)
    encoder_rnn = EncoderRNN(params).to(device)
    decoder = DecoderRNN(params, len(vocab)).to(device)

    if state:
        encoder_cnn.load_state_dict(state['encoder_cnn'])
        encoder_rnn.load_state_dict(state['encoder_rnn'])
        decoder.load_state_dict(state['decoder'])

    print('modelling and training on 1 data sample')

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    opt_params = (list(decoder.parameters()) +
                  list(encoder_cnn.linear.parameters()) +
                  list(encoder_cnn.bn.parameters()))
    optimizer = torch.optim.Adam(opt_params, lr=args.learning_rate)
    if state:
        optimizer.load_state_dict(state['optimizer'])

    # Train the models
    total_step = len(data_loader)
    print('Start Training...')
    for epoch in range(start_epoch, args.num_epochs):
        for i, (images, captions, lengths, story_id) in enumerate(data_loader):
            # forward pass
            sequence_data = torchify_sequence(images[0]).to(device)
            # print('shape of image data: ', sequence_data.shape)
            sequence_features = encoder_cnn(sequence_data)
            # print('shape of image features: ', sequence_features.shape)
            input_sequence_features = sequence_features.unsqueeze(0)
            input_sequence_features = input_sequence_features.view(1, 1, -1)
            # print('shape of input to EncoderRNN: ', input_sequence_features.shape)
            context_vector = encoder_rnn(input_sequence_features)
            # print('shape of sequence context vector: ', context_vector.shape)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            # print('shape of targets: ', targets.shape)
            outputs = decoder(context_vector, captions, lengths)
            # print('shape of decoder output: ', outputs.shape)

            # backward pass and optimize
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder_cnn.zero_grad()
            loss.backward()
            optimizer.step()

            # print log info
            if (i + 1) % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch + 1, args.num_epochs, i + 1, total_step,
                              loss.item(), np.exp(loss.item())))
                sys.stdout.flush()

        save_models(args, params, encoder_cnn, encoder_rnn, decoder, optimizer, epoch)


if __name__ == '__main__':
    print('train vist')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='vist-seq',
                        help='which dataset to use')
    parser.add_argument('--load_model', type=str,
                        help='existing model, for continuing training')
    parser.add_argument('--model_basename', type=str, default='model',
                        help='base name for model snapshot filenames')
    parser.add_argument('--model_path', type=str, default='models/',
                        help='path for saving trained models')
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
    parser.add_argument('--tmp_dir_prefix', type=str, default='image_captioning',
                        help='where in /tmp folder to store project data')
    parser.add_argument('--caption_path', type=str,
                        default='resources/sis/val.story-in-sequence.json',
                        help='path for train annotation json file')
    parser.add_argument('--log_step', type=int, default=5,
                        help='step size for prining log info')

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=256,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in lstm')
    parser.add_argument('--dropout', type=float, default=0,
                        help='dropout for the LSTM')

    # Training parameters
    parser.add_argument('--force_epoch', type=int, default=0,
                        help='Force start epoch (for broken model files...)')
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    main(parser.parse_args())
