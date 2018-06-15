import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle

from data_loader import get_loader 
from build_vocab import Vocabulary
from datetime import datetime
from model import ModelParams, EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_models(args, params, encoder, decoder, optimizer, epoch, step):
    bn = args.model_basename
    if step==0:
        file_name = '{}-{}-final.ckpt'.format(bn, epoch+1)
        epoch += epoch # set to start of next epoch
    else:
        file_name = '{}-{}-{}.ckpt'.format(bn, epoch+1, step+1)

    state = {
        'epoch': epoch,
        'step' : step,
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'embed_size': params.embed_size,
        'hidden_size': params.hidden_size,
        'num_layers': params.num_layers,
    }

    torch.save(state, os.path.join(args.model_path, file_name))

    
def load_models(args, model_path, encoder, decoder, optimizer):
    state = torch.load(model_path)
    encoder.load_state_dict(state['encoder'])
    decoder.load_state_dict(state['decoder'])
    optimizer.load_state_dict(state['optimizer'])

    va = vars(args)
    for a in 'embed_size', 'hidden_size', 'num_layers':
        assert state[a] == va[a]
    return state['epoch'], state['step']


def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        # transforms.Resize((256, 256)),
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # Build data loader
    data_loader = get_loader(args.image_dir, args.caption_path, vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers) 

    params = ModelParams.fromargs(args)
    
    # Build the models
    encoder = EncoderCNN(params.embed_size).to(device)
    decoder = DecoderRNN(params.embed_size, params.hidden_size, len(vocab), 
                         params.num_layers).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + \
             list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    start_epoch = 0
    start_step = 0
    if args.load_model:
        start_epoch, start_step = load_models(args.load_model, encoder, decoder,
                                              optimizer)
        print('Loading model {} at epoch {} and step {}.'.format(
            args.load_model, start_epoch+1, start_step+1))
    
    # Train the models
    total_step = len(data_loader)
    for epoch in range(start_epoch, args.num_epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):
            if i < start_step:
                continue
            
            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            # Forward, backward and optimize
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            if (i+1) % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch+1, args.num_epochs, i+1, total_step, 
                              loss.item(), np.exp(loss.item()))) 
                
            # Save the model checkpoints
            if args.save_step and (i+1) % args.save_step == 0:
                save_models(args, params, encoder, decoder, optimizer, epoch, i)

        start_step = 0

        save_models(args, params, encoder, decoder, optimizer, epoch, 0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', type=str,
                        help='existing model, for continuing training')
    parser.add_argument('--model_basename', type=str, default='model',
                        help='base name for model snapshot filenames')
    parser.add_argument('--model_path', type=str, default='models/' , 
                        help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224, 
                        help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='data/resized2014', 
                        help='directory for resized images')
    parser.add_argument('--caption_path', type=str,
                        default='data/annotations/captions_train2014.json', 
                        help='path for train annotation json file')
    parser.add_argument('--log_step', type=int, default=10, 
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int,
                        help='step size for saving trained models')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256, 
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, 
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, 
                        help='number of layers in lstm')

    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    args = parser.parse_args()

    begin = datetime.now()
    print('Started training at {}, with parameters:'.format(begin))
    for k, v in vars(args).items(): print('[args] {}={}'.format(k, v))

    main(args)

    end = datetime.now()
    print('Training ended at {}. Total training time: {}.'.format(end, end-begin))
