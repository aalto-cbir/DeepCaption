import os
import glob
import re
import json
import math
import numpy as np
from PIL import Image
import torch
from torch.nn.utils.rnn import pack_padded_sequence


def basename(fname, split=None):
    if split is not None:
        fname.split(split)
    return os.path.splitext(os.path.basename(fname))[0]


def feats_to_str(feats):
    return '+'.join(feats.internal + [os.path.splitext(os.path.basename(f))[0]
                                      for f in feats.external])


# This is to print the float without exponential-notation, and without trailing zeros.
# Normal formatting, e.g.: '{:f}'.format(0.01) produces "0.010000"
def f2s(f):
    return '{:0.16f}'.format(f).rstrip('0')


def get_model_name(args, params):
    """Create model name"""

    if args.model_name is not None:
        model_name = args.model_name
    elif args.load_model:
        model_name = os.path.split(os.path.dirname(args.load_model))[-1]

    else:
        bn = args.model_basename

        feat_spec = feats_to_str(params.features)
        if params.has_persist_features():
            feat_spec += '-' + feats_to_str(params.persist_features)

        model_name = ('{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.
                      format(bn, params.embed_size, params.hidden_size, params.num_layers,
                             params.batch_size, args.optimizer, f2s(params.learning_rate),
                             f2s(args.weight_decay), params.dropout, params.encoder_dropout,
                             feat_spec))
    return model_name


def get_model_path(args, params, epoch):
    model_name = get_model_name(args, params)
    file_name = 'ep{}.model'.format(epoch)
    model_path = os.path.join(args.output_root, args.model_path, model_name, file_name)
    return model_path


# TODO: convert parameters to **kwargs
def save_model(args, params, encoder, decoder, optimizer, epoch, vocab):
    state = {
        'hierarchical_model': params.hierarchical_model,
        'epoch': epoch + 1,
        # Attention models can in principle be trained without an encoder:
        'encoder': encoder.state_dict() if encoder is not None else None,
        'decoder': decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'embed_size': params.embed_size,
        'hidden_size': params.hidden_size,
        'num_layers': params.num_layers,
        'batch_size': params.batch_size,
        'learning_rate': params.learning_rate,
        'dropout': params.dropout,
        'encoder_dropout': params.encoder_dropout,
        'encoder_non_lin': params.encoder_non_lin,
        'features': params.features,
        'persist_features': params.persist_features,
        'attention': params.attention,
        'vocab': vocab,
        'skip_start_token': params.skip_start_token,
        'rnn_arch': params.rnn_arch,
        'rnn_hidden_init': params.rnn_hidden_init,
        'share_embedding_weights': params.share_embedding_weights
    }

    if params.hierarchical_model:
        state['max_sentences'] = params.max_sentences
        state['dropout_stopping'] = params.dropout_stopping
        state['dropout_fc'] = params.dropout_fc
        state['fc_size'] = params.fc_size
        state['coherent_sentences'] = params.coherent_sentences
        state['coupling_alpha'] = params.coupling_alpha
        state['coupling_beta'] = params.coupling_beta

    model_path = get_model_path(args, params, epoch + 1)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    torch.save(state, model_path)
    print('Saved model as {}'.format(model_path))
    if args.verbose:
        print(params)


def stats_filename(args, params, postfix):
    model_name = get_model_name(args, params)
    model_dir = os.path.join(args.output_root, args.model_path, model_name)

    if postfix is None:
        json_name = 'train_stats.json'
    else:
        json_name = 'train_stats-{}.json'.format(postfix)

    return os.path.join(model_dir, json_name)


def init_stats(args, params, postfix=None):
    filename = stats_filename(args, params, postfix)
    if os.path.exists(filename):
        with open(filename, 'r') as fp:
            return json.load(fp)
    else:
        return dict()


def save_stats(args, params, all_stats, postfix=None, writer=None):
    filename = stats_filename(args, params, postfix)
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as outfile:
        json.dump(all_stats, outfile, indent=2)
    print('Wrote stats to {}'.format(filename))

    # Write events to tensorboardx if available:
    if writer is not None:
        epoch = max([int(k) for k in all_stats.keys()])
        writer.add_scalars('train_stats', all_stats[str(epoch)], epoch)


def log_model_data(params, model, n_iter, writer):
    """Log model data using tensorboard"""

    def _get_weights(x):
        """Clone tensor x to numpy for logging"""
        return x.clone().cpu().detach().numpy()

    if params.hierarchical_model:
        word_decoder = model.decoder.word_decoder
        sent_decoder = model.decoder

        # Log Coherent model data:
        if params.coherent_sentences:
            cu = sent_decoder.coupling_unit
            writer.add_histogram('weights/coupling/linear_1',
                                 _get_weights(cu.linear1.weight),
                                 n_iter)
            writer.add_histogram('weights/coupling/linear_2',
                                 _get_weights(cu.linear2.weight),
                                 n_iter)
            writer.add_histogram('weights/coupling/gru_hh_l0',
                                 _get_weights(cu.gate.weight_hh_l0),
                                 n_iter)
            writer.add_histogram('weights/coupling/gru_ih_l0',
                                 _get_weights(cu.gate.weight_ih_l0),
                                 n_iter)

        # Log SentenceRNN data
        writer.add_histogram('weights/sentence_RNN/linear_1',
                             _get_weights(sent_decoder.linear1.weight),
                             n_iter)
        writer.add_histogram('weights/sentence_RNN/linear_2',
                             _get_weights(sent_decoder.linear2.weight),
                             n_iter)
        writer.add_histogram('weights/sentence_RNN/rnn_hh_l0',
                             _get_weights(sent_decoder.sentence_rnn.weight_hh_l0),
                             n_iter)
        writer.add_histogram('weights/sentence_RNN/rnn_ih_l0',
                             _get_weights(sent_decoder.sentence_rnn.weight_ih_l0),
                             n_iter)
    else:
        word_decoder = model.decoder

    # Log WordRNN data:
    writer.add_histogram('weights/word_RNN/embed',
                         _get_weights(word_decoder.embed.weight),
                         n_iter)
    writer.add_histogram('weights/word_RNN/rnn_hh_l0',
                         _get_weights(word_decoder.rnn.weight_hh_l0),
                         n_iter)
    writer.add_histogram('weights/word_RNN/rnn_ih_l0',
                         _get_weights(word_decoder.rnn.weight_ih_l0),
                         n_iter)

    if params.share_embedding_weights:
        writer.add_histogram('weights/word_RNN/embedding_projection',
                             _get_weights(word_decoder.projection.weight),
                             n_iter)


def get_teacher_prob(k, i, beta=1):
    """Inverse sigmoid sampling scheduler determines the probability
    with which teacher forcing is turned off, more info here:
    https://arxiv.org/pdf/1506.03099.pdf"""
    if k == 0:
        return 1.0

    i = i * beta
    p = k / (k + np.exp(i / k))

    return p


# Simple gradient clipper from tutorial, can be replaced with torch's own
# using it now to stay close to reference Attention implementation
def clip_gradients(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def prepare_hierarchical_targets(last_sentence_indicator, max_sentences, lengths, captions, device):
    """Prepares the training targets used by hierarchical model"""
    # Validate that the last sentence indicator is outputting correct data:
    last_sentence_indicator = last_sentence_indicator.to(device)
    word_rnn_targets = []
    for j in range(max_sentences):
        if lengths[0, j] == 0:
            break  # no more sentences at position >= j in current minibatch

        # print(lengths[:, i])
        # change to offset / first occurance of zero instead of indices
        non_zero_idxs = lengths[:, j] > 0
        # print(j)
        # print('lengths[:, j] {}'.format(lengths[:, j]))
        # print('print: captions[:, j] {}'.format(captions[:, j]))
        # print('non zero indices: {}'.format(non_zero_idxs))
        # print('filtered: {}'.format(lengths[:, i][non_zero_idxs]))
        # Pack the non-zero values for each sentence position:
        packed = pack_padded_sequence(captions[:, j][non_zero_idxs],
                                      lengths[:, j][non_zero_idxs],
                                      batch_first=True)[0]
        word_rnn_targets.append(packed)

    targets = (last_sentence_indicator, word_rnn_targets)

    return targets


def cyclical_lr(step_sz, min_lr=0.001, max_lr=1, mode='triangular', scale_func=None,
                scale_md='cycles', gamma=1.):
    """implements a cyclical learning rate policy (CLR).
    Notes: the learning rate of optimizer should be 1

    Parameters:
    ----------
    mode : str, optional
        one of {triangular, triangular2, exp_range}.
    scale_md : str, optional
        {'cycles', 'iterations'}.
    gamma : float, optional
        constant in 'exp_range' scaling function: gamma**(cycle iterations)

    Examples:
    --------
    >>> # the learning rate of optimizer should be 1
    >>> optimizer = optim.SGD(model.parameters(), lr=1.)
    >>> step_size = 2*len(train_loader)
    >>> clr = cyclical_lr(step_size, min_lr=0.001, max_lr=0.005)
    >>> scheduler = lr_scheduler.LambdaLR(optimizer, [clr])
    >>> # some other operations
    >>> scheduler.step()
    >>> optimizer.step()

    Source: https://github.com/pytorch/pytorch/pull/2016#issuecomment-387755710
    """
    if scale_func is None:
        if mode == 'triangular':
            scale_fn = lambda x: 1.
            scale_mode = 'cycles'
        elif mode == 'triangular2':
            scale_fn = lambda x: 1 / (2. ** (x - 1))
            scale_mode = 'cycles'
        elif mode == 'exp_range':
            scale_fn = lambda x: gamma ** (x)
            scale_mode = 'iterations'
        else:
            raise ValueError('The {} is not valid value!'.format(mode))
    else:
        scale_fn = scale_func
        scale_mode = scale_md

    lr_lambda = lambda iters: min_lr + (max_lr - min_lr) * rel_val(iters, step_sz, scale_mode)

    def rel_val(iteration, stepsize, mode):
        cycle = math.floor(1 + iteration / (2 * stepsize))
        x = abs(iteration / stepsize - 2 * cycle + 1)
        if mode == 'cycles':
            return max(0, (1 - x)) * scale_fn(cycle)
        elif mode == 'iterations':
            return max(0, (1 - x)) * scale_fn(iteration)
        else:
            raise ValueError('The {} is not valid value!'.format(scale_mode))

    return lr_lambda


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


def fix_caption(caption):
    m = re.match(r'^<start> (.*?)( <end>)?$', caption)
    if m is None:
        print('ERROR: unexpected caption format: "{}"'.format(caption))
        return caption.capitalize()

    ret = m.group(1)
    ret = re.sub(r'\s([.,])(\s|$)', r'\1\2', ret)
    return ret.capitalize()


def torchify_sequence(batch):
    final_tensor = torch.tensor([])
    for image in batch:
        image = image.unsqueeze(0)
        final_tensor = torch.cat([final_tensor, image])

    return final_tensor


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()
