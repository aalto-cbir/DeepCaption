import os

import torch
import torch.nn as nn
import torchvision.models as models

import numpy as np

from collections import OrderedDict, namedtuple
from torch.nn.utils.rnn import pack_padded_sequence

import external_models as ext_models

Features = namedtuple('Features', 'external, internal')


class ModelParams:
    def __init__(self, d):
        self.embed_size = self._get_param(d, 'embed_size', 256)
        self.hidden_size = self._get_param(d, 'hidden_size', 512)
        self.num_layers = self._get_param(d, 'num_layers', 1)
        self.batch_size = self._get_param(d, 'batch_size', 128)
        self.dropout = self._get_param(d, 'dropout', 0)
        self.learning_rate = self._get_param(d, 'learning_rate', 0.001)
        self.features = self._get_features(d, 'features', 'resnet152')
        self.persist_features = self._get_features(d, 'persist_features', '')
        self.attention_features = self._get_features(d, 'attention_features', '')
        self.encoder_dropout = self._get_param(d, 'encoder_dropout', 0)

    @classmethod
    def fromargs(cls, args):
        return cls(vars(args))

    def _get_param(self, d, param, default):
        if param not in d or d[param] is None:
            print('WARNING: {} not set, using default value {}'.
                  format(param, default))
            return default
        return d[param]

    def _get_features(self, d, param, default):
        p = self._get_param(d, param, default)

        # If it's already of type Features, just return it
        if hasattr(p, 'internal'):
            return p

        features = p.split(',') if p else []

        ext_feat = []
        int_feat = []
        for fn in features:
            # Check if feature has an extension, if yes assum it's
            # an external feature contained in a file with extension '.$ext':
            (tmp, ext) = os.path.splitext(fn)
            if ext:
                ext_feat.append(fn)
            else:
                int_feat.append(fn)

        return Features(ext_feat, int_feat)

    def has_persist_features(self):
        return self.persist_features.internal or self.persist_features.external

    def has_internal_features(self):
        return self.features.internal or self.persist_features.internal

    def has_external_features(self):
        return self.features.external or self.persist_features.external

    def update_ext_features(self, ef):
        self.features = self._update_ext_features(ef, self.features)

    def update_ext_persist_features(self, ef):
        self.persist_features = self._update_ext_features(ef,
                                                          self.persist_features)

    def _update_ext_features(self, ef, features):
        return features._replace(external=ef.split(','))

    def __str__(self):
        return '\n'.join(['[ModelParams] {}={}'.format(k, v) for k, v in
                          self.__dict__.items()])


class FeatureExtractor(nn.Module):
    def __init__(self, model_name, debug=False):
        """Load the pretrained model and replace top fc layer.
        Inception assumes input image size to be 299x299.
        Other models assume input image of size 224x224
        More info: https://pytorch.org/docs/stable/torchvision/models.html """
        super(FeatureExtractor, self).__init__()

        # Set flatten to False if we do not want to flatten the output features
        self.flatten = True

        if model_name == 'alexnet':
            if debug:
                print('Using AlexNet, features shape 256 x 6 x 6')
            model = models.alexnet(pretrained=True)
            self.output_dim = np.array([256 * 6 * 6], dtype=np.int32)
            modules = list(model.children())[:-1]
            self.extractor = nn.Sequential(*modules)
        elif model_name == 'densenet201':
            if debug:
                print('Using DenseNet 201, features shape 1920 x 7 x 7')
            model = models.densenet201(pretrained=True)
            self.output_dim = 1920 * 7 * 7
            modules = list(model.children())[:-1]
            self.extractor = nn.Sequential(*modules)
        elif model_name == 'resnet152':
            if debug:
                print('Using resnet 152, features shape 2048')
            model = models.resnet152(pretrained=True)
            self.output_dim = 2048
            modules = list(model.children())[:-1]
            self.extractor = nn.Sequential(*modules)
        elif model_name == 'resnet152-conv':
            if debug:
                print('Using resnet 152, '
                      'last convolutional layer, features shape 2048 x 7 x 7')
            model = models.resnet152(pretrained=True)
            self.output_dim = np.array([2048, 7, 7], dtype=np.int32)
            modules = list(model.children())[:-2]
            self.extractor = nn.Sequential(*modules)
            self.flatten = False
        elif model_name == 'resnet152caffe-torchvision':
            if debug:
                print('Using resnet 152 converted from caffe, features shape 2048')
            model = ext_models.resnet152caffe_torchvision(pretrained=True)
            self.output_dim = 2048
            modules = list(model.children())[:-1]
            self.extractor = nn.Sequential(*modules)
        elif model_name == 'resnet152caffe-original':
            if debug:
                print('Using resnet 152 converted from caffe, requires BGR images with '
                      ' pixel values in range 0..255 features shape 2048')
            model = ext_models.resnet152caffe_original(pretrained=True)
            self.output_dim = 2048
            modules = list(model.children())[:-1]
            self.extractor = nn.Sequential(*modules)
        elif model_name == 'resnet152caffe-conv':
            if debug:
                print('Using resnet 152 converted from caffe, '
                      'last convolutional layer, features shape 2048 x 7 x 7')
            model = ext_models.resnet152caffe_torchvision(pretrained=True)
            self.output_dim = np.array([2048, 7, 7], dtype=np.int32)
            modules = list(model.children())[:-2]
            self.extractor = nn.Sequential(*modules)
            self.flatten = False
        elif model_name == 'vgg16':
            if debug:
                print('Using vgg 16, features shape 4096')
            model = models.vgg16(pretrained=True)
            self.output_dim = 4096
            num_features = model.classifier[6].in_features
            features = list(model.classifier.children())[:-1]
            features.extend([nn.Linear(num_features, self.output_dim)])
            model.classifier = nn.Sequential(*features)
            self.extractor = model
        elif model_name == 'inceptionv3':
            if debug:
                print('Using Inception V3, features shape 1000')
                print('WARNING: Inception requires input images to be 299x299')
            model = models.inception_v3(pretrained=True)
            model.aux_logits = False
            self.extractor = model
        else:
            raise ValueError('Unknown model name: {}'.format(model_name))

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.extractor(images)

        if self.flatten:
            features = features.reshape(features.size(0), -1)
        return features

    @classmethod
    def list(cls, internal_features):
        el = nn.ModuleList()
        total_dim = 0
        for fn in internal_features:
            e = cls(fn)
            el.append(e)
            total_dim += e.output_dim
        return el, total_dim


class EncoderCNN(nn.Module):
    def __init__(self, p, ext_features_dim=0):
        """Load a pretrained CNN and replace top fc layer."""
        super(EncoderCNN, self).__init__()

        (self.extractors,
         int_features_dim) = FeatureExtractor.list(p.features.internal)

        # Sum of the dimensionalities of the concatenated features
        total_feat_dim = ext_features_dim + int_features_dim

        print('EncoderCNN: total feature dim={}'.format(total_feat_dim))

        # Add FC layer on top of features to get the desired output dimension
        self.linear = nn.Linear(total_feat_dim, p.embed_size)
        self.dropout = nn.Dropout(p=p.encoder_dropout)
        self.bn = nn.BatchNorm1d(p.embed_size, momentum=0.01)

    def forward(self, images, external_features=None):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            feat_outputs = []
            # Extract features with each extractor
            for extractor in self.extractors:
                feat_outputs.append(extractor(images))
            # Add external features
            if external_features is not None:
                feat_outputs.append(external_features)
            # Concatenate features
            features = torch.cat(feat_outputs, 1)
        # Apply FC layer, dropout and batch normalization
        features = self.bn(self.dropout(self.linear(features)))
        return features

    # hack to be able to load old state files which used the "resnet." prefix
    def load_state_dict(self, state_dict, strict=True):
        fixed_states = []
        for key, value in state_dict.items():
            if key.startswith('resnet.'):
                key = 'extractors.0.extractor.' + key[7:]
            fixed_states.append((key, value))

        fixed_state_dict = OrderedDict(fixed_states)
        super(EncoderCNN, self).load_state_dict(fixed_state_dict, strict)


class DecoderRNN(nn.Module):
    def __init__(self, p, vocab_size, ext_features_dim=0):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, p.embed_size)

        (self.extractors,
         int_features_dim) = FeatureExtractor.list(p.persist_features.internal)
        # Sum of the dimensionalities of the concatenated features
        total_feat_dim = ext_features_dim + int_features_dim

        print('DecoderCNN: total feature dim={}'.format(total_feat_dim))

        self.lstm = nn.LSTM(p.embed_size + total_feat_dim, p.hidden_size,
                            p.num_layers, dropout=p.dropout, batch_first=True)
        self.linear = nn.Linear(p.hidden_size, vocab_size)

    def _cat_features(self, images, external_features):
        """Concatenate internal and external features"""
        feat_outputs = []
        # Extract features with each extractor (internal feature)
        for ext in self.extractors:
            feat_outputs.append(ext(images))
        # Also add external features
        if external_features is not None:
            feat_outputs.append(external_features)
        # Return concatenated features, empty tensor if none
        return torch.cat(feat_outputs, 1) if feat_outputs else None

    def forward(self, features, captions, lengths, images, external_features=None):
        """Decode image feature vectors and generates captions."""

        # First, construct embeddings input, with initial feature as
        # the first: (batch_size, 1 + longest caption length, embed_size)
        embeddings = self.embed(captions)
        embeddings = torch.cat([features.unsqueeze(1), embeddings], 1)

        seq_length = embeddings.size()[1]

        with torch.no_grad():
            persist_features = self._cat_features(images, external_features)
            if persist_features is None:
                persist_features = features.new_empty(0)
            else:
                # Get into shape: batch_size, seq_length, embed_size
                persist_features = (persist_features.unsqueeze(1).
                                    expand(-1, seq_length, -1))

        inputs = torch.cat([embeddings, persist_features], 2)

        packed = pack_padded_sequence(inputs, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, images, external_features, states=None, max_seq_length=20):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []

        # Concatenate internal and external features
        persist_features = self._cat_features(images, external_features)
        if persist_features is None:
            persist_features = features.new_empty(0)

        # inputs: (batch_size, 1, embed_size + len(external features))
        inputs = torch.cat([features, persist_features], 1).unsqueeze(1)

        for i in range(max_seq_length):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)

            # inputs: (batch_size, 1, embed_size + len(external_features))
            embeddings = self.embed(predicted)
            inputs = torch.cat([embeddings, persist_features], 1).unsqueeze(1)

        # sampled_ids: (batch_size, max_seq_length)
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids


class EncoderDecoder(nn.Module):
    def __init__(self, params, device, vocab, state, ef_dims):
        """Vanilla EncoderDecoder model"""
        super(EncoderDecoder, self).__init__()
        print('Using device: {}'.format(device.type))
        print('Initializing EncoderDecoder model...')
        self.encoder = EncoderCNN(params, ef_dims[0]).to(device)
        self.decoder = DecoderRNN(params, len(vocab), ef_dims[1]).to(device)

        self.opt_params = (list(self.decoder.parameters()) +
                           list(self.encoder.linear.parameters()) +
                           list(self.encoder.bn.parameters()))

        if state:
            self.encoder.load_state_dict(state['encoder'])
            self.decoder.load_state_dict(state['decoder'])

    def get_opt_params(self):
        return self.opt_params

    def forward(self, images, init_features, captions, lengths, persist_features):
        features = self.encoder(images, init_features)
        outputs = self.decoder(features, captions, lengths, images, persist_features)
        return outputs

    def sample(self, image_tensor, init_features, persist_features, states=None,
               max_seq_length=20):
        feature = self.encoder(image_tensor, init_features)
        sampled_ids = self.decoder.sample(feature, image_tensor, persist_features, states,
                                          max_seq_length=max_seq_length)

        return sampled_ids
