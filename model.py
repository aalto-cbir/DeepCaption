import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from collections import OrderedDict

class ModelParams:
    def __init__(self, d):
        self.embed_size = self._get_param(d, 'embed_size', 256)
        self.hidden_size = self._get_param(d, 'hidden_size', 512)
        self.num_layers = self._get_param(d, 'num_layers', 1)
        self.batch_size = self._get_param(d, 'batch_size', 128)
        self.dropout = self._get_param(d, 'dropout', 0)
        self.learning_rate = self._get_param(d, 'learning_rate', 0.001)
        features = self._get_param(d, 'features', 'resnet152').split(',')

        self.external_features = []
        self.internal_features = []
        for fn in features:
            if fn.endswith('.h5'):
                self.external_features.append(fn)
            else:
                self.internal_features.append(fn)

        # this needs to be loaded externally ...
        self.external_features_total_dim = 0

                
    @classmethod
    def fromargs(cls, args):
        return cls(vars(args))

    def _get_param(self, d, param, default):
        if param not in d:
            print('WARNING: {} not set, using default value {}'.
                  format(param, default))
            return default
        return d[param]


class FeatureExtractor(nn.Module):
    def __init__(self, model_name, debug=False):
        """Load the pretrained model and replace top fc layer."""
        super(FeatureExtractor, self).__init__()

        if model_name == 'alexnet':
            if debug:
                print('Using AlexNet, features shape 256 x 6 x 6')
            model = models.alexnet(pretrained=True)
            self.output_dim = 256*6*6
        elif model_name == 'densenet201':
            if debug:
                print('Using DenseNet 201, features shape 1920 x 7 x 7')
            model = models.densenet201(pretrained=True)
            self.output_dim = 1920*7*7
        elif model_name == 'resnet152':
            if debug:
                print('Using resnet 152, features shape 2048')
            model = models.resnet152(pretrained=True)
            self.output_dim = 2048
        else:
            raise ValueError('Unknown model name: {}'.format(model_name))

        modules = list(model.children())[:-1]
        self.extractor = nn.Sequential(*modules)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.extractor(images)
        return features.reshape(features.size(0), -1)


class EncoderCNN(nn.Module):
    def __init__(self, p):
        """Load a pretrained CNN and replace top fc layer."""
        super(EncoderCNN, self).__init__()

        # We keep track of the sum of the dimensionalities of the
        # concatenated features
        total_output_dim = p.external_features_total_dim

        # Construct the "internal" feature extractors, i.e., those
        # that use (pretrained) pytorch models
        self.extractors = nn.ModuleList()
        for fn in p.internal_features:
            extractor = FeatureExtractor(fn)
            self.extractors.append(extractor)
            total_output_dim += extractor.output_dim

        # Add FC layer on top of features to get the desired output dimension
        self.linear = nn.Linear(total_output_dim, p.embed_size)
        self.bn = nn.BatchNorm1d(p.embed_size, momentum=0.01)
        
    def forward(self, images, external_feature_batches):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            feat_outputs = []
            # Extract features with each extractor
            for extractor in self.extractors:
                feat_outputs.append(extractor(images))
            for ext_feat in external_feature_batches:
                feat_outputs.append(ext_feat)
            # Concatenate features
            features = torch.cat(feat_outputs, 1)
        # Apply FC layer and batch normalization
        features = self.bn(self.linear(features))
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
    def __init__(self, p, vocab_size, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, p.embed_size)
        self.lstm = nn.LSTM(p.embed_size, p.hidden_size, p.num_layers,
                            dropout=p.dropout, batch_first=True)
        self.linear = nn.Linear(p.hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids
