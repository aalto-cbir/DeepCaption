import torch

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import models


class ModelParams:
    def __init__(self, d):
        self.embed_size = self._get_param(d, 'embed_size', 256)
        self.hidden_size = self._get_param(d, 'hidden_size', 512)
        self.num_layers = self._get_param(d, 'num_layers', 1)
        self.input_size = self._get_param(d, 'input_size', 5 * self.embed_size)
        self.dropout = self._get_param(d, 'dropout', 0)

    @classmethod
    def fromargs(cls, args):
        return cls(vars(args))

    @staticmethod
    def _get_param(d, param, default):
        if param not in d:
            print('WARNING: {} not set, using default value {}'.
                  format(param, default))
            return default
        return d[param]


class EncoderCNN(nn.Module):
    def __init__(self, p):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, p.embed_size)
        self.bn = nn.BatchNorm1d(p.embed_size, momentum=0.01)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class EncoderRNN(nn.Module):
    def __init__(self, p):
        """
        initialize lstm cell for reading image sequences
        :param p: model parameters
        """
        super(EncoderRNN, self).__init__()
        self.lstm_cell = nn.LSTM(input_size=p.input_size,
                                 hidden_size=p.embed_size,
                                 num_layers=p.num_layers,
                                 batch_first=True)

    def forward(self, sequence_features):
        """
        read image sequence features and output context vector
        :param sequence_features: image features extracted using a pre-trained model
        :return: last cell hidden state as sequence-context-vector
        """
        sequence_features = sequence_features.unsqueeze(0)
        sequence_features = sequence_features.view(1, 1, -1)
        # print('shape of input to EncoderRNN: ', sequence_features.shape)

        hiddens, h_n = self.lstm_cell(sequence_features)
        # print('shape of context vector: ', hiddens[-1].shape)
        return hiddens[-1]


class DecoderRNN(nn.Module):
    def __init__(self, p, vocab_size, max_seq_length=100):
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
        # print('shape of embeddings: ', embeddings.shape)
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
            hiddens, states = self.lstm(inputs, states)  # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))  # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)  # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)  # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)  # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)  # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids
