import os

import torch
import torch.nn as nn
import torchvision.models as models

import numpy as np

from collections import OrderedDict, namedtuple
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import external_models as ext_models

Features = namedtuple('Features', 'external, internal')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ModelParams:
    def __init__(self, d, arg_params=None):
        """Store parameters given e.g. on the command line when invoking
        the trainining script"""
        # Allow changing model from non-hierarchical to hierarchical when loading
        # from non-hierarchical model:
        if arg_params is not None and arg_params.hierarchical_model is not None:
            self.hierarchical_model = arg_params.hierarchical_model
        else:
            self.hierarchical_model = self._get_param(d, 'hierarchical_model', False)
        self.embed_size = self._get_param(d, 'embed_size', 256)
        # Default pooling size to embed_size for non-hierarchical models:
        self.pooling_size = self.embed_size
        self.hidden_size = self._get_param(d, 'hidden_size', 512)
        self.num_layers = self._get_param(d, 'num_layers', 1)
        self.batch_size = self._get_param(d, 'batch_size', 128)
        self.dropout = self._get_param(d, 'dropout', 0)
        self.learning_rate = self._get_param(d, 'learning_rate', 0.001)
        self.features = self._get_features(d, 'features', 'resnet152')
        self.persist_features = self._get_features(d, 'persist_features', '')
        self.encoder_dropout = self._get_param(d, 'encoder_dropout', 0)
        # Type of attention mechanism in use:
        self.attention = self._get_param(d, 'attention', None)
        # Vocab object stored in the model:
        self.vocab = self._get_param(d, 'vocab', None)
        # Boolean toggle of whether the model is trained to generate <start> token
        self.skip_start_token = self._get_param(d, 'skip_start_token', False)
        # Setting for initializing the hidden unit and cell state of the RNN:
        self.rnn_hidden_init = self._get_param(d, 'rnn_hidden_init', None)
        # Use the same embedding matrix for input and output word embeddings:
        self.share_embedding_weights = self._get_param(d, 'share_embedding_weights', False)

        # Below parameters used only by the Hierarchical model:
        if self.hierarchical_model:
            self.pooling_size = self._get_param(d, 'pooling_size', 1024)
            self.max_sentences = self._get_param(d, 'max_sentences', 6.0)
            self.weight_sentence_loss = self._get_param(d, 'weight_sentence_loss', 5.0)
            self.weight_word_loss = self._get_param(d, 'weight_word_loss', 1.0)
            self.dropout_stopping = self._get_param(d, 'dropout_stopping', 0)
            self.dropout_fc = self._get_param(d, 'dropout_fc', 0)
            self.fc_size = self._get_param(d, 'fc_size', self.pooling_size)
            # Coherent caption parameters:
            self.coherent_sentences = self._get_param(d, 'coherent_sentences', False)
            self.coupling_alpha = self._get_param(d, 'coupling_alpha', 1.0)
            self.coupling_beta = self._get_param(d, 'coupling_beta', 1.5)

    @classmethod
    def fromargs(cls, args):
        """Initialize ModelParams class from command line arguments 'args'"""
        return cls(vars(args))

    def _get_param(self, d, param, default):
        if param not in d or d[param] is None:
            # print('WARNING: {} not set, using default value {}'.
            #       format(param, default))
            return default
        return d[param]

    def _get_features(self, d, param, default):
        """Either loads a comma-separated list of feature names from command line, and stores
        features names in a namedtuple called Features, or returns itself if feature is already
        said namedtuple"""
        p = self._get_param(d, param, default)

        # If it's already of type Features, just return it
        if hasattr(p, 'internal'):
            return p

        features = p.split(',') if p else []

        ext_feat = []
        int_feat = []
        for fn in features:
            # Check if feature has an extension, if yes assume it's
            # an external feature contained in a file with extension '.$ext':
            (tmp, ext) = os.path.splitext(fn)
            if ext:
                ext_feat.append(fn)
            else:
                int_feat.append(fn)

        # Returns a named tuple containing two lists with feature names,
        # one list for external and one for internal features:
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
        if type(ef) is str:
            ef = ef.split(',')
        return features._replace(external=ef)

    def __str__(self):
        return '\n'.join(['[ModelParams] {}={}'.format(k, v) for k, v in
                          self.__dict__.items()])


class FeatureExtractor(nn.Module):
    def __init__(self, model_name, debug=False, finetune=False):
        """Load the pretrained model and replace top fc layer.
        Inception assumes input image size to be 299x299.
        Other models assume input image of size 224x224
        More info: https://pytorch.org/docs/stable/torchvision/models.html """
        super(FeatureExtractor, self).__init__()

        # Set flatten to False if we do not want to flatten the output features
        self.flatten = True

        # Toggle finetuning
        self.finetune = finetune

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
        if self.finetune:
            features = self.extractor(images)
        else:
            with torch.no_grad():
                self.extractor.eval()
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

        self.total_feat_dim = total_feat_dim

        # If features are used to initialize the RNN hidden and cell states
        # then we output different feature dimension, handled by RNN decoder
        self.rnn_hidden_init = p.rnn_hidden_init

        # The following lines are needed only if the image features are not
        # used to initialized the Decoder hidden state:

        # Add FC layer on top of features to get the desired output dimension
        self.linear = nn.Linear(total_feat_dim, p.pooling_size)
        self.dropout = nn.Dropout(p=p.encoder_dropout)
        self.bn = nn.BatchNorm1d(p.pooling_size, momentum=0.01)

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

        # Apply transformations to the raw image features in case we are not
        # initializing the RNN hidden and state vectors from features.
        if self.rnn_hidden_init is None:
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


def init_hidden_state(model, features):
    """Initialize the initial hidden and cell state of the LSTM
    :param features input features that should initialize RNN hidden state h
                    and cell state c
    :param init_h, init_c linear tranforms from features to h and c"""
    if features.dim() > 2:
        _features = features.mean(dim=1)
    else:
        _features = features

    h = None
    c = None

    # Stack hidden and cell states for all RNN layers
    for _init_h, _init_c in zip(model.init_h, model.init_c):
        if h is None and c is None:
            h = _init_h(_features).unsqueeze(0)
            c = _init_c(_features).unsqueeze(0)
        else:
            h_new = _init_h(_features).unsqueeze(0)
            c_new = _init_c(_features).unsqueeze(0)

            h = torch.stack([h, h_new], dim=0).squeeze(1)
            c = torch.stack([c, c_new], dim=0).squeeze(1)

    if h.dim() == 3 and h.size()[0] == 1:
        h.squeeze_(0)
        c.squeeze_(0)

    return h, c


class DecoderRNN(nn.Module):
    def __init__(self, p, vocab_size, ext_features_dim=0, enc_features_dim=0):
        """Set the hyper-parameters and build the layers.
        :param p model parameters class
        :param vocab_size size of training vocabulary
        :param ext_features_dim size of external features used as persistant
                                features at each time-step
        :param enc_features_dim complete size of initial (non-persistant) features"""
        super(DecoderRNN, self).__init__()

        self.embed = nn.Embedding(vocab_size, p.embed_size)

        (self.extractors,
         int_features_dim) = FeatureExtractor.list(p.persist_features.internal)
        # Sum of the dimensionalities of the concatenated features
        total_feat_dim = ext_features_dim + int_features_dim

        print('DecoderCNN: total feature dim={}'.format(total_feat_dim))

        self.skip_start_token = p.skip_start_token
        self.rnn_hidden_init = p.rnn_hidden_init
        self.share_embedding_weights = p.share_embedding_weights

        if self.rnn_hidden_init == 'from_features':
            assert enc_features_dim is not 0
            self.init_h = nn.ModuleList()
            self.init_c = nn.ModuleList()
            for _layer in range(p.num_layers):
                self.init_h.append(nn.Linear(enc_features_dim, p.hidden_size))
                self.init_c.append(nn.Linear(enc_features_dim, p.hidden_size))

        self.lstm = nn.LSTM(p.embed_size + total_feat_dim, p.hidden_size,
                            p.num_layers, dropout=p.dropout, batch_first=True)

        if self.share_embedding_weights:
            print("DecoderRNN: Sharing input and output embeddings for the RNN...")
            # Using the Output Embedding to Improve Language Models
            # https://arxiv.org/abs/1608.05859
            self.dropout_embedding = nn.Dropout(p=p.dropout)
            self.projection = nn.Linear(p.hidden_size, p.hidden_size)
            self.hidden_to_embeddings = nn.Linear(p.hidden_size, p.embed_size)
            self.embed_output = nn.Linear(p.embed_size, vocab_size)
            self.embed_output.weight.data = self.embed.weight.data.transpose(1, 0)
        else:
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

    def forward(self, features, captions, lengths, images, external_features=None,
                teacher_p=1.0, teacher_forcing='always', output_hiddens=False):
        """Decode image feature vectors and generates captions.
        :param output_hiddens - output last hidden layer of the RNN
        """

        # First, construct embeddings input, with initial feature as
        # the first: (batch_size, 1 + longest caption length, embed_size)
        embeddings = self.embed(captions)
        seq_length = embeddings.size()[1]

        if self.rnn_hidden_init == 'from_features':
            states = init_hidden_state(self, features)
        else:
            states = None
            embeddings = torch.cat([features.unsqueeze(1), embeddings], 1)

        with torch.no_grad():
            persist_features = self._cat_features(images, external_features)
            if persist_features is None:
                persist_features = features.new_empty(0)
            else:
                # Get into shape: batch_size, seq_length, embed_size
                persist_features = (persist_features.unsqueeze(1).
                                    expand(-1, seq_length, -1))

        hiddens = None

        if teacher_forcing == 'always':
            # Teacher forcing enabled -
            # Feed ground truth as next input at each time-step when training:
            inputs = torch.cat([embeddings, persist_features], 2)
            packed = pack_padded_sequence(inputs, lengths, batch_first=True)
            hiddens, _ = self.lstm(packed, states)
            if self.share_embedding_weights:
                hiddens_p = self.projection(hiddens[0])
                output_embeddings = self.hidden_to_embeddings(hiddens_p)
                #outputs = self.embed_output(output_embeddings.t())
                outputs = torch.matmul(output_embeddings, self.embed_output.weight)
            else:
                outputs = self.linear(hiddens[0])
        else:
            # Use sampled or additive scheduling mode:
            batch_size = features.size()[0]
            vocab_size = self.linear.out_features
            outputs = torch.zeros(batch_size, seq_length, vocab_size).to(device)
            inputs = torch.cat([features, persist_features], 1).unsqueeze(1)

            for t in range(seq_length - 1):
                hiddens, states = self.lstm(inputs, states)
                step_output = self.linear(hiddens.squeeze(1))
                outputs[:, t, :] = step_output

                if teacher_forcing == 'sampled':
                    # Sampled mode: sample next token from lstm with probability
                    # (1 - prob_teacher):
                    if float(torch.rand(1)) < teacher_p:
                        embed_t = embeddings[:, t + 1]
                    else:
                        _, predicted = step_output.max(1)
                        embed_t = self.embed(predicted)
                elif teacher_forcing == 'additive':
                    # Additive mode: add embeddings using weights determined by
                    # sampling schedule:

                    teacher_p = torch.tensor([teacher_p]).to(device)

                    # Embedding of the next token from the ground truth:
                    embed_gt_t = embeddings[:, t + 1]

                    _, predicted = step_output.max(1)
                    # Embedding of the next token sampled from the model:
                    embed_sampled_t = self.embed(predicted)

                    # Weighted sum of the above embeddings:
                    embed_t = teacher_p * embed_gt_t + (1 - teacher_p) * embed_sampled_t
                elif teacher_forcing == 'additive_sampled':
                    # If we are in teacher forcing use ground truth as input
                    if float(torch.rand(1)) < teacher_p:
                        embed_t = embeddings[:, t + 1]
                    # Otherwise use additive input
                    else:
                        teacher_p = torch.tensor([teacher_p]).to(device)

                        # Embedding of the next token from the ground truth:
                        embed_gt_t = embeddings[:, t + 1]

                        _, predicted = step_output.max(1)
                        # Embedding of the next token sampled from the model:
                        embed_sampled_t = self.embed(predicted)

                        # Weighted sum of the above embeddings:
                        embed_t = teacher_p * embed_gt_t + (1 - teacher_p) * embed_sampled_t
                else:
                    # Invalid teacher forcing mode specified
                    return None

                inputs = torch.cat([embed_t, persist_features], 1).unsqueeze(1)

            # Generate a packed sequence of outputs with generated captions assuming
            # exactly the same lengths are ground-truth. If needed, model could be modified
            # to check for the <end> token (by for-example hardcoding it to same value
            # for all models):
            outputs = pack_padded_sequence(outputs, lengths, batch_first=True)[0]

        # Some variants of hierarchical model require final RNN hidden layer value:
        if output_hiddens:
            return (outputs, hiddens)
        else:
            return outputs

    def sample(self, features, images, external_features, states=None,
               max_seq_length=20, start_token_id=None, end_token_id=None,
               output_hiddens=False):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []

        batch_size = len(images)

        # Concatenate internal and external features
        persist_features = self._cat_features(images, external_features)
        if persist_features is None:
            persist_features = features.new_empty(0)

        if output_hiddens:
            all_hiddens = torch.zeros(batch_size,
                                      max_seq_length, self.lstm.hidden_size).to(device)

        # Initialize the rnn from input features, instead of setting the hidden
        # state to zeros (as done by default):
        if self.rnn_hidden_init == 'from_features':
            states = init_hidden_state(self, features)

            assert start_token_id is not None

            # Start generating the sentence by first feeding in the start token:
            start_token_embedding = self.embed(torch.tensor(start_token_id).to(device))
            embed_size = len(start_token_embedding)
            start_token_embedding = start_token_embedding.unsqueeze(0).expand(batch_size,
                                                                              embed_size)
            inputs = torch.cat([start_token_embedding, persist_features], 1).unsqueeze(1)
        else:
            # inputs: (batch_size, 1, embed_size + len(external features))
            inputs = torch.cat([features, persist_features], 1).unsqueeze(1)

        for i in range(max_seq_length):
            hiddens, states = self.lstm(inputs, states)

            # If we are sharing embedding weights before and after the RNN processing:
            if self.share_embedding_weights:
                hiddens_p = self.projection(hiddens.squeeze(1))
                output_embeddings = self.hidden_to_embeddings(hiddens_p)
                outputs = torch.matmul(output_embeddings, self.embed_output.weight)
            else:
                outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)

            # inputs: (batch_size, 1, embed_size + len(external_features))
            embeddings = self.embed(predicted)
            inputs = torch.cat([embeddings, persist_features], 1).unsqueeze(1)

            if output_hiddens:
                all_hiddens[:, i] = hiddens.squeeze(1)

        # sampled_ids: (batch_size, max_seq_length)
        sampled_ids = torch.stack(sampled_ids, 1)

        if output_hiddens:
            return sampled_ids, all_hiddens,
        else:
            return sampled_ids


class EncoderDecoder(nn.Module):
    def __init__(self, params, device, vocab_size, state, ef_dims, lr_dict={}):
        """Vanilla EncoderDecoder model
        :param params_lr - optional parameter specifying a dict of parameter groups for which
        a special learning rate needs to be applied"""
        super(EncoderDecoder, self).__init__()
        print('Using device: {}'.format(device.type))
        print('Initializing EncoderDecoder model...')

        if params.hierarchical_model:
            self.model_type = 'hierarchical_model'
            _DecoderRNN = HierarchicalDecoderRNN
        else:
            self.model_type = 'regular_model'
            _DecoderRNN = DecoderRNN

        self.encoder = EncoderCNN(params, ef_dims[0]).to(device)
        self.decoder = _DecoderRNN(params, vocab_size, ef_dims[1],
                                   self.encoder.total_feat_dim).to(device)

        encoder_params = (list(self.encoder.linear.parameters()) +
                          list(self.encoder.bn.parameters()))

        # Use separate parameter groups for hierarchical model for more control
        # over the learning rate.
        if params.hierarchical_model:
            if lr_dict.get('word_decoder'):
                lr = lr_dict.get('word_decoder')
            else:
                lr = None

            dec_params = self.decoder.named_parameters()
            sent_params = [v for k, v in dec_params if not k.startswith('word_decoder')]
            word_params = list(self.decoder.word_decoder.parameters())
            decoder_params = [{'params': sent_params},
                              {'params': word_params, 'lr': lr, 'name': 'word_decoder'}]

            encoder_params = [{'params': encoder_params}]
        else:
            decoder_params = list(self.decoder.parameters())

        self.opt_params = decoder_params + encoder_params

        if state:
            # If the user has specified that they would like train
            # a hierarchical model and are also specifying an old model to be loaded
            # then selectively attempt to use the decoder of the loaded model
            # as the word-RNN of the hierarchical model:
            if params.hierarchical_model and not state.get('hierarchical_model'):
                print("Applying non-hierarchical decoder weights to hierarchical model...")
                self.decoder.word_decoder.load_state_dict(state['decoder'])
            else:
                self.encoder.load_state_dict(state['encoder'])
                self.decoder.load_state_dict(state['decoder'])

    def get_opt_params(self):
        return self.opt_params

    def forward(self, images, init_features, captions, lengths, persist_features,
                teacher_p=1.0, teacher_forcing='always', sorting_order=None,
                writer_data=None):
        features = self.encoder(images, init_features)
        if self.model_type == 'hierarchical_model':
            # TODO: Make the hierarchical and regular decoder take the same arguments
            # if possible:
            outputs = self.decoder(features, captions, lengths, images, sorting_order,
                                   writer_data=writer_data)
        else:
            outputs = self.decoder(features, captions, lengths, images, persist_features,
                                   teacher_p, teacher_forcing)
        return outputs

    def sample(self, image_tensor, init_features, persist_features, states=None,
               max_seq_length=20, start_token_id=None, end_token_id=None):
        feature = self.encoder(image_tensor, init_features)
        sampled_ids = self.decoder.sample(feature, image_tensor, persist_features, states,
                                          max_seq_length=max_seq_length,
                                          start_token_id=start_token_id,
                                          end_token_id=end_token_id)

        return sampled_ids


class SharedEmbeddingXentropyLoss(nn.Module):
    def __init__(self, param_lambda=0.15):
        super(SharedEmbeddingXentropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.param_lambda = param_lambda

    def forward(self, P, outputs, targets):
        """:param P - projection matrix of size (hidden_size x hidden_size)"""
        reg_term = self.param_lambda * P.norm()

        return self.loss(outputs, targets) + reg_term


class HierarchicalDecoderRNN(nn.Module):
    def __init__(self, p, vocab_size, ext_features_dim=0, enc_features_dim=0):
        """Set the hyper-parameters and build the layers."""
        super(HierarchicalDecoderRNN, self).__init__()
        self.sentence_lstm = nn.LSTM(p.pooling_size, p.hidden_size,
                                     dropout=p.dropout, batch_first=True)

        self.coherent_sentences = p.coherent_sentences
        if self.coherent_sentences:
            self.coupling_unit = HierarchicalCoupling(p.hidden_size,
                                                      p.embed_size,
                                                      p.coupling_alpha,
                                                      p.coupling_beta)

        # Stopping state classifier
        self.dropout_stopping = nn.Dropout(p=p.dropout_stopping)
        self.linear_stopping = nn.Linear(p.hidden_size, 2)
        # Two fully connected layers (assuming same dimension):
        self.linear1 = nn.Linear(p.hidden_size, p.fc_size)
        self.dropout_fc = nn.Dropout(p=p.dropout_fc)
        self.linear2 = nn.Linear(p.fc_size, p.embed_size)
        self.non_lin = nn.SELU()
        # Word LSTM:
        self.word_decoder = DecoderRNN(p, vocab_size)

        self.max_sentences = p.max_sentences

    def _get_global_topic(self, topics):
        topic_vector_norms = torch.norm(topics, dim=2)
        topic_sums = torch.sum(topic_vector_norms, dim=1)
        topic_sums = topic_sums.unsqueeze(1).expand(-1, self.max_sentences)
        # alphas denote topic weights:
        alphas = topic_vector_norms / topic_sums
        alphas = alphas.unsqueeze(-1)

        G = torch.sum(alphas * topics, dim=1)

        return G

    def forward(self, features, captions, lengths, images, sorting_order,
                use_teacher_forcing=True, writer_data=None):
        """Decode image feature vectors and generates captions.
        features: image features
        captions: paragraph captions, regular captions treated as paragraphs
        of length = 1
        lengths: of sentences in paragraph"""

        # Repeat features so that each time-step of sentence LSTM receives the same
        # feature vector as its input:

        # Setup logging objects and allow certain important values
        # to be logged. Usually this is triggered once per epoch by calling
        # side i.e train.py setting writer_data to be non-zero dict 
        # containing the writer object and current epoch number:
        log_values = False
        if writer_data is not None:
            _writer = writer_data['writer']
            _epoch = writer_data['epoch']
            log_values = True


        batch_size = lengths.size()[0]
        n_sentences = lengths.size()[1]

        features_repeated = features.unsqueeze(1).expand(-1, n_sentences, -1)
        hiddens, _ = self.sentence_lstm(features_repeated)

        # Dims: (batch_size X max_sentences X 2)
        sentence_stopping = torch.zeros(batch_size, n_sentences, 2).to(device)
        # sentence_stopping = torch.zeros(lengths.size()[0], ).to(device)

        unsorting_order = torch.zeros(batch_size, n_sentences, 1).long().to(device)
        unsorted_non_zero_idxs = torch.zeros(batch_size,
                                             n_sentences, 1).byte().to(device)

        topics = torch.zeros(batch_size, n_sentences, self.linear2.out_features).to(device)

        # We will use hiddens for logistic regression over stopping state
        # and fc2 as the context for the word RNN
        for t in range(n_sentences):
            if lengths[0, t] == 0:
                break   # no more sentences at position >= t in current minibatch

            # Get the unsurted hidden layer outputs from the SentenceRNN
            h_t = hiddens[:, t]

            # Store the hidden state output for {CONTINUE = 0, STOP = 1} classifier
            # NOTE: For this step, we do no not need to sort sentences at each position t
            sentence_stopping[:, t] = self.dropout_stopping(nn.Sigmoid()(
                self.linear_stopping(h_t)).squeeze())

            # Sort and filter a minibatch of hidden layer t based on:
            # 1) Image order (as defined in data loader based on caption lengths)
            # 2) Whether a sentence caption for image at sentence position t
            # is not-empty

            # Sort mini-batch hidden states based on current sentence position and
            # get rid of zero-length sentences (note that the lengths are already sorted)
            # NOTE: lengths[:, t] is already sorted according to the sorting order defined
            # in sorting_order[t]
            non_zero_idxs = lengths[:, t] > 0
            #h_t = h_t[sorting_order[t]][non_zero_idxs]

            # Fully connected layer 1 with ReLU activation:
            fc1 = self.non_lin(self.linear1(h_t))
            # Output from the following layer is our context vector:
            fc2 = self.dropout_fc(self.linear2(fc1))

            # Some sentences at position t will be zero length and will have no topic vector,
            # we need to account for that:
            # num_results = fc2.size()[0]

            # Create unsorting indices:
            unsorting_order[:, t].scatter_(0,
                                           sorting_order[:, t].unsqueeze(1).to(device),
                                           sorting_order[:, 0].unsqueeze(1).to(device))

            unsorted_non_zero_idxs[:, t].scatter_(0,
                                                  sorting_order[:, t].unsqueeze(1).to(device),
                                                  non_zero_idxs.unsqueeze(1).to(device))

            # Keep topics unsorted, sort them on demand at latter stages:
            #topics[:, t][unsorted_non_zero_idxs] = fc2[unsorting_order[unsorted_non_zero_idxs]]
            topics[:, t] = fc2

            if log_values:
                _writer.add_histogram('values/topics_pre_coherence' + str(t),
                                      topics[:, t].clone().cpu().detach().numpy(),
                                      _epoch)

        word_rnn_out = []

        # Use WordRNN hidden layer value to initialize the next topic vector (coherent mode):
        if self.coherent_sentences:
            # Need to sort topics back to get the right G!
            G = self._get_global_topic(topics)

            if log_values:
                _writer.add_histogram('values/global_topic_G',
                                      G.clone().cpu().detach().numpy(),
                                      _epoch)

            # Hidden layer output of WordRNN for previous sentence, initialize to None
            # for the first sentence:
            hiddens_w = None

            for t in range(n_sentences):
                non_zero_idxs = lengths[:, t] > 0
                topic = topics[:, t][sorting_order[:, t]][non_zero_idxs]

                if hiddens_w is not None:
                    # Note: pad_packed_sequence() returns a tuple where first element is
                    # a tensor containing padded sentences and second element is an array of
                    # last element with actual value for each sequence
                    _hiddens_w, _seq_lengths = pad_packed_sequence(hiddens_w, batch_first=True)

                    # Convert lengths to indices:
                    seq_end_idxs = (_seq_lengths - 1).to(device)

                    # Select hidden layer values at end of sequence indices for each sequence
                    # TODO? rewrite below line by merging 2 first dimensions of the _hiddens
                    # and using a cumulative sume in seq_end_idxs to index over that!
                    hiddens_w = torch.stack(
                        [_hiddens_w[i, j] for i, j in enumerate(seq_end_idxs)])

                    # Next lines ensure that hiddens_w are always sorted relative to the order
                    # defined in t=0, and not relative to the order defined in previous
                    # time step t=t-1

                    # Do magic -
                    # 1) Sort from order defined for step t=t-1 to order defined in step t=0,
                    # 2) Sort from order defined for t=0 to order defined for t=t
                    # 3) Remove elements that do not have corresponding sentences at t=t
                    resort_idxs = unsorting_order[:, t - 1][sorting_order[:, t]][non_zero_idxs]

                    # Hiddens_w need to be sorted according to the sorting order defined for
                    # SentenceRNN output at position t, not position t-1, hence antics above
                    hiddens_w = hiddens_w[resort_idxs]

                topic = self.coupling_unit(hiddens_w,
                                           G[sorting_order[:, t]][non_zero_idxs],
                                           topic)

                if log_values:
                    _writer.add_histogram('values/topics_post_coherence' + str(t),
                                          topic.clone().cpu().detach().numpy(),
                                          _epoch)
                    if hiddens_w is not None:
                        _writer.add_histogram('values/hiddens_w' + str(t),
                                              hiddens_w.clone().cpu().detach().numpy(),
                                              _epoch)

                output, hiddens_w = self.word_decoder(topic, captions[:, t][non_zero_idxs],
                                                      lengths[:, t][non_zero_idxs],
                                                      images[sorting_order[:, t]],
                                                      output_hiddens=True)
                word_rnn_out.append(output)
        # Otherwise do the "regular" hierarchical model (Krause et al 2017):
        else:
            for t in range(n_sentences):
                non_zero_idxs = lengths[:, t] > 0
                topic = topics[:, t][sorting_order[:, t]][non_zero_idxs]
                word_rnn_out.append(self.word_decoder(topic, captions[:, t][non_zero_idxs],
                                                      lengths[:, t][non_zero_idxs],
                                                      images[sorting_order[:, t]]))

        return sentence_stopping, word_rnn_out

    def sample(self, features, images, external_features, states=None,
               max_seq_length=50, start_token_id=None, end_token_id=None):
        """Generate captions for given image features using greedy search."""

        inputs = features.unsqueeze(1)
        batch_size = inputs.size()[0]
        # Output tensor:
        paragraphs = torch.zeros(batch_size, self.max_sentences,
                                 max_seq_length).long().to(device)

        # Mini batch indices where the next sentence should be generated
        # when mask element is set to zero, it means that more sentences should be
        # generated for input image at that index
        masks = torch.ones(batch_size, self.max_sentences).byte().to(device)

        # Tensor storing raw hidden layer output from Sentence RNN:
        hiddens_s = torch.zeros(batch_size, self.max_sentences, 1,
                                self.sentence_lstm.hidden_size).to(device)

        # Get stopping indicators for sentences each position t:
        for t in range(self.max_sentences):
            # Output a sentence state
            # hiddens: (batch_size, 1, hidden_size)
            hiddens_s[:, t], states = self.sentence_lstm(inputs, states)
            # (1 x 2)
            stopping = nn.Sigmoid()(self.linear_stopping(hiddens_s[:, t].squeeze(1)))

            # If stopping[:, 0] > stopping[:, 1] => CONTINUE
            # If stopping[:, 0] <= stopping[:, 1] => STOP
            if t + 1 < self.max_sentences:
                masks[:, t + 1] = (
                    masks[:, t] & (stopping[:, 0] > stopping[:, 1]))

        # Tensor storing sentence topics which are used as input for Word RNN:
        topics = torch.zeros(batch_size,
                             self.max_sentences, self.linear2.out_features).to(device)

        # Generate sentence topics:
        hiddens_s = hiddens_s.squeeze(2)
        for t in range(self.max_sentences):
            fc = self.non_lin(self.linear1(hiddens_s[:, t]))
            topics[:, t] = self.linear2(fc).squeeze(1)

        if self.coherent_sentences:
            # hiddens_w stores the hidden state at the last token in the
            # preceding sentence. In this case it corresponds to either the
            # end of line character ('.' == <end>) or last word in the sequence,
            # if sequence is truncated at max_seq_length.
            hiddens_w = None
            G = self._get_global_topic(topics)

        # Generate sentences from topics
        for t in range(self.max_sentences):
            topic = topics[:, t]
            if self.coherent_sentences:
                topic = self.coupling_unit(hiddens_w, G, topic)
                sentence, hiddens_w = self.word_decoder.sample(topic, images,
                                                               external_features,
                                                               max_seq_length=max_seq_length,
                                                               output_hiddens=True)

                # TODO: Get the right hiddens_w from the previous crop of outputs
                # For each sentence, get the index of the token corresponding to end_token_idx
                assert end_token_id is not None

                # Index the end of sentence token for each sentence.
                # If end of sentence token not found, index is set to last column
                # of the "sentence" tensor.
                _, eos_indices = torch.max((sentence == end_token_id).t(), 0)

                hiddens_w = torch.stack([hiddens_w[i, j] for i, j in enumerate(eos_indices)])
            else:
                sentence = self.word_decoder.sample(topic, images, external_features,
                                                    max_seq_length=max_seq_length)

            mask = masks[:, t]
            paragraphs[:, t][mask] = sentence[mask]

        return paragraphs


class HierarchicalCoupling(nn.Module):
    def __init__(self, hidden_size, embed_size, alpha, beta):
        super(HierarchicalCoupling, self).__init__()
        self.linear1 = nn.Linear(hidden_size, embed_size)
        self.linear2 = nn.Linear(embed_size, embed_size)
        self.non_lin = nn.SELU()
        self.gate = nn.GRU(embed_size, embed_size, 1, batch_first=True)
        self.alpha = alpha
        self.beta = beta

    def forward(self, hiddens, G, T):
        if hiddens is not None:
            x = self.non_lin(self.linear1(hiddens.squeeze(1)))
            C = self.linear2(x)
        else:
            C = 0

        T_fused_C = (self.alpha * T + self.beta * C) / (self.alpha + self.beta)
        # RNN wants inputs of dim (bs x input_size x n_layers_in_rnn):
        T_prime = self.gate(T_fused_C.unsqueeze(1), G.unsqueeze(0))

        return T_prime[0].squeeze(1)


class HierarchicalXEntropyLoss(nn.Module):
    def __init__(self, weight_sentence_loss=5.0, weight_word_loss=1.0):
        super(HierarchicalXEntropyLoss, self).__init__()
        self.weight_sent = torch.Tensor([weight_sentence_loss]).to(device)
        self.weight_word = torch.Tensor([weight_word_loss]).to(device)
        self.sent_loss = nn.CrossEntropyLoss()
        self.word_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        """ outputs and targets are both tuples of length = 2
        First element: a Tensor of size (batchsize X max_sentences)
            it tells us the sentence RNN stopping indices
        Second element:  a list of length=batchsize, of Tensors of
            size at most = max_sentence the actual captions for each sentence position"""
        # Compare number of sentences?
        # outputs: tuple = (captions, stopping_states)
        # targets: target paragraphs
        # Do regular loss on individual sentences?
        # 1) Cross entropy loss of current sentence being last sentence
        # 2) Cross entropy loss of curent word being correct
        # For each sample in a mini-batch we want the number of sentences
        # Max number of sentences in the target mini-batch:
        max_sentences = len(targets[1])
        # print('max_sentences: {}'.format(max_sentences))

        self.loss_s = torch.Tensor([0]).to(device)

        for j in range(max_sentences):
            # print("Size of outputs[0][{}]: {}, size of targets[0][{}] {}".
            #      format(j, outputs[0][j].size(), j, targets[0][j].size()))
            self.loss_s += self.sent_loss(outputs[0][:, j], targets[0][:, j])

        self.loss_w = torch.Tensor([0]).to(device)

        for j in range(max_sentences):
            # print("Size of outputs[1][{}]: {}, size of targets[1][{}] {}".
            #      format(j, outputs[1][j].size(), j, targets[1][j].size()))
            self.loss_w += self.word_loss(outputs[1][j], targets[1][j])

        return self.weight_sent * self.loss_s + self.weight_word * self.loss_w

    def item_terms(self):
        return self.weight_sent, self.loss_s, self.weight_word, self.loss_w


class SpatialAttention(nn.Module):
    """Spatial attention network implementation based on
    https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning and
    "Knowing when to look" by Lu et al"""

    def __init__(self, feature_size, num_attention_locs, hidden_size, lr_dict={}):
        """Initialize a network that learns the attention weights for each time step
        feature_size - size C of a single 1x1xC tensor at each image location
        num_attention_locs - number of feature vectors in one image
        hidden_size - size of the hidden layer of decoder LSTM"""

        super(SpatialAttention, self).__init__()

        self.image_att = nn.Linear(feature_size, hidden_size)
        self.lstm_att = nn.Linear(hidden_size, hidden_size)
        self.combined_att = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # (dim=0 is our current minitbatch)

    def forward(self, features, h):
        """ Forward step for attention network
        features - convolutional image features of shape ((W' * H') , C)
        h - hidden state of the decoder"""

        # torch.Size([128, 49, hidden_size])
        att_img = self.image_att(features)
        # torch.Size([128, 49])
        att_h = self.lstm_att(h)

        # [bs, num_locs, 1], last dimension squeezed away
        att_logits = self.combined_att(self.relu(att_img + att_h.unsqueeze(1))).squeeze(2)

        alphas = self.softmax(att_logits)

        att_context = (features * alphas.unsqueeze(2)).sum(dim=1)

        return att_context, alphas


class SoftAttentionDecoderRNN(nn.Module):
    # Show, attend, and tell soft attention implementation based on
    # https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
    def __init__(self, p, vocab_size, ext_features_dim=0):
        """Set the hyper-parameters and build the layers."""
        super(SoftAttentionDecoderRNN, self).__init__()

        (self.extractors,
         int_features_dim) = FeatureExtractor.list(p.persist_features.internal)

        # Attention needs 3d tensors inputs, so we just use the one available tensor:
        total_feat_dim = ext_features_dim if ext_features_dim else int_features_dim

        print('SoftAttentionDecoderCNN: total feature dim={}'.format(total_feat_dim))

        assert len(total_feat_dim) == 3, \
            "wrong number of input feature dimensions %d" % len(total_feat_dim)

        self.vocab_size = vocab_size

        # number of channels in the convolutional features / or alternatively
        # dimension D of the average-poooled convnet output:
        self.feature_size = int(total_feat_dim[0])

        # How many grid locations do we do attention on
        # for a 7 x 7 x 2048 convolutional layer this would be 7*7=49
        self.num_attention_locs = int(total_feat_dim[1] * total_feat_dim[2])

        # We use the same size for LSTM hidden unit and Attention network:
        self.hidden_size = p.hidden_size

        # Attention network, implements function f_att() or phi() depending on paper:
        self.attention = SpatialAttention(self.feature_size, self.num_attention_locs,
                                          p.hidden_size)

        self.embed = nn.Embedding(vocab_size, p.embed_size)

        # Next 2 functions transform mean feature vector into c_0 and h_0
        # self.init_h = nn.Linear(self.feature_size, p.hidden_size)
        # self.init_c = nn.Linear(self.feature_size, p.hidden_size)

        self.init_h = nn.ModuleList()
        self.init_c = nn.ModuleList()
        for _layer in range(p.num_layers):
            self.init_h.append(nn.Linear(self.feature_size, p.hidden_size))
            self.init_c.append(nn.Linear(self.feature_size, p.hidden_size))

        # Gating scalar used for element wise weighing of context vector:
        self.f_beta = nn.Linear(p.hidden_size, self.feature_size)
        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(p=p.dropout)

        self.lstm_step = nn.LSTMCell(p.embed_size + self.feature_size, p.hidden_size)

        # fc layer that predicts scores for each word in vocabulary
        self.linear = nn.Linear(p.hidden_size, vocab_size)

        # Init some weights from uniform distribution
        self.init_weights()

    def init_weights(self):
        """ Initializes some parameters with values from the uniform distribution,
        for easier convergence. """
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-0.1, 0.1)

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

    def forward(self, encoder_features, captions, lengths, images, external_features=None,
                teacher_p=1.0, teacher_forcing='always'):

        batch_size = captions.size()[0]
        seq_length = captions.size()[1]

        with torch.no_grad():
            persist_features = self._cat_features(images, external_features)
            assert persist_features is not None, 'ERROR: Persistant features '
            'are needed for Attention decoder'

        # Flatten (BS x C x W x H) image to (BS x C x (W*H)),
        # where self.feature_size is for example 2048 in case of ResNet152
        # 224x224 input images:
        # features = persist_features.view(batch_size, -1, self.feature_size)
        features = external_features.view(batch_size, -1, self.feature_size)

        embeddings = self.embed(captions)

        h, c = init_hidden_state(self, features)

        # Store predictions and alphas here:
        outputs = torch.zeros(batch_size, seq_length, self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, seq_length, self.num_attention_locs).to(device)

        index = captions[:, 0].unsqueeze(1)
        # Create one-hot encoding of the <start> token:
        outputs[:, 0] = torch.zeros(batch_size,
                                    self.vocab_size).to(device).scatter_(1, index, 1)

        for t in range(seq_length - 1):
            batch_size_t = sum([l > t for l in lengths])
            att_context, alpha = self.attention(features[:batch_size_t], h[:batch_size_t])

            # Perform the gating as per Show, Attend and Tell:
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))

            att_context = gate * att_context
            h, c = self.lstm_step(
                torch.cat([embeddings[:batch_size_t, t], att_context], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))

            # TODO: Show attend and tell fed both the h_t and z_t here:
            # (z_t is the attention context at given time-step)
            outputs_t = self.linear(self.dropout(h))
            outputs[:batch_size_t, t + 1] = outputs_t

            alphas[:batch_size_t, t + 1] = alpha

        outputs = pack_padded_sequence(outputs, lengths, batch_first=True)[0]

        return outputs, alphas

    def sample(self, features, images, external_features, states=None, max_seq_length=20,
               start_token_id=None):
        sampled_ids = []

        # Concatenate internal and external features
        persist_features = self._cat_features(images, external_features)
        if persist_features is None:
            persist_features = features.new_empty(0)

        batch_size = len(images)
        alphas = torch.zeros(batch_size, max_seq_length, self.num_attention_locs).to(device)

        features = persist_features.view(batch_size, -1, self.feature_size)

        h, c = init_hidden_state(self, features)

        assert start_token_id is not None

        # Append start token to the output:
        predicted_initial = torch.ones([batch_size],
                                       dtype=torch.int64).to(device) * start_token_id
        sampled_ids.append(predicted_initial)

        start_token_embedding = self.embed(torch.tensor(start_token_id).to(device))
        embed_size = len(start_token_embedding)

        embeddings = start_token_embedding.unsqueeze(0).expand(batch_size,
                                                               embed_size)

        for t in range(max_seq_length):
            att_context, alpha = self.attention(features, h)

            gate = self.sigmoid(self.f_beta(h))

            att_context = gate * att_context

            h, c = self.lstm_step(
                torch.cat([embeddings, att_context], dim=1), (h, c))
            alphas[:, t] = alpha

            outputs = self.linear(h)
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)

            # inputs: (batch_size, 1, embed_size + len(external_features))
            embeddings = self.embed(predicted)

        # sampled_ids: (batch_size, max_seq_length)
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids, alphas


class SoftAttentionEncoderDecoder(nn.Module):
    def __init__(self, params, device, vocab_size, state, ef_dims, lr_dict={}):
        super(SoftAttentionEncoderDecoder, self).__init__()
        print('Using device: {}'.format(device.type))
        print('Initializing SoftAttentionEncoderDecoder model...')
        self.encoder = EncoderCNN(params, ef_dims[0]).to(device)
        self.decoder = SoftAttentionDecoderRNN(params, vocab_size, ef_dims[1]).to(device)

        self.opt_params = (list(self.decoder.parameters()) +
                           list(self.encoder.linear.parameters()) +
                           list(self.encoder.bn.parameters()))

        if state:
            self.encoder.load_state_dict(state['encoder'])
            self.decoder.load_state_dict(state['decoder'])

    def get_opt_params(self):
        return self.opt_params

    def forward(self, images, init_features, captions, lengths, persist_features,
                teacher_p=1.0, teacher_forcing='always'):
        features = self.encoder(images, init_features)
        outputs, alphas = self.decoder(features, captions, lengths, images, persist_features,
                                       teacher_p, teacher_forcing)
        return outputs, alphas

    def sample(self, image_tensor, init_features, persist_features, states=None,
               max_seq_length=20, start_token_id=None):
        feature = self.encoder(image_tensor, init_features)
        sampled_ids, alphas = self.decoder.sample(feature, image_tensor, persist_features,
                                                  states, max_seq_length=max_seq_length,
                                                  start_token_id=start_token_id)

        return sampled_ids, alphas


class SpatialAttentionDecoderRNN(nn.Module):
    def __init__(self, p, vocab_size, ext_features_dim=0):
        """Set the hyper-parameters and build the layers."""
        super(SpatialAttentionDecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, p.embed_size)

        print('SpatialAttentionDecoderRNN: total feature dim: {}'.
              format(ext_features_dim))

        assert len(ext_features_dim) == 3, \
            "wrong number of input feature dimensions %d" % len(ext_features_dim)

        self.vocab_size = vocab_size
        self.feature_size = ext_features_dim[0]
        self.num_attention_locs = ext_features_dim[1] * ext_features_dim[2]

        self.hidden_size = p.hidden_size
        # Next 2 functions transform mean feature vector into c_0 and h_0
        self.init_h = nn.Linear(self.feature_size, p.hidden_size)
        self.init_c = nn.Linear(self.feature_size, p.hidden_size)

        self.dropout = nn.Dropout(p=p.dropout)

        self.lstm_step = nn.LSTMCell(p.embed_size, p.hidden_size)
        self.attention = SpatialAttention(self.feature_size, self.num_attention_locs,
                                          p.hidden_size)
        self.linear = nn.Linear(p.hidden_size + self.feature_size, vocab_size)


    def forward(self, encoder_features, captions, lengths, images, external_features=None,
                teacher_p=1.0, teacher_forcing='always'):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat([encoder_features.unsqueeze(1), embeddings], 1)
        seq_length = embeddings.size()[1]
        batch_size = embeddings.size()[0]

        features = external_features.view(batch_size, -1, self.feature_size)

        # h, c = self.init_hidden_state(features)

        h = torch.zeros(batch_size, self.hidden_size).to(device)
        c = torch.zeros(batch_size, self.hidden_size).to(device)

        outputs = torch.zeros(batch_size, seq_length, self.vocab_size).to(device)
        # Insert the <start> token into outputs tensors at the first location of each sequence:
        index = captions[:, 0].unsqueeze(1)
        # Create one-hot encoding of the <start> token:
        # outputs[:, 0] = torch.zeros(batch_size, vocab_size).to(device).scatter_(1, index, 1)

        alphas = torch.zeros(batch_size, seq_length, self.num_attention_locs).to(device)

        for t in range(seq_length - 1):
            batch_size_t = sum([l > t for l in lengths])
            h, c = self.lstm_step(embeddings[:batch_size_t, t],
                                  (h[:batch_size_t], c[:batch_size_t]))
            att_context, alpha = self.attention(features[:batch_size_t], h[:batch_size_t])

            outputs_t = self.linear(torch.cat([self.dropout(h), att_context], dim=1))
            outputs[:batch_size_t, t] = outputs_t

            alphas[:batch_size_t, t] = alpha

        outputs = pack_padded_sequence(outputs, lengths, batch_first=True)[0]

        return outputs, alphas

    def sample(self, features, images, external_features, states=None, max_seq_length=20):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        batch_size = len(images)
        alphas = torch.zeros(batch_size, max_seq_length, self.num_attention_locs).to(device)

        features = external_features.view(batch_size, -1, self.feature_size)

        h, c = init_hidden_state(self, features)

        # inputs: (batch_size, 1, embed_size + len(external features))
        inputs = features.unsqueeze(1)

        for t in range(max_seq_length):
            h, c = self.lstm_step(inputs, (h, c))
            att_context, alpha = self.attention(features, h)
            alphas[:, t] = alpha
            outputs = self.linear(torch.cat([h, att_context], dim=1))
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)

            # inputs: (batch_size, 1, embed_size + len(external_features))
            embeddings = self.embed(predicted)
            inputs = embeddings.unsqueeze(1)

        # sampled_ids: (batch_size, max_seq_length)
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids, alphas


class SpatialAttentionEncoderDecoder(nn.Module):
    def __init__(self, params, device, vocab_size, state, ef_dims):
        super(SpatialAttentionEncoderDecoder, self).__init__()
        print('Using device: {}'.format(device.type))
        print('Initializing SpatialAttentionEncoderDecoder model...')
        self.encoder = EncoderCNN(params, ef_dims[0]).to(device)
        self.decoder = SpatialAttentionDecoderRNN(params, vocab_size, ef_dims[1]).to(device)

        self.opt_params = (list(self.decoder.parameters()) +
                           list(self.encoder.linear.parameters()) +
                           list(self.encoder.bn.parameters()))

        if state:
            self.encoder.load_state_dict(state['encoder'])
            self.decoder.load_state_dict(state['decoder'])

    def get_opt_params(self):
        return self.opt_params

    def forward(self, images, init_features, captions, lengths, persist_features,
                teacher_p=1.0, teacher_forcing='always'):
        features = self.encoder(images, init_features)
        outputs, alphas = self.decoder(features, captions, lengths, images, persist_features,
                                       teacher_p, teacher_forcing)
        return outputs, alphas

    def sample(self, image_tensor, init_features, persist_features, states=None,
               max_seq_length=20):
        features = self.encoder(image_tensor, init_features)
        sampled_ids = self.decoder.sample(features, image_tensor, 
                                          external_features=persist_features,
                                          states=states, max_seq_length=max_seq_length)

        return sampled_ids



