import argparse


def parse_args():
    # default_dataset = 'coco:train2014'
    default_features = 'resnet152'

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='which dataset to use')
    parser.add_argument('--dataset_config_file', type=str,
                        default='datasets/datasets.conf',
                        help='location of dataset configuration file')
    parser.add_argument('--load_model', type=str, nargs='+',
                        help='existing model, for continuing training')
    parser.add_argument('--output_root', type=str, default='output',
                        help='Default directory for model output')
    parser.add_argument('--optimizer_reset', action="store_true",
                        help='reset optimizer parameters for loaded model')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--model_basename', type=str, default='model',
                        help='base name for model snapshot filenames')
    parser.add_argument('--model_path', type=str, default='models/',
                        help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')
    parser.add_argument('--tmp_dir_prefix', type=str,
                        default='image_captioning',
                        help='where in /tmp folder to store project data')
    parser.add_argument('--log_step', type=int, default=10,
                        help='step size for printing log info')
    parser.add_argument('--verbose', action="store_true", help="Increase verbosity")
    parser.add_argument('--profiler', action="store_true", help="Run in profiler")
    parser.add_argument('--cpu', action="store_true",
                        help="Use CPU even when GPU is available")
    parser.add_argument('--tensorboard', action="store_true",
                        help="Enable logging to TensorBoardX if is_available")

    # Vocabulary configuration:
    parser.add_argument('--vocab', type=str, default=None,
                        help='Vocabulary directive or path. '
                             'Directives are all-caps, no special characters. '
                             'Vocabulary file formats supported - *.{pkl,txt}.\n'
                             'AUTO: If vocabulary corresponding to current training set '
                             'combination exits in the vocab/ folder load it. '
                             'If not, generate a new vocabulary file\n'
                             'REGEN: Regenerate a new vocabulary file and place it in '
                             'vocab/ folder\n'
                             'path/to/vocab.\{pkl,txt\}: path to Pickled or plain-text '
                             'vocabulary file\n')
    parser.add_argument('--vocab_root', type=str, default='vocab_cache',
                        help='Cached vocabulary files folder')
    parser.add_argument('--no_tokenize', action='store_true')
    parser.add_argument('--show_tokens', action='store_true')
    parser.add_argument('--vocab_threshold', type=int, default=4,
                        help='minimum word count threshold')
    parser.add_argument('--show_vocab_stats', action="store_true",
                        help='show generated vocabulary word counts')
    parser.add_argument('--skip_start_token', action="store_true",
                        help='Do not prepend <start> token to caption')

    # Model parameters:
    parser.add_argument('--features', type=str, default=default_features,
                        help='features to use as the initial input for the '
                             'caption generator, given as comma separated list, '
                             'multiple features are concatenated, '
                             'features ending with .npy are assumed to be '
                             'precalculated features read from the named npy file, '
                             'example: "resnet152,foo.npy"')
    parser.add_argument('--persist_features', type=str,
                        help='features accessible in all caption generation '
                             'steps, given as comma separated list')
    parser.add_argument('--embed_size', type=int, default=256,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in lstm')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout for the LSTM')
    parser.add_argument('--encoder_dropout', type=float, default=0.0,
                        help='dropout for the encoder FC layer')
    parser.add_argument('--encoder_non_lin', action="store_true",
                        help='set this flag if you want EncoderRNN output to be passed '
                             'through a non-linearity (currently SELU non-linearity is used')
    parser.add_argument('--rnn_arch', type=str, default='LSTM',
                        help='RNN architecture to use in the decoder. Supported options: '
                             'GRU, LSTM. Default option: LSTM')
    parser.add_argument('--rnn_hidden_init', type=str,
                        help='initization strategy for RNN hidden and cell states. '
                             'Supported values: None (set to zeros), from_features '
                             '(using a linear transform on (mean/average pooled) '
                             'image feature vector')

    # Hierarchical model parameters, only in use if --hierarchical_model flag is set:
    parser.add_argument('--hierarchical_model', action='store_true',
                        help='Add this flag to train a model with hierarchical decoder RNN')
    parser.add_argument('--lr_word_decoder', type=float,
                        help='Set own learning rate for WordRNN in hierarchical model')
    parser.add_argument('--pooling_size', type=int, default=1024,
                        help='encoder pooling size')
    parser.add_argument('--max_sentences', type=int, default=6,
                        help='defines maximum number of sentences per caption'
                             'for hierachichal model')
    parser.add_argument('--weight_sentence_loss', type=float, default=5.0,
                        help='weight for the sentence loss for hierarchical model')
    parser.add_argument('--weight_word_loss', type=float, default=1.0,
                        help='weight for the word loss for hierarchical model')
    parser.add_argument('--dropout_stopping', type=float, default=0.0,
                        help='dropout for the stopping distribution')
    parser.add_argument('--dropout_fc', type=float, default=0.0,
                        help='dropout for the encoder FC layer')
    parser.add_argument('--fc_size', type=int,
                        help='size of the fully connected layer in the sentence LSTM '
                             'Defaults to pooling_size')
    # Enabled architecture by Chatterjee and Schwing, 2018:
    parser.add_argument('--coherent_sentences', action='store_true',
                        help="Enable coherence between sentences")

    # Training parameters
    parser.add_argument('--replace', action='store_true',
                        help='Replace the saved model with a new one')
    parser.add_argument('--force_epoch', type=int, default=0,
                        help='Force start epoch (for broken model files...)')
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--num_batches', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--grad_clip', type=float,
                        help='Value at which to clip weight gradients. Disabled by default')
    parser.add_argument('--validate', type=str,
                        help='Dataset to validate against after each epoch')
    parser.add_argument('--validation_step', type=int, default=1,
                        help='After how many epochs to perform validation, default=1')
    parser.add_argument('--validation_scoring', type=str)
    parser.add_argument('--validate_only', action='store_true',
                        help='Just perform validation with given model, no training')
    parser.add_argument('--skip_existing_validations', action='store_true')
    parser.add_argument('--optimizer', type=str, default="rmsprop")
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--lr_scheduler', nargs='?', const='ReduceLROnPlateau', type=str,
                        help='Use learning rate scheduler. Supported scheduler types: \n'
                             'ReduceLROnPlateau (used by default) .i.e plain --lr_scheduler\n'
                             'Other options: \n'
                             'StepLR - reduce learning rate by a factor of 0.5 every'
                             'lr_step_size steps\n'
                             'CyclicalLR - alternate learning rate between '
                             'lr_cyclical_max and lr_cyclical_min')
    parser.add_argument('--lr_cyclical_min', type=float, default=1e-5,
                        help='minimum learning rate for cyclical lr scheduler')
    parser.add_argument('--lr_cyclical_max', type=float, default=1e-3,
                        help='maximum learning rate for cyclical lr scheduler')
    parser.add_argument('--lr_step_size', type=int, default=5,
                        help='Default step size for StepLR lr scheduler')
    parser.add_argument('--share_embedding_weights', action='store_true',
                        help='Share weights for language model input and output embeddings')
    parser.add_argument('--self_critical_from_epoch', type=int, default=-1,
                        help='After what epoch do we start finetuning the model? '
                             '(-1 = disable; never finetune, 0 = finetune from start)')
    parser.add_argument('--self_critical_loss', type=str, default='sc_with_penalty',
                        help='Select the self-critical variant. Some types:\n'
                             'sc - Plain self-critical\n'
                             'sc_with_penalty - (default) With token penalty in case <start> or <end> arent generated\n'
                             'mixed - Mixed Self-critical with penalty and Cross entropy losses. '
                             'Mixture controled by --gamma_ml_rl arg.')
    parser.add_argument('--gamma_ml_rl', type=float, default=0.9995,
                        help='Between 0 and 1. Controls the mixture of RL and ML loss in MixedLoss')

    # For teacher forcing schedule see - https://arxiv.org/pdf/1506.03099.pdf
    parser.add_argument('--teacher_forcing', type=str, default='always',
                        help='Type of teacher forcing to use for training the Decoder RNN: \n'
                             'always: always use groundruth as LSTM input when training'
                             'sampled: follow a sampling schedule detemined by the value '
                             'of teacher_forcing_parameter\n'
                             'additive: use the sampling schedule formula to determine weight '
                             'ratio between the teacher and model inputs\n'
                             'additive_sampled: combines two of the above modes')
    parser.add_argument('--teacher_forcing_k', type=float, default=6500,
                        help='value of the sampling schedule parameter k. '
                             'Good values can be found in a range between 400 - 20000'
                             'small values = start using model output quickly, large values -'
                             ' wait for a while before start using model output')
    parser.add_argument('--teacher_forcing_beta', type=float, default=0.3,
                        help='sample scheduling parameter that determins the slope of '
                             'the middle segment of the sigmoid')

    args = parser.parse_args()

    # args asserts if needed

    return args
