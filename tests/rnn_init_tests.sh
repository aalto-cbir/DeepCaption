#!/bin/bash

# Train baseline model with rnn initialized from image features

# Include test functions:

# Get the relative directory of the testing script:
DIR=`dirname "$0"`
. $DIR/functions.sh

MODEL="__test/coco-max-default-nl2-rnn_init_tests"

load_python3
# Command where we disable <start> token generation should fail:
./train.py --num_workers 4 --lr_schedule  --optimizer adam --num_epochs 5  \
                 --dataset coco:train2014 --num_layers 2 --dropout 0.2 --vocab AUTO \
                 --features coco\:train2014+coco\:val2014-resnet152-max-normalize-default.lmdb \
                 --validate coco:val2014 \
                 --model_name EncoderDecoder/test/${MODEL} \
                 --rnn_hidden_init from_features \
                 --skip_start_token 
append_command_to_log 1

# The following should work:
./train.py --num_workers 4 --lr_schedule  --optimizer adam --num_epochs 5  \
                 --dataset coco:train2014 --num_layers 2 --dropout 0.2 --vocab AUTO \
                 --features coco\:train2014+coco\:val2014-resnet152-max-normalize-default.lmdb \
                 --validate coco:val2014 \
                 --model_name EncoderDecoder/test/${MODEL} \
                 --rnn_hidden_init from_features --num_batches 10 --validation_scoring cider
append_command_to_log                

# Infer model with skipped start token:
./infer.py --model models/EncoderDecoder/test/${MODEL}/ep5.model \
                  --dataset coco:val2014 --output_format json
append_command_to_log