#!/bin/bash

TRAIN_PY=$1
INFER_PY=$2

# Train baseline model with rnn initialized from image features

MODEL="__test/coco-max-default-nl2-rnn_init_tests"

# Command where we disable <start> token generation should fail:
$TRAIN_PY --num_workers 4 --lr_schedule  --optimizer adam --num_epochs 15  \
                 --dataset coco:train2014 --num_layers 2 --dropout 0.2 --vocab AUTO \
                 --features coco\:train2014+coco\:val2014-resnet152-max-normalize-default.lmdb \
                 --validate coco:val2014 \
                 --model_name EncoderDecoder/test/${MODEL} \
                 --rnn_hidden_init from_features \
                 --skip_start_token


# The following should work:
$TRAIN_PY --num_workers 4 --lr_schedule  --optimizer adam --num_epochs 15  \
                 --dataset coco:train2014 --num_layers 2 --dropout 0.2 --vocab AUTO \
                 --features coco\:train2014+coco\:val2014-resnet152-max-normalize-default.lmdb \
                 --validate coco:val2014 \
                 --model_name EncoderDecoder/test/${MODEL} \
                 --rnn_hidden_init from_features --num_batches 10
                

# Infer model with skipped start token:
$INFER_PY --model models/EncoderDecoder/test/${MODEL}/ep1.model \
                  --dataset coco:val2014 --output_format json