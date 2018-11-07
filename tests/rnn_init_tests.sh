#!/bin/bash

TRAIN_PY=$1
INFER_PY=$2

# Train baseline model with rnn initialized from image features

MODEL="coco-max-default-nl2-skip_start_token"

# Command where we do not disable <start> token generation should fail:
python3 $TRAIN_PY --num_workers 4 --lr_schedule  --optimizer adam --num_epochs 15  \
                 --num_layers 2 --dropout 0.2 --vocab AUTO \
                 --features coco\:train2014+coco\:val2014-resnet152-max-normalize-default.lmdb \
                 --validate coco:val2014 \
                 --model_name EncoderDecoder/test/${MODEL} \
                 --rnn_hidden_init from_features 

# The following should work:
python3 $TRAIN_PY --num_workers 4 --lr_schedule  --optimizer adam --num_epochs 15  \
                 --num_layers 2 --dropout 0.2 --vocab AUTO \
                 --features coco\:train2014+coco\:val2014-resnet152-max-normalize-default.lmdb \
                 --validate coco:val2014 \
                 --model_name EncoderDecoder/test/${MODEL} \
                 --rnn_hidden_init from_features \
                 --skip_start_token


# Infer model with skipped start token:
python3 $INFER_PY --model models/EncoderDecoder/test/${MODEL}/ep12.model \
                  --dataset coco:val2014 --output_format json