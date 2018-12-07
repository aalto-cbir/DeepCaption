#!/bin/bash

TRAIN_PY=$1 # ./train.py
INFER_PY=$2 # ./infer.py

# Train baseline model without start-token:

MODEL="__test/coco-max-default-nl2-skip_start_token"
$TRAIN_PY --num_workers 4 --lr_schedule  --optimizer adam --num_epochs 15  \
                 --dataset coco:train2014 --num_layers 2 --dropout 0.2 --vocab AUTO \
                 --features coco\:train2014+coco\:val2014-resnet152-max-normalize-default.lmdb \
                 --validate coco:val2014 \
                 --model_name EncoderDecoder/test/${MODEL} \
                 --skip_start_token


# Infer model with skipped start token:
$INFER_PY --model models/EncoderDecoder/test/${MODEL}/ep12.model \
                  --dataset coco:val2014 --output_format json