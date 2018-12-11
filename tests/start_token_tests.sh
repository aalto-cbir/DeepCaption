#!/bin/bash

# Include test functions:

# Get the relative directory of the testing script:
DIR=`dirname "$0"`
. $DIR/functions.sh

load_python3

# Smoke test training without start token:
MODEL="__test/coco-max-default-nl2-skip_start_token_smoke_test"
./train.py --num_workers 4 --lr_schedule  --optimizer adam --num_epochs 5  \
                 --dataset coco:train2014 --num_layers 2 --dropout 0.2 --vocab AUTO \
                 --features coco\:train2014+coco\:val2014-resnet152-max-normalize-default.lmdb \
                 --validate coco:val2014 --num_batches 10 \
                 --model_name EncoderDecoder/test/${MODEL} \
                 --skip_start_token --validation_scoring cider
append_command_to_log

# Infer model with skipped start token:
./infer.py --model models/EncoderDecoder/test/${MODEL}/ep5.model \
                  --dataset coco:val2014 --output_format json
append_command_to_log

if [ -z $SKIP_LONG_COMMANDS ]; then
    # Train full baseline model without start token:
    MODEL="__test/coco-max-default-nl2-skip_start_token"
    ./train.py --num_workers 4 --lr_schedule  --optimizer adam --num_epochs 11  \
                     --dataset coco:train2014 --num_layers 2 --dropout 0.2 --vocab AUTO \
                     --features coco\:train2014+coco\:val2014-resnet152-max-normalize-default.lmdb \
                     --validate coco:val2014 \
                     --model_name EncoderDecoder/test/${MODEL} \
                     --skip_start_token --validation_scoring cider
    append_command_to_log
fi

# Infer model with skipped start token:
./infer.py --model models/EncoderDecoder/test/${MODEL}/ep11.model \
                  --dataset coco:val2014 --output_format json
append_command_to_log
