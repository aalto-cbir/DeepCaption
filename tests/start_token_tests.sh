#!/bin/bash

# Train baseline model without start-token:

MODEL="__test/coco-max-default-nl2-skip_start_token"
./train.py --num_workers 4 --lr_schedule  --optimizer adam --num_epochs 15  \
                 --dataset coco:train2014 --num_layers 2 --dropout 0.2 --vocab AUTO \
                 --features coco\:train2014+coco\:val2014-resnet152-max-normalize-default.lmdb \
                 --validate coco:val2014 \
                 --model_name EncoderDecoder/test/${MODEL} \
                 --skip_start_token --validation_scoring cider


# Infer model with skipped start token:
./infer.py --model models/EncoderDecoder/test/${MODEL}/ep12.model \
                  --dataset coco:val2014 --output_format json