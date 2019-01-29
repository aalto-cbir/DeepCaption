#!/bin/bash

# Include test functions:

# Get the relative directory of the testing script:
DIR=`dirname "$0"`
. $DIR/functions.sh

VOCAB_PATH="output/vocab_tests"

if [ -f $VOCAB_PATH ]; then
     echo "!!! $VOCAB_PATH exists please delete it to make sure all tests are run correctly"
fi

# Test vocab building with just COCO:
DATASET_1="coco:train2014"
OUTPUT_PKL_1="${VOCAB_PATH}/vocab-${DATASET_1}.pkl"
OUTPUT_TXT_1="${VOCAB_PATH}/vocab-${DATASET_1}.txt"
./build_vocab.py --dataset ${DATASET_1} \
          --vocab_output_path ${OUTPUT_PKL_1} \
          --vocab_threshold 4 \
          --num_workers 4
append_command_to_log

./build_vocab.py --dataset ${DATASET_1} \
          --vocab_output_path ${OUTPUT_TXT_1} \
          --vocab_threshold 4 \
          --num_workers 4
append_command_to_log

# Test vocab building with combined dataset:
DATASET_2="coco:train2014+msrvtt:train"
OUTPUT_PKL_2="${VOCAB_PATH}/vocab-${DATASET_2}.pkl"
OUTPUT_TXT_2="${VOCAB_PATH}/vocab-${DATASET_2}.txt"
./build_vocab.py --dataset ${DATASET_2} \
          --vocab_output_path ${OUTPUT_PKL_2} \
          --vocab_threshold 4 \
          --num_workers 4
append_command_to_log

./build_vocab.py --dataset ${DATASET_2} \
          --vocab_output_path ${OUTPUT_TXT_2} \
          --vocab_threshold 4 \
          --num_workers 4
append_command_to_log

# If vocab parameter is not set the script should terminate gracefully:
./train.py --dataset coco:train2014
append_command_to_log 1

# Train COCO with generated vocab file / pkl
./train.py --dataset coco:train2014 --vocab ${OUTPUT_PKL_1} \
          --model_name "test/vocab_pkl_${DATASET_1}" --num_epochs 1 --num_batches 1
append_command_to_log

./train.py --dataset coco:train2014 --vocab ${OUTPUT_TXT_1} \
          --model_name "test/vocab_txt_${DATASET_1}" --num_epochs 1 --num_batches 1
append_command_to_log

./train.py --dataset coco:train2014 --vocab REGEN \
          --model_name "test/vocab_pkl_${DATASET_1}_REGEN" --num_epochs 1 --num_batches 1
append_command_to_log

# Second time should fail because the file exists already
./train.py --dataset coco:train2014 --vocab REGEN \
          --model_name "test/vocab_pkl_${DATASET_1}_REGEN" --num_epochs 1 --num_batches 1 --replace
append_command_to_log 1

./train.py --dataset coco:train2014 --vocab AUTO \
          --model_name "test/vocab_pkl_${DATASET_1}_AUTO" --num_epochs 1 --num_batches 1
append_command_to_log

# Resume training a model:
./train.py --load_model "output/models/test/vocab_pkl_${DATASET_1}/ep1.model" --num_epochs 2 --num_batches 1 --replace
append_command_to_log

# Train COCO + MSRVTT with generated vocab file / pkl
./train.py --dataset coco:train2014+msrvtt:train --vocab ${OUTPUT_PKL_2} \
          --model_name "test/vocab_pkl_${DATASET_2}" --num_epochs 1 --num_batches 1
append_command_to_log

./train.py --dataset coco:train2014+msrvtt:train --vocab ${OUTPUT_TXT_2} \
          --model_name "test/vocab_txt_${DATASET_2}" --num_epochs 1 --num_batches 1
append_command_to_log

./train.py --dataset coco:train2014+msrvtt:train --vocab REGEN \
          --model_name "test/vocab_pkl_${DATASET_2}_REGEN" --num_epochs 1 --num_batches 1
append_command_to_log

# Second time should fail because the file exists already
./train.py --dataset coco:train2014+msrvtt:train --vocab REGEN \
          --model_name "test/vocab_pkl_${DATASET_2}_REGEN" --num_epochs 1 --num_batches 1 --replace
append_command_to_log 1

./train.py --dataset coco:train2014+msrvtt:train --vocab AUTO \
          --model_name "test/vocab_pkl_${DATASET_2}_AUTO" --num_epochs 1 --num_batches 1
append_command_to_log

# Infer COCO with in-model vocab file
./infer.py --model output/models/test/vocab_pkl_${DATASET_1}_AUTO/ep1.model --dataset coco:val2014
append_command_to_log

# Infer COCO with supplied vocab file
./infer.py --model output/models/test/vocab_pkl_${DATASET_1}_AUTO/ep1.model \
                  --dataset coco:val2014 --vocab ${OUTPUT_PKL_1}
append_command_to_log

