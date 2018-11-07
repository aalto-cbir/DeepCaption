#!/bin/bash

BUILD_VOCAB_PY=$1
TRAIN_PY=$2
INFER_PY=$3
VOCAB_USER="vocab"
VOCAB_CACHE="vocab_cache"

# Test vocab building with just COCO:
DATASET_1="coco:train2014"
OUTPUT_PKL_1="${VOCAB_USER}/vocab-${DATASET_1}.pkl"
OUTPUT_TXT_1="${VOCAB_USER}/vocab-${DATASET_1}.txt"
python3 $BUILD_VOCAB_PY --dataset ${DATASET_1} \
          --vocab_output_path ${OUTPUT_PKL_1} \
          --vocab_threshold 4 \
          --num_workers 4

python3 $BUILD_VOCAB_PY --dataset ${DATASET_1} \
          --vocab_output_path ${OUTPUT_TXT_1} \
          --vocab_threshold 4 \
          --num_workers 4

# Test vocab building with combined dataset:
DATASET_2="coco:train2014+msrvtt:train"
OUTPUT_PKL_2="${VOCAB_USER}/vocab-${DATASET_2}.pkl"
OUTPUT_TXT_2="${VOCAB_USER}/vocab-${DATASET_2}.txt"
python3 $BUILD_VOCAB_PY --dataset ${DATASET_2} \
          --vocab_output_path ${OUTPUT_PKL_2} \
          --vocab_threshold 4 \
          --num_workers 4

python3 $BUILD_VOCAB_PY --dataset ${DATASET_2} \
          --vocab_output_path ${OUTPUT_TXT_2} \
          --vocab_threshold 4 \
          --num_workers 4

# If vocab parameter is not set the script should terminate gracefully:
python3 $TRAIN_PY

# Train COCO with generated vocab file / pkl
python3 $TRAIN_PY --vocab ${OUTPUT_PKL_1} --model_name "test/vocab_pkl_${DATASET_1}" --num_epochs 1 --num_batches 1
python3 $TRAIN_PY --vocab ${OUTPUT_TXT_1} --model_name "test/vocab_txt_${DATASET_1}" --num_epochs 1 --num_batches 1
python3 $TRAIN_PY --vocab REGEN --model_name "test/vocab_pkl_${DATASET_1}_REGEN" --num_epochs 1 --num_batches 1
# Second time should fail because the file exists already
python3 $TRAIN_PY --vocab REGEN --model_name "test/vocab_pkl_${DATASET_1}_REGEN" --num_epochs 1 --num_batches 1
python3 $TRAIN_PY --vocab AUTO --model_name "test/vocab_pkl_${DATASET_1}_AUTO" --num_epochs 1 --num_batches 1

# Resume training a model:
python3 $TRAIN_PY --load_model "models/test/vocab_pkl_${DATASET_1}" --num_epochs 1 --num_batches 1

# Train COCO + MSRVTT with generated vocab file / pkl
python3 $TRAIN_PY --vocab ${OUTPUT_PKL_2} --model_name "test/vocab_pkl_${DATASET_2}" --num_epochs 1 --num_batches 1
python3 $TRAIN_PY --vocab ${OUTPUT_TXT_2} --model_name "test/vocab_txt_${DATASET_2}" --num_epochs 1 --num_batches 1
python3 $TRAIN_PY --vocab REGEN --model_name "test/vocab_pkl_${DATASET_2}_REGEN" --num_epochs 1 --num_batches 1
# Second time should fail because the file exists already
python3 $TRAIN_PY --vocab REGEN --model_name "test/vocab_pkl_${DATASET_2}_REGEN" --num_epochs 1 --num_batches 1
python3 $TRAIN_PY --vocab AUTO --model_name "test/vocab_pkl_${DATASET_2}_AUTO" --num_epochs 1 --num_batches 1

# Infer COCO with in-model vocab file
python3 $INFER_PY --model models/test/vocab_pkl_${DATASET_1}_AUTO/ep1.model --dataset coco:val2014

# Infer COCO with supplied vocab file
python3 $INFER_PY --model models/test/vocab_pkl_${DATASET_1}_AUTO/ep1.model \
                  --dataset coco:val2014 --vocab ${OUTPUT_PKL_1}

