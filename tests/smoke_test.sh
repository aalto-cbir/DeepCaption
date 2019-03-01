#!/bin/bash

# Smoke test the training, inference and evaluation code to make sure that the
# basic functionality is in place

# Include test functions:

# Get the relative directory of the testing script:
DIR=`dirname "$0"`
. $DIR/functions.sh

# Uncomment lines below to test the testing functionality itself: 
#echo "foo"
#append_command_to_log
#grep foo foo
#append_command_to_log
#date foo
#append_command_to_log
#date
#append_command_to_log
#exit

# Basic training run:

load_python3

# Expected result: training script starts training on MS COCO Train 2014 dataset, 
# each epoch only uses the first 10 batches for training.
# The model is using internal ResNet-152 features, and trains for 1 epoch:
./train.py --dataset coco:train2014 --vocab AUTO \
                  --num_batches 10 --model_name __test/simple --num_epochs 1 
append_command_to_log

# Expected result FAILURE: no vocab specified
# If the expected result for the command is different from 0, give it as a parameter to the following line:
./train.py --dataset coco:train2014 --num_batches 10 --model_name __test/simple --num_epochs 1 --replace
append_command_to_log 1

# Expected result FAILURE: no dataset selected
./train.py --vocab AUTO --num_batches 10 --model_name __test/simple --num_epochs 1 --replace
append_command_to_log 1

# Expected result: same as first, but validation loss and validation score
# are also calculated for each epoch:
./train.py --dataset coco:train2014 --validate coco:val2014 --vocab AUTO \
                  --num_batches 10 --model_name __test/validation --num_epochs 1 \
                  --validation_scoring cider
append_command_to_log

# Expected result: same as above, but using external features:
FEATURES=c_in12_rn152_pool5o_d_a.lmdb
./train.py --dataset coco:train2014 --validate coco:val2014 --features $FEATURES \
                  --vocab AUTO --num_batches 10 --num_epochs 1 \
                  --model_name __test/ext_features --validation_scoring cider
append_command_to_log


# Run the below section only if long running commands are not supposed to be skipped
if [ -z $SKIP_LONG_COMMANDS ]; then
  # Expected result: Model trained on MS COCO dataset with validation loss approaching
  # around 2.06 around epochs 11-12
  # Validation CIDEr 0.7250630490480385 at epoch 11
  ./train.py --dataset coco:train2014  --validate coco:val2014 --features $FEATURES \
                  --vocab AUTO --model_name __test/coco_full --lr_schedule \
                  --num_epochs 11 --num_workers 4 --num_layers 2 \
                 --dropout 0.2 --validation_scoring cider
  append_command_to_log
else
  echo "Skipping long running command..."
fi

MODEL=output/models/__test/coco_full/ep11.model
RESULTS_PATH=output/results/captions/__test/coco_full/
RESULTS_FILENAME=ep11.json

# Inference: Expected result after evaluating the resultant json file: 
# METEOR: 0.240, CIDEr: 0.851
# 
# Script output for test CIDEr 0.7403850771265595
./infer.py --model $MODEL --dataset coco:val2014 --num_workers 4 \
                  --results_path $RESULTS_PATH \
                  --output_file $RESULTS_FILENAME --scoring cider
append_command_to_log

# Training with validation only:
# Validation CIDEr 0.7250630490480385
./train.py --load_model $MODEL --validate coco:val2014 --num_workers 4 \
           --validate_only --validation_scoring cider
append_command_to_log

RESULTS_FILE=$RESULTS_PATH/$RESULTS_FILENAME

echo "We are on hostname: $HOSTNAME"

if [ ! -z $COCO_GT ]; then
  echo "Setting ground truth path for COCO from command line parameter: $COCO_GT"
  GROUND_TRUTH=$COCO_GT
elif [[ $ENVIRONMENT == "taito" ]]; then
  echo "Setting ground truth path for COCO evaluation for Taito..."
  GROUND_TRUTH=/proj/mediaind/picsom/databases/COCO/download/annotations/captions_val2014.json
elif [[ $ENVIRONMENT == "aalto" ]]; then
  echo "Setting ground truth path for COCO evaluation for Aalto/Triton..."
  GROUND_TRUTH=/m/cs/scratch/imagedb/picsom/databases/COCO/download/annotations/captions_val2014.json
else
  echo "Unknown environment - please use --coco_gt command line argument to specify the ground_truth"
fi


load_python2
# Expected results: Meteor close to 0.240, CIDEr close to: 0.851 -
./eval_coco.py $RESULTS_FILE --ground_truth $GROUND_TRUTH
append_command_to_log

    
