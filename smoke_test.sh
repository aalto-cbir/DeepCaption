#!/bin/bash

TRAIN_PY=$1
INFER_PY=$2
EVAL_COCO_PY=$3

# Array of commands
declare -a COMMANDS

# Array of return codes:
declare -a RESULTS

append_command_to_log() {
    local exit_code=$?
    local command=$(echo `history |tail -n3 |head -n1` | sed 's/[0-9]* //')
    COMMANDS+=($command)
    RESULTS+=($exit_code)
    echo "Exit code: $exit_code, command: $command"
}

print_results() {
    NUM_SUCCESSES=0
    for code in ${RESULTS[*]}; do
        if [ code -eq 0 ]; then
            NUM_SUCCESSES+=1
        fi
    done

    echo "Test execution finished, $NUM_SUCCESSES / ${#COMMANDS[@]} commands succeeded"

    for i in ${#COMMANDS[@]}; do
        STATUS=""
        if [ ${RESULTS[$i]} -eq 0 ]; then
            STATUS='SUCCESS'
        else
            STATUS='FAILURE'
        fi

        echo "[$STATUS]: ${COMMANDS[$i]}"
    done
}

trap print_results EXIT

# Basic training run:
python3 $TRAIN_PY --num_batches 10 --model_name __test/simple
append_command_to_log

exit

# Expected result: training script starts training on MS COCO Train 2014 dataset, 
# each epoch only uses the first 10 batches for training.
# The model is using internal ResNet-152 features, and trains for 5 epochs

python3 $TRAIN_PY --validate coco:val2014 --num_batches 10 --model_name __test/validation
append_command_to_log

# Expected result: same as above, but validation loss is also calculated for each epoch
INIT=c_in12_rn152_pool5o_d_a.lmdb
python3 $TRAIN_PY --validate coco:val2014 --features $INIT \
                --num_batches 10 --model_name __test/ext_features
append_command_to_log

# Expected result: same as above, but using external features.

python3 $TRAIN_PY --validate coco:val2014 --features c_in12_rn152_pool5o_d_a.lmdb \
                --model_name __test/coco_full --lr_schedule \
                --num_epochs 12 --num_workers 4 --num_layers 2\
                --dropout 0.2
append_command_to_log
# Expected result: Model trained on MS COCO dataset with validation loss approaching
# around 2.06 around epochs 11-12

MODEL=models/__test/coco_full/ep11.model

python3 $INFER_PY --model $MODEL --dataset coco:val2014 --num_workers 4 \
                  --output_format json
append_command_to_log
# Expected result: reasonable looking results in results/__test_coco_full_ep11.json

RESULTS=results/__test/coco_full-ep11.json
GROUND_TRUTH=/m/cs/scratch/imagedb/picsom/databases/COCO/download/annotations/captions_val2014.json
python2 $EVAL_COCO_PY $RESULTS --ground_truth $GROUND_TRUTH
append_command_to_log
# Expected results: Meteor close to 0.240, CIDEr close to: 0.853

#INIT=c_in12_rn101_pool5o_d_a.lmdb,c_in12_rn152_pool5o_d_a.lmdb,fasterRcnn_clasDetFEat80.lmdb
INIT=c_in12_rn101_pool5o_d_a.lmdb,c_in12_rn152_pool5o_d_a.lmdb
#PERS=fasterRcnn_clasDetFEat80.lmdb,fasterRcnn_spatMapFeat3+3GaussScaleDet.lmdb,f::6gr::RBF::sun-397.lmdb

python3 $TRAIN_PY --dataset coco:train2014+msrvtt:train \
          --model_name __test/coco+msrvtt-rn \
          --embed_size=512 --hidden_size=1024 \
          --num_layers=2 --dropout=0.5 --encoder_dropout=0.5 \
          --optimizer=rmsprop \
          --validation msrvtt:validate --vocab_path vocab-coco+msrvtt.pkl \
          --features $INIT \
          --num_workers 2 --num_epochs=20
append_command_to_log

MODEL=models/__test/coco+msrvtt-rn/ep11.model

python3 $INFER_PY --model $MODEL --dataset trecvid2018 \
         --vocab_path vocab/vocab-coco+msrvtt.pkl \
         --output_format json
append_command_to_log

    
