#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=16GB
#SBATCH --time=0-3
#SBATCH --mail-user=arturs.polis@aalto.fi
#SBATCH --mail-type=FAIL,REQUEUE,TIME_LIMIT_80
#SBATCH -o "infer-%j.out"
#SBATCH -c 4
#SBATCH -n 4

module purge
module load python-env/intelpython3.6-2018.3
module load gcc/5.4.0 cuda/9.0 cudnn/7.1-cuda9

EPOCH=$1 # Example input: 11
MODELS=$2 # Example input - a string that contains space-separated list: 'model1 model2 model3'
MODEL_TEMPLATE=$3 # Example input - template string with substitution placeholders: 
                  #models/EncoderDecoder/${NAME}/ep${EPOCH}.model' 
DATASET=$4 # Which dataset to use for inference
MODELS_ARR=($MODELS) # Convert a list of models into a BASH array
for NAME in $MODELS; do
   MODEL_FILE=$(eval echo $MODEL_TEMPLATE) # models/EncoderDecoder/${NAME}/ep${EPOCH}.model'
   if [ -f $MODEL_FILE ]; then
       # Check if the output for that model exists already:
       MODEL_ARRAY=(${MODEL_FILE//\// })
       ARR_LEN=${#MODEL_ARRAY[@]}
       MODEL_NAME=${MODEL_ARRAY[$ARR_LEN-2]}
       MODEL_EPOCH=${MODEL_ARRAY[$ARR_LEN-1]%.model}
       MODEL_RESULTS_JSON="results/${MODEL_NAME}-${MODEL_EPOCH}.json"
       if [ -f $MODEL_RESULTS_JSON ]; then
           echo "File $MODEL_RESULTS_JSON already exists, skipping"
           continue
       fi
       echo "Infering from $MODEL_FILE..."
       srun python3.6 infer.py --model $MODEL_FILE --dataset $DATASET --num_workers 4 --output_format json &
   else
       echo "Model $MODEL does not exist"
   fi
done
wait
