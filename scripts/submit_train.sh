#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=16GB
#SBATCH --time=0-3
#SBATCH --mail-user=arturs.polis@aalto.fi
#SBATCH --mail-type=FAIL,REQUEUE,TIME_LIMIT_80
#SBATCH -o "train-%A_%a.out"
#SBATCH -c 2

module purge
module load python-env/intelpython3.6-2018.3
module load gcc/5.4.0 cuda/9.0 cudnn/7.1-cuda9

MODEL_TEMPLATE=$1 #  'coco-tf-additive-k${K}-beta${BETA}'
K=$2 # single number
BETAS=$3 # "0.025 0.05 0.1 0.5 1.0"
TF_MODE=$4 # always, additive, or sampled
# Convert input to array so that we can index it using task ID:
BETAS_ARRAY=($BETAS)
BETA1=${BETAS_ARRAY[$(($SLURM_ARRAY_TASK_ID * 2))]}
BETA2=${BETAS_ARRAY[$(($SLURM_ARRAY_TASK_ID * 2 + 1))]}

for BETA in $BETA1 $BETA2; do
    MODEL=$(eval echo $MODEL_TEMPLATE)
    if [ -f models/EncoderDecoder/$MODEL/ep15.model ]; then
       echo "models/EncoderDecoder/$MODEL/ep15.model already exists, skipping this model"
    else
       echo "Training $MODEL..."
       srun python3.6 train.py --model_name EncoderDecoder/$MODEL --validate coco:val2014 \
                               --num_workers 4 --lr_schedule --num_layers 2 --dropout 0.2 \
                               --teacher_forcing $TF_MODE --optimizer adam \
                               --teacher_forcing_k $K --teacher_forcing_beta $BETA \
                               --features c_in12_rn152_pool5o_d_a.lmdb --num_epochs 15 & 
    fi
done
wait
