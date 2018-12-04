#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=16GB
#SBATCH --time=0-1
#SBATCH --mail-user=arturs.polis@aalto.fi
#SBATCH --mail-type=FAIL,REQUEUE,TIME_LIMIT_80

# Set billing project
# newgrp mvsjober

# Load modules
# module purge
# module load python-env/3.5.3-ml
# module list

# research-support@csc.fi suggested for pytorch 0.4.0
module purge
module load python-env/intelpython3.6-2018.3
module load gcc/5.4.0 cuda/9.0 cudnn/7.1-cuda9


for model in models/EncoderDecoder/coco*alpha*; do
    for epoch in {11..15}; do
        srun python infer.py --dataset coco:val2014 \
                             --model ${model}/ep${epoch}.model \
                             --output_format json \
                             --num_workers 4
    done
done
