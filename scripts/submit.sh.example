#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=8GB
#SBATCH --time=0-2
#SBATCH --mail-user=mats.sjoberg@aalto.fi
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

# Launch python script
srun $*
