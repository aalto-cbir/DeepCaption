#!/bin/bash

#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p gpu
#SBATCH -t 06:00:00
#SBATCH -J gpu_job
#SBATCH --gres=gpu:p100:3
#SBATCH --mem-per-cpu=64000
#SBATCH

date
printf "\n"

echo "setting things for Taito!"

module load intelconda/python3.6-2018.3
cd /proj/memad/storytelling/VIST/image_captioning/

srun python resize.py --image_dir /proj/memad/storytelling/vist-baseline/dataset/images/train_full/ --output_dir /proj/memad/storytelling/vist-baseline/dataset/images/train_full/resized/

OUT=$?
if [ ${OUT} -eq 0 ];then
    echo "end: resizing complete"
else
    echo "something broke: resizing"
    exit 42
fi

printf "\n"
date
