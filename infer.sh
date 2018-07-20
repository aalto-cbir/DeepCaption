#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -t 010:00:00
#SBATCH -J gpu_job
#SBATCH --gres=gpu:p100:3
#SBATCH --mem-per-cpu=64000
#SBATCH

date
printf "\n"

echo "setting things for Taito!"

module load intelconda/python3.6-2018.3
cd /proj/memad/storytelling/VIST/image_captioning/

srun python infer_vist.py --dataset vist-seq --vocab_path ./vocab_train.pkl --image_dir /proj/memad/storytelling/vist-baseline/dataset/images/validate/ --caption_path /proj/memad/storytelling/vist-baseline/dataset/sis/val.story-in-sequence.json

OUT=$?
if [ ${OUT} -eq 0 ];then
    echo "end: inference complete"
else
    echo "something broke: infering"
    exit 42
fi

printf "\n"
date
