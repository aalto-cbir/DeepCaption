#!/bin/bash

#SBATCH --time=07:00:00 --mem-per-cpu=32000
#SBATCH --gres=gpu:teslap100:1

date

module load anaconda3/5.1.0-gpu CUDA/9.0.176 cuDNN/6.0-CUDA-8.0.61

source activate /scratch/cs/imagedb/picsom/databases/vist/download/conda_vist

echo "start: extracting image features"

cd /scratch/cs/imagedb/picsom/databases/vist/analysis/adi/image_captioning/

python train.py

OUT=$?
if [ $OUT -eq 0 ];then
    echo "end: features saved as pickle objects"
else
    echo "something broke"
fi

source deactivate

date
