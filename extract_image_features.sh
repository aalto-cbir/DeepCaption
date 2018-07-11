#!/bin/bash

date
printf "\n"

environment=$1
images_location=$2
features_location=$3
resized=$4

if [ -z "$1" ]; then
    echo "environment not specified, exiting now!"
    exit 0
fi

if [[ ${environment} = *"triton"* ]]; then
    echo "setting things for Triton!"

    #SBATCH -N 1
    #SBATCH -n 1
    #SBATCH -t 01:00:00
    #SBATCH --gres=gpu:teslap100:1
    #SBATCH --mem-per-cpu=32000

    module purge
    module load anaconda3/5.1.0-gpu CUDA/9.0.176 cuDNN/6.0-CUDA-8.0.61

    if [ -z "$2" ]; then
        echo "No images location, defaulting to COCO train-2014 images"
        images_location="/scratch/cs/imagedb/picsom/databases/COCO/download/images/train2014"
    fi
elif [[ ${environment} = *"taito"* ]]; then
    echo "setting things for Taito!"

    #SBATCH -N 1
    #SBATCH -n 1
    #SBATCH -p gpu
    #SBATCH -t 01:00:00
    #SBATCH -J gpu_job
    #SBATCH --gres=gpu:p100:1
    #SBATCH --mem-per-cpu=32000

    module purge
    module load intelconda/python3.6-2018.3

    if [ -z "$2" ]; then
        echo "No images location, defaulting to COCO train-2014 images"
        images_location="/proj/memad/COCO/train2014"
    fi

elif [[ ${environment} = *"MacbookPro"* ]]; then
    echo "testing locally :)"
    if [ -z "$2" ]; then
        echo "No images location, defaulting to some imageset"
        images_location="~/images/train"
    fi
else
    echo "haunted house, getting outta here!"
    exit 0
fi

echo "Images location: $images_location"
if [ -z "$3" ]; then
    ${features_location}="./features/"
fi
if [ ! -d "$features_location" ]; then
    echo "creating output directory"
    mkdir ${features_location}
fi
echo "Features location: $features_location"

if [ -z "$4" ]; then
    echo "Images will be resized"
    python resize.py --image_dir ${images_location} --output_dir "./resized_temp"
    OUT=$?
    if [ ${OUT} -eq 0 ];then
        echo "end: resizing complete"
    else
        echo "something broke: resizing"
        exit 42
    fi
else
    echo "Image resizing will be skipped"
fi

python feature_extractor.py --output_dir ${features_location}

OUT=$?
if [ ${OUT} -eq 0 ];then
    echo "end: feature extraction complete"
else
    echo "something broke: extracting features"
    exit 42
fi


# optional (could be modified to move elsewhere)
rm -rf ./resized_temp/

printf "\n"
date
