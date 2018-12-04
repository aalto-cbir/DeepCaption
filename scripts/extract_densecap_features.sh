#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:k80:1
#SBATCH --mem=128GB
#SBATCH --time=0-16
#SBATCH --mail-user=arturs.polis@aalto.fi
#SBATCH --mail-type=FAIL,REQUEUE,TIME_LIMIT_80
#SBATCH -o "train-%A_%a.out"
#SBATCH -c 1

module purge
module load gcc/4.9.3 mkl/11.3.0 intelmpi/5.1.1 fftw/3.3.4 hdf5-serial/1.8.15 cuda/7.5
source $USERAPPL/torch/install/bin/torch-activate

# $SLURM_ARRAY_TASK_ID is the current task id inside the array
# if we have array of size 10, and each job takes one input file we will have
# SLURM_ARRAY_TASK_COUNT is the current task

# print file names per array

INPUT_TEMPLATE=$1 # file_lists/image_file_list-coco:train2014:no_resize+coco:val2014:no_resize-taito-gpu.csc.fi_${n}_of_${N}.txt
OUTPUT_TEMPLATE=$2 # features/densecap_features-coco:train2014:no_resize+coco:val2014:no_resize_${n}_of_${N}.h5
# Current task index (1 to N):
n=$(($SLURM_ARRAY_TASK_ID + 1))
# Total number of tasks in array:
N=$SLURM_ARRAY_TASK_COUNT

# Path to DenseCap feature extractor:
DENSECAP_PATH='../densecap'

# Input and output files for feature extrator lua script:
INPUT_FILE=$(eval echo $INPUT_TEMPLATE)
OUTPUT_FILE=$(eval echo $OUTPUT_TEMPLATE)

echo "Densecap feature extractor:"
echo "Setting up job $SLURM_ARRAY_TASK_ID out of $SLURM_ARRAY_TASK_MAX"
echo "Input file: $INPUT_FILE"
echo "Output file: $OUTPUT_FILE"
echo "Starting job:"

cd $DENSECAP_PATH
srun th extract_features.lua -boxes_per_image 50 -input_txt $INPUT_FILE -output_h5 $OUTPUT_FILE
