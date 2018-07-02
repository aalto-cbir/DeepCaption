#!/bin/bash

# Take csv as input
# first row of CSV = param names
# all other rows = param values

# Usage: $0 hyperparams.csv --other_param1 value1 --other_paramN valueN

if [ -z "$*" ]; then
    echo "Usage: $0 path_to_csv.csv --other_param1 val1 ... --other_paramN valN"
    exit 1
fi

PASS_THROUGH_PARAMS=${@:2} 
CSV=$1

echo "Script params:"
echo "Loading parameters from $CSV" 
echo "Using common parameters: $PASS_THROUGH_PARAMS" 


# Read CSV column names into an array - these should be the same as parameter names
HEADER=`head -n1 $CSV | tr -d '\r'` # need to remove trailing carriage return to avoid surprises
PARAM_NAMES=( ${HEADER//,/ } )

# Read in the rows of the CSV starting with row 2:
PARAM_ROWS=`tail -n+2 $CSV`
for ROW in $PARAM_ROWS; do
    PARAMS=""
    ROW=`echo $ROW | tr -d '\r'` # Again, remove carriage return
    ROW_ARR=( ${ROW//,/ } )
    for ((i=0; i<${#PARAM_NAMES[@]}; i++)); do
        PARAMS=${PARAMS}" --${PARAM_NAMES[$i]} ${ROW_ARR[$i]}"
    done
    ALL_PARAMS="$PARAMS $PASS_THROUGH_PARAMS"
    (set -x; sbatch --time=0-5 submit.sh train.py ${ALL_PARAMS})
    sleep 1
done
#echo $PARAM_ROWS


#for MODEL in "$@"
#do
#    echo ${MODEL}
#    echo "${@:2}"
    #(set -x; sbatch --time=0-5 submit.sh infer.py --model ${MODEL})
    #sleep 1
#done
