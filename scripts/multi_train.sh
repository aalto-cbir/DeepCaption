#!/bin/bash
#
# Evaluate multiple combinations of parameters from a CSV file
# - first row of CSV = param names
# - each subsequent row = corresponding values
#
if [ -z "$*" ]; then
    echo "Usage: $0 path/to/params.csv --common_model_param1 val1 ... --common_model_paramN valN"
    exit 1
fi

PASS_THROUGH_PARAMS=${@:2} 
CSV_FILE=$1

echo "Loading parameters from $CSV_FILE" 
echo "Using common parameters: $PASS_THROUGH_PARAMS" 

# Read CSV column names into an array - these should be the same as parameter names
HEADER=`head -n1 $CSV_FILE | tr -d '\r'` # remove trailing carriage return
PARAM_NAMES=( ${HEADER//,/ } )

# Read in the rows of the CSV starting from row 2 and 
# start batch jobs corresponding to each row:
PARAM_ROWS=`tail -n+2 $CSV_FILE`
for ROW in $PARAM_ROWS; do
    PARAMS=""
    ROW=`echo $ROW | tr -d '\r'` # Again, remove carriage return
    ROW_ARR=( ${ROW//,/ } )
    for ((i=0; i<${#PARAM_NAMES[@]}; i++)); do
        PARAMS=${PARAMS}" --${PARAM_NAMES[$i]} ${ROW_ARR[$i]}"
    done
    ALL_PARAMS="$PARAMS $PASS_THROUGH_PARAMS"
    (set -x; sbatch --time=0-8 submit.sh train.py ${ALL_PARAMS})
    sleep 1
done
