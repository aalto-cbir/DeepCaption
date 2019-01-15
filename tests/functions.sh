#!/bin/bash

print_separator() {
    echo "============================================================================"
}

print_separator
echo "Loading test helper functions..."
TEST_FILE_NAME=`basename "$0"`
echo "Running $TEST_FILE_NAME..."
print_separator

# Use a temporary history file to record previous commands so that 
# we can output test results:
HISTFILE=/tmp/bash_history
set -o history

# Bash argument parser code from: 
# https://github.com/mattbryson/bash-arg-parse/blob/master/arg_parse_example
usage() {
    echo "Simple test runner: TEST_SCRIPT.SH -a AN_ARG -s SOME_MORE_ARGS [-y YET_MORE_ARGS || -h]"
    echo "   ";
    echo "  --coco_gt : STRING - Path to ground truth file for COCO";
    echo "  --skip_long_commands : BOOLEAN - Skip commands that take long time";
    echo "  --env : String - manually specify environment."
    echo "    supported environments: taito, aalto"
    echo "  -h | --help : This message";
}

parse_args() {
  while [ "$1" != "" ]; do
      # Note - shift inside case statement only for key-value arguments, not the flags!
      # SO here for COCO_GT we do shift because it's a key-value argument
      # AND, for SKIP_LONG_COMMANDS we don't because it's a flag, like 'store_action' in
      # Python's argparse.
      case "$1" in
          "--coco_gt" ) COCO_GT=$2; echo "COCO ground truth captions path: $COCO_GT"; shift;;
          "--skip_long_commands" ) SKIP_LONG_COMMANDS=TRUE; echo "Skipping long commands..."; ;;
          "--env" ) ENVIRONMENT=$2; shift;;
          "-h" | "--help" ) usage; exit;; # quit and show usage
      esac
      shift # move to next kv pair
  done
}

# Detect any known command line arguments:
parse_args $*

# Determine the environment from the args or the hostname
# Manual using --env flag:
if [[ ! -z $ENVIRONMENT ]]; then
  if [[ $ENVIRONMENT == "taito" ]] || [[ $ENVIRONMENT == "aalto" ]]; then
    echo "Setting user specified environment: $ENVIRONMENT";
  else
    echo "Unknown environment specified: $ENVIRONMENT"
    exit 1
  fi
# Automatic by checking the $HOSTNAME environmental variable:
else
  if [[ $HOSTNAME == *"taito"* ]]; then
    ENVIRONMENT="taito"
  elif [[ $HOSTNAME == *"triton"* ]] || [[ $HOSTNAME == *"aalto"* ]]; then
    ENVIRONMENT="aalto"
  fi

  if [[ ! -z $ENVIRONMENT ]]; then
    echo "Setting automatically detected environment: $ENVIRONMENT"
  else
    echo "Running in unknown environment. Some tests may fail."
  fi
fi

# Array of commands
declare -a COMMANDS

# Array of return codes:
declare -a RESULTS

# Array of expected return codes
declare -a EXPECTED_RESULTS


# Add a call to this function after every command that you want to track:
append_command_to_log() {
    local exit_code=$?

    if [ "$1" == "" ]; then
        local expected_exit_code=0
    else
        local expected_exit_code=$1
    fi

    # Get the last executed command:
    local command=$(echo `history |tail -n2 |head -n1` | sed 's/[0-9]* //')
    COMMANDS+=("$command")
    RESULTS+=($exit_code)
    EXPECTED_RESULTS+=($expected_exit_code)
    print_separator
    echo "Exit code: $exit_code, command: $command"
    print_separator
}

# Following function is executed before the script exits using the trap macro
# (Add a line with set -e to make the script exit on first error)
print_results() {
    NUM_SUCCESSES=0
    # Count the successful executions:
    for i in ${!COMMANDS[@]}; do 
        if [ ${RESULTS[$i]} -eq ${EXPECTED_RESULTS[$i]} ]; then
            ((NUM_SUCCESSES++))
        fi
    done

    print_separator
    echo "Finished running $TEST_FILE_NAME"
    echo "Test execution finished, $NUM_SUCCESSES / ${#COMMANDS[@]} commands succeeded"
    print_separator

    for i in ${!COMMANDS[@]}; do
        STATUS=""
        if [ ${RESULTS[$i]} -eq ${EXPECTED_RESULTS[$i]} ]; then
            STATUS='SUCCESS'
        else
            STATUS='FAILURE'
        fi

        echo "[$STATUS]: ${COMMANDS[$i]}"
    done

    print_separator
}

# Function for loading correct Python 3 version
load_python3() {
  if [[ $ENVIRONMENT == "taito" ]]; then
    print_separator
    echo "Loading Python 3 Environment on Taito"
    module load python-env/intelpython3.6-2018.3
    module load gcc/5.4.0 cuda/9.0 cudnn/7.1-cuda9
    print_separator
  else
    echo "WARNING: Did not load python3 due to an unknown environment"
  fi
  #TODO: Add the same for Triton or any other environments
}

# Function for loading correct Python 2 version
load_python2() {
  if [[ $ENVIRONMENT == "taito" ]]; then
    print_separator
    echo "Loading Python 2 Environment on Taito"
    module load python-env/2.7.10
    print_separator
  else
    echo "WARNING: Did not load python2 due to an unknown environment"
  fi
  #TODO: Add the same for Triton or any other environments
}

# Always run print_results command when the script exists for whatever reason
trap print_results EXIT