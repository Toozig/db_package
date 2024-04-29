#!/bin/bash

# exp type = PWM/ AAA
EXP_TYPE=$1
# exp name ( e.g CHS / PBM)
EXP_NAME=$2
# submission file
SUBMISSION_FILE=$3

cd ibis-challenge

# check if cona 'bibis' exists
if ! conda env list | grep -q bibis; then
    conda env create -f environment.yml
fi
# if could not create conda env, exit
if [ $? -ne 0 ]; then
    echo "Error: Could not create conda environment."
    exit 1
fi
conda init bash
conda activate bibis
exit

# run PWM submission if EXP_TYPE is PWM
if [ "$EXP_TYPE" == "PWM" ]; then
    python cli/validate_pwm.py \
    --benchmark leaderboard_examples/example_PWM_benchmark.json \
    --pwm_sub leaderboard_examples/pwm_submission.txt \
    --bibis_root "."
# run AAA submission if EXP_TYPE is AAA
elif [ "$EXP_TYPE" == "AAA" ]; then
    python cli/validate_aaa.py \
    --benchmark leaderboard_examples/${EXP_NAME}_benchmark.json \
    --aaa_sub leaderboard_examples/example_${EXP_NAME}_sub.tsv \
    --bibis_root "."
fi

# deactivate conda env
conda deactivate