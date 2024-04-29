#!/bin/bash
workon deepBindEnv
# find all the files in the directory and run them
files=$(find ~/orenstein_lab/train/CHS/ -name '*.peaks')

#run the train script on all the files
for file in $files
do
    echo $file
    python3 train_ibis_chip_seq_v2.py $file v2 version_2
    # if failed, stop the script
    if [ $? -ne 0 ]; then
        echo "Failed to run the script"
        exit 1
    fi
done    

