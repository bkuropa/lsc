#!/bin/bash

dataset=$1
dataset_type=$2
metric=$3
batchSize=$4
tfenv=$5
gpu=$6
num_test_folds=$7
val_folds_per_test=$8
time_split=$9
val_epochs=${10}
test_epochs=${11}

source activate $tfenv

regression=""
if [ "$dataset_type" == "regression" ]
then
    regression=" -regression "
fi

for ((j=0;j<num_test_folds;j++)); do
    CUDA_VISIBLE_DEVICES=$gpu python step2.py -maxProc 5 -availableGPUs $gpu $regression -batchSize ${batchSize} -metric ${metric} -originalData ../../../scripts/lsc_data/${dataset}/ -dataset semi -saveBasePath ../../../compare_lsc_random/${dataset}/test/fold_$j/ -dataPathFolds ../../../data/${dataset}/random/fold_$j/0/split_indices.pckl -epochs $test_epochs -valBasePath ../../../compare_lsc_random/${dataset}/val/fold_$j/ -numValFolds $val_folds_per_test
done