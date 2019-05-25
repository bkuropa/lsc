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

regression=""
if [ "$dataset_type" == "regression" ]
then
    regression=" -regression "
fi

source activate $tfenv

echo "random split"
python estGPUSize.py -availableGPU $gpu $regression -batchSize $batchSize -metric ${metric} -originalData ../../../scripts/lsc_data/${dataset}/ -dataset semi -saveBasePath ../../../compare_lsc_random/${dataset}/val/fold_base/ -dataPathFolds ../../../data/${dataset}/random/fold_0/0/split_indices.pckl
rm ../../../compare_lsc_random/${dataset}/val/fold_base/semi/o0003.start.pckl
for ((j=0;j<num_test_folds;j++)); do
    for ((k=0;k<val_folds_per_test;k++)); do
        rm -r ../../../compare_lsc_random/${dataset}/val/fold_$j/$k/
        # cp -r ../../../compare_lsc_random/${dataset}/val/fold_base ../../../compare_lsc_random/${dataset}/val/fold_$j/$k/
        mkdir -p ../../../compare_lsc_random/${dataset}/val/fold_$j/$k/semi
        cp ../../../compare_lsc_random/${dataset}/val/fold_base/semi/* ../../../compare_lsc_random/${dataset}/val/fold_$j/$k/semi/
    done
done
echo "step1"
for ((j=0;j<num_test_folds;j++)); do
    for ((k=0;k<val_folds_per_test;k++)); do
        CUDA_VISIBLE_DEVICES=$gpu python step1.py -maxProc 10 -availableGPUs $gpu $regression -batchSize ${batchSize} -metric ${metric} -originalData ../../../scripts/lsc_data/${dataset}/ -dataset semi -saveBasePath ../../../compare_lsc_random/${dataset}/val/fold_$j/$k/ -dataPathFolds ../../../data/${dataset}/random/fold_$j/$k/split_indices.pckl -epochs $val_epochs
    done
done
echo "step2" # just use 0th fold for testing; it's just selecting the val set
for ((j=0;j<num_test_folds;j++)); do
    CUDA_VISIBLE_DEVICES=$gpu python step2.py -maxProc 10 -availableGPUs $gpu $regression -batchSize ${batchSize} -metric ${metric} -originalData ../../../scripts/lsc_data/${dataset}/ -dataset semi -saveBasePath ../../../compare_lsc_random/${dataset}/test/fold_$j/ -dataPathFolds ../../../data/${dataset}/random/fold_$j/0/split_indices.pckl -epochs $test_epochs -valBasePath ../../../compare_lsc_random/${dataset}/val/fold_$j/ -numValFolds $val_folds_per_test
done

if [ "$time_split" == "true" ]
then
    echo "time window split"
    python estGPUSize.py -availableGPU $gpu $regression -batchSize $batchSize -metric ${metric} -originalData ../../../scripts/lsc_data/${dataset}/ -dataset semi -saveBasePath ../../../compare_lsc_time_window/random/${dataset}/val/fold_base/ -dataPathFolds ../../../data/${dataset}/time_window/random/fold_0/0/split_indices.pckl
    for ((j=0;j<num_test_folds;j++)); do
        rm -r ../../../compare_lsc_time_window/random/${dataset}/val/fold_$j/0/
        rm -r ../../../compare_lsc_time_window/scaffold/${dataset}/val/fold_$j/0/
        rm -r ../../../compare_lsc_time_window/time/${dataset}/val/fold_$j/0/
        # cp -r ../../../compare_lsc_time_window/random/${dataset}/val/fold_base ../../../compare_lsc_time_window/random/${dataset}/val/fold_$j/0/
        # cp -r ../../../compare_lsc_time_window/random/${dataset}/val/fold_base ../../../compare_lsc_time_window/scaffold/${dataset}/val/fold_$j/0/
        # cp -r ../../../compare_lsc_time_window/random/${dataset}/val/fold_base ../../../compare_lsc_time_window/time/${dataset}/val/fold_$j/0/
        mkdir -p ../../../compare_lsc_time_window/random/${dataset}/val/fold_$j/0/semi
        mkdir -p ../../../compare_lsc_time_window/scaffold/${dataset}/val/fold_$j/0/semi
        mkdir -p ../../../compare_lsc_time_window/time/${dataset}/val/fold_$j/0/semi
        cp ../../../compare_lsc_time_window/random/${dataset}/val/fold_base/semi/* ../../../compare_lsc_time_window/random/${dataset}/val/fold_$j/0/semi/
        cp ../../../compare_lsc_time_window/random/${dataset}/val/fold_base/semi/* ../../../compare_lsc_time_window/scaffold/${dataset}/val/fold_$j/0/semi/
        cp ../../../compare_lsc_time_window/random/${dataset}/val/fold_base/semi/* ../../../compare_lsc_time_window/time/${dataset}/val/fold_$j/0/semi/
    done
    echo "step1"
    for ((j=0;j<num_test_folds;j++)); do
        CUDA_VISIBLE_DEVICES=$gpu python step1.py -maxProc 10 -availableGPUs $gpu $regression -batchSize ${batchSize} -metric ${metric} -originalData ../../../scripts/lsc_data/${dataset}/ -dataset semi -saveBasePath ../../../compare_lsc_time_window/random/${dataset}/val/fold_$j/0/ -dataPathFolds ../../../data/${dataset}/time_window/random/fold_$j/0/split_indices.pckl -epochs $val_epochs
        CUDA_VISIBLE_DEVICES=$gpu python step1.py -maxProc 10 -availableGPUs $gpu $regression -batchSize ${batchSize} -metric ${metric} -originalData ../../../scripts/lsc_data/${dataset}/ -dataset semi -saveBasePath ../../../compare_lsc_time_window/scaffold/${dataset}/val/fold_$j/0/ -dataPathFolds ../../../data/${dataset}/time_window/scaffold/fold_$j/0/split_indices.pckl -epochs $val_epochs
        CUDA_VISIBLE_DEVICES=$gpu python step1.py -maxProc 10 -availableGPUs $gpu $regression -batchSize ${batchSize} -metric ${metric} -originalData ../../../scripts/lsc_data/${dataset}/ -dataset semi -saveBasePath ../../../compare_lsc_time_window/time/${dataset}/val/fold_$j/0/ -dataPathFolds ../../../data/${dataset}/time_window/time/fold_$j/0/split_indices.pckl -epochs $val_epochs
    done
    echo "step2" # just use 0th fold for testing; it's just selecting the val set
    for ((j=0;j<num_test_folds;j++)); do
        CUDA_VISIBLE_DEVICES=$gpu python step2.py -maxProc 10 -availableGPUs $gpu $regression -batchSize ${batchSize} -metric ${metric} -originalData ../../../scripts/lsc_data/${dataset}/ -dataset semi -saveBasePath ../../../compare_lsc_time_window/random/${dataset}/test/fold_$j/0/ -dataPathFolds ../../../data/${dataset}/time_window/random/fold_$j/0/split_indices.pckl -epochs $test_epochs -valBasePath ../../../compare_lsc_time_window/random/${dataset}/val/fold_$j/ -numValFolds 1
        CUDA_VISIBLE_DEVICES=$gpu python step2.py -maxProc 10 -availableGPUs $gpu $regression -batchSize ${batchSize} -metric ${metric} -originalData ../../../scripts/lsc_data/${dataset}/ -dataset semi -saveBasePath ../../../compare_lsc_time_window/scaffold/${dataset}/test/fold_$j/0/ -dataPathFolds ../../../data/${dataset}/time_window/scaffold/fold_$j/0/split_indices.pckl -epochs $test_epochs -valBasePath ../../../compare_lsc_time_window/scaffold/${dataset}/val/fold_$j/ -numValFolds 1
        CUDA_VISIBLE_DEVICES=$gpu python step2.py -maxProc 10 -availableGPUs $gpu $regression -batchSize ${batchSize} -metric ${metric} -originalData ../../../scripts/lsc_data/${dataset}/ -dataset semi -saveBasePath ../../../compare_lsc_time_window/time/${dataset}/test/fold_$j/0/ -dataPathFolds ../../../data/${dataset}/time_window/time/fold_$j/0/split_indices.pckl -epochs $test_epochs -valBasePath ../../../compare_lsc_time_window/time/${dataset}/val/fold_$j/ -numValFolds 1
    done

    echo "time split"
    python estGPUSize.py -availableGPU $gpu $regression -batchSize $batchSize -metric ${metric} -originalData ../../../scripts/lsc_data/${dataset}/ -dataset semi -saveBasePath ../../../compare_lsc_time/${dataset}/val/fold_base/ -dataPathFolds ../../../data/${dataset}/time/fold_0/0/split_indices.pckl
    rm ../../../compare_lsc_time/${dataset}/val/fold_base/semi/o0003.start.pckl
    rm -r ../../../compare_lsc_time/${dataset}/val/fold_0/0/
    # cp -r ../../../compare_lsc_time/${dataset}/val/fold_base ../../../compare_lsc_time/${dataset}/val/fold_0/0/
    mkdir -p ../../../compare_lsc_time/${dataset}/val/fold_0/0/semi
    cp ../../../compare_lsc_time/${dataset}/val/fold_base/semi/* ../../../compare_lsc_time/${dataset}/val/fold_0/0/semi/
    echo "step1" # no loop for time since it's just one split
    CUDA_VISIBLE_DEVICES=$gpu python step1.py -maxProc 10 -availableGPUs $gpu $regression -batchSize ${batchSize} -metric ${metric} -originalData ../../../scripts/lsc_data/${dataset}/ -dataset semi -saveBasePath ../../../compare_lsc_time/${dataset}/val/fold_0/0/ -dataPathFolds ../../../data/${dataset}/time/fold_0/0/split_indices.pckl -epochs $val_epochs
    echo "step2"
    for ((j=0;j<num_test_folds;j++)); do
        CUDA_VISIBLE_DEVICES=$gpu python step2.py -maxProc 10 -availableGPUs $gpu $regression -batchSize ${batchSize} -metric ${metric} -originalData ../../../scripts/lsc_data/${dataset}/ -dataset semi -saveBasePath ../../../compare_lsc_time/${dataset}/test/fold_$j/ -dataPathFolds ../../../data/${dataset}/time/fold_$j/0/split_indices.pckl -epochs $test_epochs -valBasePath ../../../compare_lsc_time/${dataset}/val/fold_0/ -numValFolds 1
    done
fi

echo "scaffold split"
python estGPUSize.py -availableGPU $gpu $regression -batchSize $batchSize -metric ${metric} -originalData ../../../scripts/lsc_data/${dataset}/ -dataset semi -saveBasePath ../../../compare_lsc_scaffold/${dataset}/val/fold_base/ -dataPathFolds ../../../data/${dataset}/scaffold/fold_0/0/split_indices.pckl
rm ../../../compare_lsc_scaffold/${dataset}/val/fold_base/semi/o0003.start.pckl
for ((j=0;j<num_test_folds;j++)); do
    for ((k=0;k<val_folds_per_test;k++)); do
        rm -r ../../../compare_lsc_scaffold/${dataset}/val/fold_$j/$k/
        # cp -r ../../../compare_lsc_scaffold/${dataset}/val/fold_base ../../../compare_lsc_scaffold/${dataset}/val/fold_$j/$k/
        mkdir -p ../../../compare_lsc_scaffold/${dataset}/val/fold_$j/$k/semi
        cp ../../../compare_lsc_scaffold/${dataset}/val/fold_base/semi/* ../../../compare_lsc_scaffold/${dataset}/val/fold_$j/$k/semi/
    done
done
echo "step1"
for ((j=0;j<num_test_folds;j++)); do
    for ((k=0;k<val_folds_per_test;k++)); do
        CUDA_VISIBLE_DEVICES=$gpu python step1.py -maxProc 10 -availableGPUs $gpu $regression -batchSize ${batchSize} -metric ${metric} -originalData ../../../scripts/lsc_data/${dataset}/ -dataset semi -saveBasePath ../../../compare_lsc_scaffold/${dataset}/val/fold_$j/$k/ -dataPathFolds ../../../data/${dataset}/scaffold/fold_$j/$k/split_indices.pckl -epochs $val_epochs
    done
done
echo "step2"
for ((j=0;j<num_test_folds;j++)); do
    CUDA_VISIBLE_DEVICES=$gpu python step2.py -maxProc 10 -availableGPUs $gpu $regression -batchSize ${batchSize} -metric ${metric} -originalData ../../../scripts/lsc_data/${dataset}/ -dataset semi -saveBasePath ../../../compare_lsc_scaffold/${dataset}/test/fold_$j/ -dataPathFolds ../../../data/${dataset}/scaffold/fold_$j/0/split_indices.pckl -epochs $test_epochs -valBasePath ../../../compare_lsc_scaffold/${dataset}/val/fold_$j/ -numValFolds $val_folds_per_test
done

source deactivate