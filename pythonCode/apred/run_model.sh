#!/bin/bash

dataset=$1
dataset_type=$2
metric=$3
batchSize=$4
tfenv=$5
val_epochs=('300')
test_epochs=('300')
cp_folds=('1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12')
val_folds=('0' '1' '2')
test_folds=('3' '4' '5' '6' '7' '8' '9' '10' '11' '12')

regression=""
if [ "$dataset_type" == "regression" ]
then
    regression=" -regression "
fi

source activate $tfenv

echo "random split"
python estGPUSize.py -availableGPU 0 $regression -batchSize $batchSize -metric ${metric} -originalData ../../../scripts/lsc_data/${dataset}/ -dataset semi -saveBasePath ../../../compare_lsc_random/${dataset}/fold_0/ -dataPathFolds ../../../data/${dataset}/random/fold_0/split_indices.pckl
rm ../../../compare_lsc_random/${dataset}/fold_0/semi/o0003.start.pckl
for j in ${!cp_folds[@]}; do
    cp -r ../../../compare_lsc_random/${dataset}/fold_0/ ../../../compare_lsc_random/${dataset}/fold_${cp_folds[$j]}/
done
echo "step1"
for j in ${!val_folds[@]}; do
    CUDA_VISIBLE_DEVICES=0 python step1.py -maxProc 5 -availableGPUs 0 $regression -batchSize ${batchSize} -metric ${metric} -originalData ../../../scripts/lsc_data/${dataset}/ -dataset semi -saveBasePath ../../../compare_lsc_random/${dataset}/fold_${val_folds[$j]}/ -dataPathFolds ../../../data/${dataset}/random/fold_${val_folds[$j]}/split_indices.pckl -epochs ${val_epochs[$j]}
done
echo "step2"
for j in ${!test_folds[@]}; do
    CUDA_VISIBLE_DEVICES=0 python step2.py -maxProc 5 -availableGPUs 0 $regression -batchSize ${batchSize} -metric ${metric} -originalData ../../../scripts/lsc_data/${dataset}/ -dataset semi -saveBasePath ../../../compare_lsc_random/${dataset}/fold_${test_folds[$j]}/ -dataPathFolds ../../../data/${dataset}/random/fold_${test_folds[$j]}/split_indices.pckl -epochs ${test_epochs[$j]}
done

echo "time split"
python estGPUSize.py -availableGPU 0 $regression -batchSize $batchSize -metric ${metric} -originalData ../../../scripts/lsc_data/${dataset}/ -dataset semi -saveBasePath ../../../compare_lsc_time/${dataset}/fold_0/ -dataPathFolds ../../../data/${dataset}/time/fold_0/split_indices.pckl
rm ../../../compare_lsc_time/${dataset}/fold_0/semi/o0003.start.pckl
for j in ${!cp_folds[@]}; do
    cp -r ../../../compare_lsc_time/${dataset}/fold_0/ ../../../compare_lsc_time/${dataset}/fold_${cp_folds[$j]}/
done
echo "step2"
for j in ${!test_folds[@]}; do
    CUDA_VISIBLE_DEVICES=0 python step2.py -maxProc 5 -availableGPUs 0 $regression -batchSize ${batchSize} -metric ${metric} -originalData ../../../scripts/lsc_data/${dataset}/ -dataset semi -saveBasePath ../../../compare_lsc_time/${dataset}/fold_${test_folds[$j]}/ -dataPathFolds ../../../data/${dataset}/time/fold_${test_folds[$j]}/split_indices.pckl -epochs ${test_epochs[$j]} -valBasePath ../../../compare_lsc_random/${dataset}/fold_${test_folds[$j]}/
done

echo "scaffold split"
python estGPUSize.py -availableGPU 0 $regression -batchSize $batchSize -metric ${metric} -originalData ../../../scripts/lsc_data/${dataset}/ -dataset semi -saveBasePath ../../../compare_lsc_scaffold/${dataset}/fold_0/ -dataPathFolds ../../../data/${dataset}/scaffold/fold_0/split_indices.pckl
rm ../../../compare_lsc_scaffold/${dataset}/fold_0/semi/o0003.start.pckl
for j in ${!cp_folds[@]}; do
    cp -r ../../../compare_lsc_scaffold/${dataset}/fold_0/ ../../../compare_lsc_scaffold/${dataset}/fold_${cp_folds[$j]}/
done
echo "step2"
for j in ${!test_folds[@]}; do
    CUDA_VISIBLE_DEVICES=0 python step2.py -maxProc 5 -availableGPUs 0 $regression -batchSize ${batchSize} -metric ${metric} -originalData ../../../scripts/lsc_data/${dataset}/ -dataset semi -saveBasePath ../../../compare_lsc_scaffold/${dataset}/fold_${test_folds[$j]}/ -dataPathFolds ../../../data/${dataset}/scaffold/fold_${test_folds[$j]}/split_indices.pckl -epochs ${test_epochs[$j]} -valBasePath ../../../compare_lsc_random/${dataset}/fold_${test_folds[$j]}/
done

source deactivate