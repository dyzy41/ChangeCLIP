#!/bin/bash

data_name=$1
config_file=$2
num_gpu=$3
work_dirs=$4
model_checkpoint=$(find "$work_dirs" -name 'best_mIoU_iter_*.pth' -type f -print -quit)
echo $model_checkpoint
if [ "$data_name" == "WHUCD" ]; then
    label_dir="/home/ps/HDD/zhaoyq_data/CDdata/WHUCD/cut_data/label"
    bash tools/dist_test.sh "$config_file" "$model_checkpoint" $num_gpu --out "$work_dirs/test_result"
    python tools/metric.py --pppred "$work_dirs/test_result" --gggt "$label_dir"
elif [ "$data_name" == "SYSU" ]; then
    label_dir="/home/ps/HDD/zhaoyq_data/CDdata/SYSU-CD/test/label"
    bash tools/dist_test.sh "$config_file" "$model_checkpoint" $num_gpu --out "$work_dirs/test_result"
    python tools/metric.py --pppred "$work_dirs/test_result" --gggt "$label_dir"
elif [ "$data_name" == "EGY" ]; then
    label_dir="/home/ps/HDD/zhaoyq_data/CDdata/EGY_BCD/256/label"
    bash tools/dist_test.sh "$config_file" "$model_checkpoint" $num_gpu --out "$work_dirs/test_result"
    python tools/metric.py --pppred "$work_dirs/test_result" --gggt "$label_dir"
elif [ "$data_name" == "CDD" ]; then
    label_dir="/home/ps/HDD/zhaoyq_data/CDdata/ChangeDetectionDataset/Real/subset/test/OUT_new"
    bash tools/dist_test.sh "$config_file" "$model_checkpoint" $num_gpu --out "$work_dirs/test_result"
    python tools/metric.py --pppred "$work_dirs/test_result" --gggt "$label_dir"
elif [ "$data_name" == "LEVIR" ]; then
    bigimg_dir="/home/ps/HDD/zhaoyq_data/CDdata/LEVIR-CD/test/A"
    label_dir="/home/ps/HDD/zhaoyq_data/CDdata/LEVIR-CD/test/label"
    bash tools/dist_test.sh "$config_file" "$model_checkpoint" $num_gpu --out "$work_dirs/test_result"
    python tools/merge_data.py --p_predslice "$work_dirs/test_result" --p_bigimg "$bigimg_dir"
    python tools/metric.py --pppred "$work_dirs/test_result_big" --gggt "$label_dir"
elif [ "$data_name" == "LEVIRPLUS" ]; then
    bigimg_dir="/home/ps/HDD/zhaoyq_data/CDdata/LEVIR_CD_PLUS/test/A"
    label_dir="/home/ps/HDD/zhaoyq_data/CDdata/LEVIR_CD_PLUS/test/label"
    bash tools/dist_test.sh "$config_file" "$model_checkpoint" $num_gpu --out "$work_dirs/test_result"
    python tools/merge_data.py --p_predslice "$work_dirs/test_result" --p_bigimg "$bigimg_dir"
    python tools/metric.py --pppred "$work_dirs/test_result_big" --gggt "$label_dir"
fi
