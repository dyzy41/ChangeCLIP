#!/bin/bash

data_name=$1
config_file=$2
num_gpu=$3
work_dirs=$4
model_checkpoint=$(find "$work_dirs" -name 'best_mIoU_iter_*.pth' -type f -print -quit)
echo $model_checkpoint
if [ "$data_name" == "WHUCD" ]; then
    label_dir="$CDPATH/WHUCD/cut_data/label"
    bash tools/dist_test.sh "$config_file" "$model_checkpoint" $num_gpu --out "$work_dirs/test_result"
    python tools/general/metric.py --pppred "$work_dirs/test_result" --gggt "$label_dir"
elif [ "$data_name" == "HRCUS" ]; then
    label_dir="$CDPATH/HRCUS-CD/test/label"
    bash tools/dist_test.sh "$config_file" "$model_checkpoint" $num_gpu --out "$work_dirs/test_result"
    python tools/general/metric.py --pppred "$work_dirs/test_result" --gggt "$label_dir" --pred_suffix ".png" --gt_suffix ".tif"
elif [ "$data_name" == "EGY" ]; then
    label_dir="$CDPATH/EGY_BCD/256/label"
    bash tools/dist_test.sh "$config_file" "$model_checkpoint" $num_gpu --out "$work_dirs/test_result"
    python tools/general/metric.py --pppred "$work_dirs/test_result" --gggt "$label_dir" --pred_suffix ".png" --gt_suffix ".png"
elif [ "$data_name" == "SYSU" ]; then
    label_dir="$CDPATH/SYSU-CD/test/label"
    bash tools/dist_test.sh "$config_file" "$model_checkpoint" $num_gpu --out "$work_dirs/test_result"
    python tools/general/metric.py --pppred "$work_dirs/test_result" --gggt "$label_dir"
elif [ "$data_name" == "CDD" ]; then
    label_dir="$CDPATH/ChangeDetectionDataset/Real/subset/test/OUT_new"
    bash tools/dist_test.sh "$config_file" "$model_checkpoint" $num_gpu --out "$work_dirs/test_result"
    python tools/general/metric.py --pppred "$work_dirs/test_result" --gggt "$label_dir"
elif [ "$data_name" == "CLCD" ]; then
    label_dir="$CDPATH/CLCD/test/label"
    bash tools/dist_test.sh "$config_file" "$model_checkpoint" $num_gpu --out "$work_dirs/test_result"
    python tools/general/metric.py --pppred "$work_dirs/test_result" --gggt "$label_dir"
elif [ "$data_name" == "GZCD" ]; then
    label_dir="$CDPATH/CD_Data_GZ/cut_data/labels_change"
    bash tools/dist_test.sh "$config_file" "$model_checkpoint" $num_gpu --out "$work_dirs/test_result"
    python tools/general/metric.py --pppred "$work_dirs/test_result" --gggt "$label_dir"
elif [ "$data_name" == "LEVIR" ]; then
    bigimg_dir="$CDPATH/LEVIR-CD/test/A"
    label_dir="$CDPATH/LEVIR-CD/test/label"
    bash tools/dist_test.sh "$config_file" "$model_checkpoint" $num_gpu --out "$work_dirs/test_result"
    python tools/general/merge_data.py --p_predslice "$work_dirs/test_result" --p_bigimg "$bigimg_dir"
    python tools/general/metric.py --pppred "$work_dirs/test_result_big" --gggt "$label_dir"
elif [ "$data_name" == "DSIFN" ]; then
    bigimg_dir="$CDPATH/DSIFN/test/t1"
    label_dir="$CDPATH/DSIFN/test/mask"
    bash tools/dist_test.sh "$config_file" "$model_checkpoint" $num_gpu --out "$work_dirs/test_result"
    # python tools/general/merge_data.py --p_predslice "$work_dirs/test_result" --p_bigimg "$bigimg_dir"
    python tools/general/metric.py --pppred "$work_dirs/test_result_big" --gggt "$label_dir"
elif [ "$data_name" == "LEVIRPLUS" ]; then
    bigimg_dir="$CDPATH/LEVIR_CD_PLUS/test/A"
    label_dir="$CDPATH/LEVIR_CD_PLUS/test/label"
    bash tools/dist_test.sh "$config_file" "$model_checkpoint" $num_gpu --out "$work_dirs/test_result"
    python tools/general/merge_data.py --p_predslice "$work_dirs/test_result" --p_bigimg "$bigimg_dir"
    python tools/general/metric.py --pppred "$work_dirs/test_result_big" --gggt "$label_dir"
fi


