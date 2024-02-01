#!/bin/bash

# bash tools/dist_test.sh configs/0cd_ce/changeclip_levir_vit_test.py \
#                         work_dirs_best/changeclip_levir_vit/best_mIoU_iter_18000.pth \
#                         2 \
#                         --out work_dirs_best/changeclip_levir_vit/test_result_slide_v2


# bash tools/dist_test.sh configs/0cd_ce/changeclip_levir_test.py \
#                         work_dirs_best/changeclip_levir/best_mIoU_iter_17000.pth \
#                         2 \
#                         --out work_dirs_best/changeclip_levir/test_result_slide_v2


python tools/test.py configs/0cd_ce/changeclip_levirplus_vit.py \
                        work_dirs_best/changeclip_levirplus_vit/best_mIoU_iter_19000.pth \
                        --out work_dirs_best/changeclip_levirplus_vit/test_result_slide_v2 \
                        --cfg-options model.test_cfg.mode='slide' \
                        --cfg-options model.test_cfg.crop_size="(256, 256)" \
                        --cfg-options model.test_cfg.stride="(128, 128)" \
                        --cfg-options test_dataloader.dataset.ann_file="/home/ps/HDD/zhaoyq_data/CDdata/LEVIR_CD_PLUS/test.txt"


# bash tools/dist_test.sh configs/0cd_ce/changeclip_levirplus_test.py \
#                         work_dirs_best/changeclip_levir/best_mIoU_iter_17000.pth \
#                         2 \
#                         --out work_dirs_best/changeclip_levir/test_result_slide_v2