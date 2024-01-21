#!/usr/bin/env bash



# bash tools/dist_train.sh configs/0cd_ce/changeclip_sysu.py 2 --work-dir work_dirs_ce/changeclip_sysu
# bash tools/test.sh SYSU configs/0cd_ce/changeclip_sysu.py 2 work_dirs_ce/changeclip_sysu

# bash tools/dist_train.sh configs/0cd_ce/changeclip_sysu_vit.py 2 --work-dir work_dirs_ce/changeclip_sysu_vit
# bash tools/test.sh SYSU configs/0cd_ce/changeclip_sysu_vit.py 2 work_dirs_ce/changeclip_sysu_vit

# bash tools/dist_train.sh configs/0cd_ce/changeclip_whucd.py 2 --work-dir work_dirs_ce/changeclip_whucd
# bash tools/test.sh WHUCD configs/0cd_ce/changeclip_whucd.py 2 work_dirs_ce/changeclip_whucd

# bash tools/dist_train.sh configs/0cd_ce/changeclip_whucd_vit.py 2 --work-dir work_dirs_ce/changeclip_whucd_vit
# bash tools/test.sh WHUCD configs/0cd_ce/changeclip_whucd_vit.py 2 work_dirs_ce/changeclip_whucd_vit


bash tools/dist_train.sh configs/0cd_ce/changeclip_levir.py 2 --work-dir work_dirs_ce/changeclip_levir_v2
bash tools/test.sh LEVIR configs/0cd_ce/changeclip_levir.py 2 work_dirs_ce/changeclip_levir_v2

bash tools/dist_train.sh configs/0cd_ce/changeclip_levir_vit.py 2 --work-dir work_dirs_ce/changeclip_levir_vit_v2
bash tools/test.sh LEVIR configs/0cd_ce/changeclip_levir_vit.py 2 work_dirs_ce/changeclip_levir_vit_v2

# bash tools/dist_train.sh configs/0cd_ce/changeclip_levirplus.py 2 --work-dir work_dirs_ce/changeclip_levirplus
# bash tools/test.sh LEVIRPLUS configs/0cd_ce/changeclip_levirplus.py 2 work_dirs_ce/changeclip_levirplus

# bash tools/dist_train.sh configs/0cd_ce/changeclip_levirplus_vit.py 2 --work-dir work_dirs_ce/changeclip_levirplus_vit
# bash tools/test.sh LEVIRPLUS configs/0cd_ce/changeclip_levirplus_vit.py 2 work_dirs_ce/changeclip_levirplus_vit

bash tools/dist_train.sh configs/0cd_ce/changeclip_cdd.py 2 --work-dir work_dirs_ce/changeclip_cdd_v2
bash tools/test.sh CDD configs/0cd_ce/changeclip_cdd.py 2 work_dirs_ce/changeclip_cdd_v2

bash tools/dist_train.sh configs/0cd_ce/changeclip_cdd_vit.py 2 --work-dir work_dirs_ce/changeclip_cdd_vit_v2
bash tools/test.sh CDD configs/0cd_ce/changeclip_cdd_vit.py 2 work_dirs_ce/changeclip_cdd_vit_v2