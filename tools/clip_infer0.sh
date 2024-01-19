python clip_inference.py --src_path /home/ps/HDD/zhaoyq_data/DATASET/LEVIR_CD_PLUS/cut_data \
                        --split train test \
                        --img_split A B \
                        --model_name ViT-B/16 \
                        --class_names_path /home/ps/zhaoyq_files/changeclip/ChangeCLIP/tools/rscls56.txt \
                        --device cuda:0 \
                        --tag 56_vit16

