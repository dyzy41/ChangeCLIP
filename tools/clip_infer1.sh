python clip_inference.py --src_path /home/ps/HDD/zhaoyq_data/DATASET/SYSU-CD \
                        --split train val test \
                        --img_split time1 time2 \
                        --model_name ViT-B/16 \
                        --class_names_path /home/ps/zhaoyq_files/changeclip/ChangeCLIP/tools/rscls.txt \
                        --device cuda:0 \
                        --tag 56_vit16

