python clip_inference.py --src_path /home/ps/HDD/zhaoyq_data/CDdata/EGY_BCD \
                        --split 256 \
                        --img_split A B \
                        --model_name ViT-B/16 \
                        --class_names_path /home/ps/zhaoyq_files/changeclip/ChangeCLIP/tools/rscls.txt \
                        --device cuda:1 \
                        --tag 56_vit16

