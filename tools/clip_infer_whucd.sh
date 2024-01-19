python clip_inference.py --src_path $CDPATH/EGY_BCD \
                        --split 256 \
                        --img_split A B \
                        --model_name ViT-B/16 \
                        --class_names_path rscls.txt \
                        --device cuda:1 \
                        --tag 56_vit16

