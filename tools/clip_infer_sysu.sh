python clip_inference.py --src_path $CDPATH/SYSU-CD \
                        --split train val test \
                        --img_split time1 time2 \
                        --model_name ViT-B/16 \
                        --class_names_path rscls.txt \
                        --device cuda:0 \
                        --tag 56_vit16

