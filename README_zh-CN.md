# ChangeCLIP: Remote sensing change detection with multimodal vision-language representation learning  
https://www.sciencedirect.com/science/article/pii/S0924271624000042  


## 1. 为了方便使用相对路径，我在~/.bashrc中设置了CDPATH。你可以按照下面的方式在~/.bashrc文件中加入。
   ![CDPATH设置方法](image.png)  
   按照上述方法添加CDPATH以后，你可以使用这样的方式快速定位到相应的数据路径：  
```
import os  
data_root = os.path.join(os.environ.get("CDPATH"), 'SYSU-CD')
```
## 2. 我以SYSU-CD数据集为例介绍代码的使用方法，首先使用tools/general/write_path.py生成数据集路径的txt文件。格式如下(详情可查看代码)  
```
/home/user/dsj_files/CDdata/SYSU-CD/test/time1/03414.png  /home/user/dsj_files/CDdata/SYSU-CD/test/time2/03414.png  /home/user/dsj_files/CDdata/SYSU-CD/test/label/03414.png
/home/user/dsj_files/CDdata/SYSU-CD/test/time1/00708.png  /home/user/dsj_files/CDdata/SYSU-CD/test/time2/00708.png  /home/user/dsj_files/CDdata/SYSU-CD/test/label/00708.png
/home/user/dsj_files/CDdata/SYSU-CD/test/time1/03907.png  /home/user/dsj_files/CDdata/SYSU-CD/test/time2/03907.png  /home/user/dsj_files/CDdata/SYSU-CD/test/label/03907.png
/home/user/dsj_files/CDdata/SYSU-CD/test/time1/03107.png  /home/user/dsj_files/CDdata/SYSU-CD/test/time2/03107.png  /home/user/dsj_files/CDdata/SYSU-CD/test/label/03107.png
/home/user/dsj_files/CDdata/SYSU-CD/test/time1/02776.png  /home/user/dsj_files/CDdata/SYSU-CD/test/time2/02776.png  /home/user/dsj_files/CDdata/SYSU-CD/test/label/02776.png
/home/user/dsj_files/CDdata/SYSU-CD/test/time1/01468.png  /home/user/dsj_files/CDdata/SYSU-CD/test/time2/01468.png  /home/user/dsj_files/CDdata/SYSU-CD/test/label/01468.png
/home/user/dsj_files/CDdata/SYSU-CD/test/time1/00026.png  /home/user/dsj_files/CDdata/SYSU-CD/test/time2/00026.png  /home/user/dsj_files/CDdata/SYSU-CD/test/label/00026.png
/home/user/dsj_files/CDdata/SYSU-CD/test/time1/02498.png  /home/user/dsj_files/CDdata/SYSU-CD/test/time2/02498.png  /home/user/dsj_files/CDdata/SYSU-CD/test/label/02498.png
/home/user/dsj_files/CDdata/SYSU-CD/test/time1/02439.png  /home/user/dsj_files/CDdata/SYSU-CD/test/time2/02439.png  /home/user/dsj_files/CDdata/SYSU-CD/test/label/02439.png
/home/user/dsj_files/CDdata/SYSU-CD/test/time1/01057.png  /home/user/dsj_files/CDdata/SYSU-CD/test/time2/01057.png  /home/user/dsj_files/CDdata/SYSU-CD/test/label/01057.png
```
## 3.使用CLIP模型对SYSU-CD数据集进行推理，https://github.com/openai/CLIP, 生成置信度json文件  
   3.1  首先需要安装CLIP项目，运行如下命令  
```
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```
   3.2  然后运行以下命令
```
cd tools
bash clip_infer_sysu.sh
```
   3.3  运行完成后，会生成如下的文件：
```
/home/user/dsj_files/CDdata/SYSU-CD/train/time1_clipcls_56_vit16.json
/home/user/dsj_files/CDdata/SYSU-CD/train/time2_clipcls_56_vit16.json
/home/user/dsj_files/CDdata/SYSU-CD/val/time1_clipcls_56_vit16.json
/home/user/dsj_files/CDdata/SYSU-CD/val/time2_clipcls_56_vit16.json
/home/user/dsj_files/CDdata/SYSU-CD/test/time1_clipcls_56_vit16.json
/home/user/dsj_files/CDdata/SYSU-CD/test/time2_clipcls_56_vit16.json
```

## 4.可以查看tools/train.sh文件的内容，自行设定训练计划

# 致谢  
This repo benefits from awesome works of [mmsegmentation](https://github.com/open-mmlab/mmsegmentation), [DenseCLIP](https://github.com/raoyongming/DenseCLIP),
[CLIP](https://github.com/openai/CLIP). Please also consider citing them.  

# 引用
```bibtex
@article{DONG202453,
title = {ChangeCLIP: Remote sensing change detection with multimodal vision-language representation learning},
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
volume = {208},
pages = {53-69},
year = {2024},
issn = {0924-2716},
doi = {https://doi.org/10.1016/j.isprsjprs.2024.01.004},
url = {https://www.sciencedirect.com/science/article/pii/S0924271624000042},
author = {Sijun Dong and Libo Wang and Bo Du and Xiaoliang Meng}
}
```