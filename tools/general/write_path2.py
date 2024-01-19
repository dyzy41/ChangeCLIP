import os
import re


p = '/home/user/dsj_files/SegDATA/OpenEarthMap/raw/OpenEarthMap/OpenEarthMap/OpenEarthMap_wo_xBD'
save_path = '/home/user/dsj_code/mmsegcd/mmsegmentation/data/OpenEarthMap'
txtfiles = ['train.txt', 'val.txt', 'test.txt']

for item in txtfiles:
    names = open(os.path.join(p, item)).readlines()
    with open(os.path.join(save_path, item), 'w') as ff:
        for name in names:
            dir_name = re.sub(r"_\d+\.tif$", "", name.strip())
            sub_dir = os.path.join(p, dir_name)
            img_path = os.path.join(sub_dir, 'images', name.strip())
            lab_path = os.path.join(sub_dir, 'labels', name.strip())
            if os.path.exists(img_path) is True and os.path.exists(lab_path) is True:
                ff.writelines('{}  {}\n'.format(img_path, lab_path))
    


        
