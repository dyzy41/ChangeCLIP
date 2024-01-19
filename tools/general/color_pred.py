import os
import cv2
import numpy as np

p = '/home/ps/zhaoyq_files/changeclip/ChangeCLIP/work_dirs/changeclip_levircdplus/test_pred'
save_result = '/home/ps/zhaoyq_files/changeclip/ChangeCLIP/work_dirs/changeclip_levircdplus/test_pred_vis'
os.makedirs(save_result, exist_ok=True)
files = os.listdir(p)
for item in files:
    img = cv2.imread(os.path.join(p, item), 0)
    img_vis = np.where(img > 0, 255, 0)
    cv2.imwrite(os.path.join(save_result, item), img_vis)
