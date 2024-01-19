import cv2
import numpy as np
import os
import tqdm
import argparse


def process():
    imgs = os.listdir(args.p_bigimg)

    for item in tqdm.tqdm(imgs):
        bigimg = cv2.imread(os.path.join(args.p_bigimg, item))
        h, w, _ = bigimg.shape
        count_idx = np.zeros((h, w))
        down, left = cut_size, cut_size
        h_new = ((h - cut_size) // (cut_size - over_lap) + 1) * (cut_size - over_lap) + cut_size
        h_pad = h_new - h
        w_new = ((w - cut_size) // (cut_size - over_lap) + 1) * (cut_size - over_lap) + cut_size
        w_pad = w_new - w

        pad_u = h_pad // 2
        pad_d = h_pad - pad_u
        pad_l = w_pad // 2
        pad_r = w_pad - pad_l

        count_idx = np.pad(count_idx, ((pad_u, pad_d), (pad_l, pad_r)), 'reflect')
        pred = count_idx.copy()

        ni = 0
        while left <= w_new:
            slice_pred = pred[:, left - cut_size:left]

            ni += 1
            nj = 0
            while down <= h_new:
                lab_s = slice_pred[down - cut_size:down, :]
                nj += 1
                cut_lab = cv2.imread(
                    os.path.join(args.p_predslice, '{}_{}_{}.{}'.format(item.split('.')[0], ni, nj, suffix)), 0)
                pred[:, left - cut_size:left][down - cut_size:down, :] += cut_lab
                count_idx[:, left - cut_size:left][down - cut_size:down, :] += 1
                down = down + cut_size - over_lap
            down = cut_size
            left = left + cut_size - over_lap
        pred = pred / count_idx
        pred = pred.astype(np.uint8)
        pred = pred[pad_u:-pad_d, pad_l:-pad_r]
        pred = np.where(pred>0, 255, 0)
        cv2.imwrite(os.path.join(p_predbig, item), pred)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='参数')
    parser.add_argument('--p_bigimg', type=str, default='/home/CommonDatasets/CDdata/LEVIR-CD/test/A')
    parser.add_argument('--p_predslice', type=str, default='/home/dsj/0code_hub/cd_code/SegNeXt-main/work_dirs/segnext.base.512x512.levir.160k/test_result')
    args = parser.parse_args()

    p_predbig = args.p_predslice+'_big'

    if os.path.exists(os.path.join(p_predbig)) is False:
        os.mkdir(p_predbig)

    cut_size = 256
    over_lap = 64
    suffix = 'png'

    process()
