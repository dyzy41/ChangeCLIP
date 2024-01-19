# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import torch
from mmengine.model import revert_sync_batchnorm
from mmengine.dataset import Compose
from collections import defaultdict
import numpy as np
import cv2
import os
from mmseg.apis import init_model
import tqdm


def _preprare_data(imgs, model):

    cfg = model.cfg
    for t in cfg.test_pipeline:
        if t.get('type') == 'LoadAnnotations':
            cfg.test_pipeline.remove(t)

    imgs = [imgs]
    is_batch = False

    # TODO: Consider using the singleton pattern to avoid building
    # a pipeline for each inference
    pipeline = Compose(cfg.test_pipeline)

    data = defaultdict(list)
    for img in imgs:
        data_ = dict(img_path=img[0], img_path2=img[1])
        data_ = pipeline(data_)
        data['inputs'].append(data_['inputs'])
        data['data_samples'].append(data_['data_samples'])

    return data, is_batch


def main():
    parser = ArgumentParser()
    parser.add_argument('file_list', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-folder', default='out_folder', help='folder to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)
    os.makedirs(args.out_folder, exist_ok=True)
    # test a single image
    file_list = open(args.file_list).readlines()
    for item in tqdm.tqdm(file_list):
        item = item.strip()
        pathA = item.split('  ')[0]
        pathB = item.split('  ')[1]
        img_name = os.path.basename(pathA).split('.')[0]+'.png'

        data, _ = _preprare_data([pathA, pathB], model)
        with torch.no_grad():
            results = model.test_step(data)
        # show the results
        pred_result = results[0].pred_sem_seg.data[0].cpu().detach().numpy()
        pred_result = np.where(pred_result>0, 255, 0).astype(np.uint8)
        cv2.imwrite(os.path.join(args.out_folder, img_name), pred_result)


if __name__ == '__main__':
    main()
