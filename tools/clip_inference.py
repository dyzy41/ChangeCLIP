import os
import json
import torch
import clip
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse


def read_class_names(path):
    with open(path, 'r') as f:
        class_names = [line.strip().split(', ')[0] for line in f]
    return class_names


def predict_image_class(image_path, model, preprocess, text, device):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        # image_features = model.encode_image(image)
        # text_features = model.encode_text(text)
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    return probs[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Change Detection Dataset Arguments')
    # Add arguments
    parser.add_argument('--src_path', type=str, default='/home/ps/HDD/zhaoyq_data/DATASET/WHUCD', help='Path to the source dataset')
    parser.add_argument('--split', nargs='+', type=str, default=[''], help='Split(s) to use (train, val, test)')
    parser.add_argument('--img_split', nargs='+', type=str, default=['A', 'B'], help='Image split(s) to use (A, B)')
    parser.add_argument('--model_name', type=str, default='ViT-B/16', help='Name of the model')
    parser.add_argument('--class_names_path', type=str, default='/home/ps/zhaoyq_files/changeclip/ChangeCLIP/tools/rscls.txt', help='Path to the class names file')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for model execution')
    parser.add_argument('--tag', type=str, default='56_vit16', help='Batch size for model execution')

    # Parse arguments
    args = parser.parse_args()

    # Access arguments
    src_path = args.src_path
    split = args.split
    img_split = args.img_split

    model_name = args.model_name
    class_names_path = args.class_names_path
    device = args.device

    model, preprocess = clip.load(model_name, device=device)
    class_names = read_class_names(class_names_path)
    text = clip.tokenize(class_names).to(device)

    for sp in split:
        for isp in img_split:
            image_folder_path = os.path.join(src_path, sp, isp)
            print('process path: {}'.format(image_folder_path))
            results = []

            for filename in tqdm(os.listdir(image_folder_path)):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    image_path = os.path.join(image_folder_path, filename)
                    probs = predict_image_class(image_path, model, preprocess, text, device)
                    sorted_probs = sorted(zip(class_names, probs.astype(np.float32)), key=lambda x: x[1], reverse=True)
                    result = {"image_path": image_path}
                    for cn, p in sorted_probs:
                        result[cn] = "{:.4f}".format(p)
                    results.append(result)

            with open(os.path.join(image_folder_path+'_clipcls_{}.json'.format(args.tag)), mode='w') as predictions_file:
                json.dump(results, predictions_file, indent=4, ensure_ascii=False)
