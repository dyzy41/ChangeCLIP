import os
import json
import torch
import clip
import numpy as np


def clip_infer(image_pilA, image_pilB, model, preprocess):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = [
    "airport", "bare land", "basketball court", "beach", "bridge", "dense residential",
    "farmland", "forest", "freeway", "ground track field", "harbor", "island", "lake",
    "meadow", "mine", "mountain", "oil tank", "runway", "sea", "solar panel", "stadium",
    "tennis court", "terrace", "wetland", "commercial area", "cotton field", "industrial area",
    "railway", "interchange", "single-family residential", "highway", "intersection",
    "golf course", "desert", "river", "campus", "prairie", "building", "fertile land",
    "park", "parking lot", "square", "road", "tree", "cars", "shrubbery",
    "impermeable surface", "ship", "airplane", "pond", "church", "chaparral",
    "storage tanks", "snow land", "container", "cabin"
    ]
    text = clip.tokenize(class_names).to(device)
    imageA = preprocess(image_pilA).unsqueeze(0).to(device)
    with torch.no_grad():
        logits_per_image, logits_per_text = model(imageA, text)
        probsA = logits_per_image.softmax(dim=-1).cpu().numpy()

    imageB = preprocess(image_pilB).unsqueeze(0).to(device)
    with torch.no_grad():
        logits_per_image, logits_per_text = model(imageB, text)
        probsB = logits_per_image.softmax(dim=-1).cpu().numpy()

    sorted_probsA = sorted(zip(class_names, probsA.astype(np.float32)), key=lambda x: x[1], reverse=True)[:9]
    sorted_probsB = sorted(zip(class_names, probsB.astype(np.float32)), key=lambda x: x[1], reverse=True)[:9]
    jsonA = [i[0] for i in sorted_probsA]
    jsonB = [i[0] for i in sorted_probsB]
    return jsonA, jsonB

def init_clip():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, preprocess = clip.load('ViT-B/16', device=device)
    return model, preprocess



if __name__ == '__main__':
    clip_infer('/home/user/dsj_code/mmsegcd/ChangeCLIP/demo/demo.png')