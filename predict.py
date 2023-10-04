from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms, models
from torch import nn, optim
from PIL import Image
import json
import seaborn as sb
import argparse
from validators import validate

def parse():
    parser = argparse.ArgumentParser(description = "Parser for prediction")
    parser.add_argument('image_path', help = 'Required argument of image path', type = str)
    parser.add_argument('model_path', help = 'Required argument of path to the checkpoint',  type = str)
    parser.add_argument('--top_k', help = 'Top K classes', default=5, type = int)
    parser.add_argument('--category_names', help ='Json file of classes', default = 'cat_to_name.json', type = str)

    parser.add_argument('--gpu', help = "Boolean value for using gpu or not(yes/no)", default = 'yes', type = str)

    args = parser.parse_args()
    return args

def set_device(args):
    if args["gpu"] == 'yes':
        args["device"] = 'cuda'
    else:
        args["device"] = 'cpu'

def process_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path)
    image = preprocess(image)

    return image

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    model = models.vgg16(pretrained=True)

    model.classifier = checkpoint['classifier']

    model.load_state_dict(checkpoint['state_dict'])

    class_to_idx = checkpoint['class_to_idx']

    return model, class_to_idx

def predict(args, model, class_to_idx, cat_to_name, topk=5):
    model.eval()
    if args["gpu"] == "yes":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
    with torch.no_grad():
        processed_image = process_image(args["image_path"])

        image_tensor = processed_image.unsqueeze(0).to(device)
        model.to(device)
        output = model.forward(image_tensor)
        probabilities = torch.exp(output)

        top_probs, top_indices = probabilities.topk(topk)
        top_probs = top_probs.squeeze().cpu().numpy()
        top_indices = top_indices.squeeze().cpu().numpy()

        idx_to_class = {val: key for key, val in class_to_idx.items()}
        top_labels = [idx_to_class[idx] for idx in top_indices]
        top_flowers = [cat_to_name[str(label)] for label in top_labels]

    return top_probs, top_labels, top_flowers

def get_data_from_json(json_file_path):
    with open(json_file_path, "r") as json_file:
        data = json.load(json_file)
    return data

def main():
    args = vars(parse())
    print(args)
    validate(args)
    set_device(args)
    model, class_to_idx  = load_checkpoint(args["model_path"])
    cat_to_name = get_data_from_json(args["category_names"])
    top_k = args["top_k"]
    top_p, top_labels, top_flowers = predict(args, model, class_to_idx, cat_to_name, topk=top_k)
    print('Top probabilities: ', top_p)
    print('Top labels: ', top_labels)
    print('Top flower names: ', top_flowers)

if __name__ == '__main__':
    main()
