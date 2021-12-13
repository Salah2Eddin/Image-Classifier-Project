from image_processing import process_image

import numpy as np

import json

from torch import nn, optim
from torchvision import datasets, transforms, models
import torch


def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    
    # Recreate our model
    if checkpoint['arch'] == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif checkpoint['arch'] == 'alexnet':
        model = models.alexnet(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state'])
    model.class_to_idx = checkpoint['classes_to_indices']
    
    return model

def predict(image_path, model, gpu, categories_file="", topk=1):
    device = "cuda" if gpu else "cpu"
    # Processing to turn the image into a tensor
    np_image = process_image(image_path)
    image_tensor = torch.tensor(np_image)
    # The model takes data in batchs, so we are adding
    # one more dim to the tensor as if its a batch of one image
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.type(torch.FloatTensor)
    image_tensor = image_tensor.to(device)
    
    # Making sure our model is in the correct modes
    model.to(device)
    model.eval()
    with torch.no_grad():
        # Feedforward
        log_ps = model.forward(image_tensor)
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(topk)
        
        # Processing the result
        idx_to_class = {val: key for key, val in model.class_to_idx.items()}
        probs = np.array(top_p).tolist()[0]
        indexs = [idx_to_class[each] for each in np.array(top_class[0])]
        classes = name_from_indexs(indexs, categories_file) if categories_file else indexs
        
    return probs, classes


def name_from_indexs(indexs, names_json):
    with open(names_json, 'r') as f:
        cat_to_name = json.load(f)
    return [cat_to_name[index] for index in indexs]
    