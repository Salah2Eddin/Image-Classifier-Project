import numpy as np

from PIL import Image
import PIL

def process_image(image_path):
    # Open and resize image
    image = Image.open(image_path)
    image = image.resize((256, 256))
    
    # Image.crop takes a rect that selects the area to cut
    # To crop from center we calculate the positions like this
    width, height = image.size
    left = (width-224)/2
    top = (height-224)/2
    right = (width+224)/2
    bottom = (width+224)/2
    rect = ((left, top, right, bottom))
    image = image.crop(rect)
    
    np_image = np.array(image)
    
    # Normalization
    np_image = np_image/255
    color_mean = np.array([0.485, 0.456, 0.406])
    color_std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image-color_mean)/color_std
    
    # Make Color Channel the first Dim
    np_image = np_image.transpose((2, 0, 1))
    return np_image