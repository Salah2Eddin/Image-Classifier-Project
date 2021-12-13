import torch
from torch import nn
from torchvision import models, transforms
from training_functions import *
import argparse

parser = argparse.ArgumentParser(description="Creates a new model and trains it. Hyperparameters can be chosen using command line args")
parser.add_argument('data_dir', action='store', help='path to data for training and validation', type=str)
parser.add_argument('--save_dir', '-sd', action='store', default='',help='Path to save model checkpoint', type=str)
parser.add_argument('--learning_rate', '-lr', action='store', default=0.001, help='Model learning rate', type=float)
parser.add_argument('--epochs', '-e',action='store', default=3, help='Number of epochs to train', type=int)
parser.add_argument('--arch', '-a', action='store', default='vgg13', choices=['vgg13', 'alexnet'], help='Model Architecture')
parser.add_argument('--hidden_units', '-hu',action='store', default=1024, help='Model Architecture', type=int)
parser.add_argument('--gpu', '-g',action='store_true', default=False, help='Enable GPU Training')
args = parser.parse_args()

# to avoid problems
if args.save_dir == "":
    save_dir = ""
else:
    save_dir = args.save_dir + '/'

train_dir = args.data_dir + '/train'
valid_dir = args.data_dir + '/valid'

model = create_model(args.hidden_units, args.arch)
checkpoint = train_model(train_dir, valid_dir, model, args.epochs, args.learning_rate, args.arch, args.gpu)
torch.save(checkpoint, save_dir+"model.pth")