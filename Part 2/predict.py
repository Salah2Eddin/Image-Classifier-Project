from prediction_functions import load_checkpoint, predict
import torch
import argparse

parser = argparse.ArgumentParser(description="Predict image class")
parser.add_argument('image_dir', action='store', help='Image Path', type=str)
parser.add_argument('--checkpoint_path', '-cp', action='store', default='checkpoint.pth',help='path to checkpoint.pth file', type=str)
parser.add_argument('--topk', '-tk', action='store', default=1, help='K most likely classes', type=int)
parser.add_argument('--category_names', '-cn',action='store', default="", help='Use a Mapping of categories to real name', type=str)
parser.add_argument('--gpu', '-g',action='store_true', default=False, help='Enable GPU Training')
args = parser.parse_args()

model = load_checkpoint(args.checkpoint_path)
probs, categories = predict(args.image_dir, model, args.gpu, args.category_names, topk=args.topk)

print("Probability || Class")
for prob, category in zip(probs, categories):
    print(prob, category, sep='||')
