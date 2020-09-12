from torchvision import datasets, transforms, models

#cd ImageClassifier
import torch
from torch import nn
import argparse
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser(description='Short sample app')
parser.add_argument('image_filename')
parser.add_argument("checkpoint_filename", default="/home/workspace/ImageClassifier/commandline-checkpoint.pth")
parser.add_argument("--gpu", action="store_true", default=False)
parser.add_argument("--top_k", default=1, type=int)
parser.add_argument("--category_names", default="use checkpoint")


args = parser.parse_args()

image_filename = args.image_filename
checkpoint_filename = args.checkpoint_filename
gpu_option = args.gpu

print(gpu_option)

#cd ImageClassifier
#example image file /home/workspace/ImageClassifier/flowers/valid/1/image_06739.jpg
#command format: python predict.py /path/to/image checkpoint
#sample command: python predict.py /home/workspace/ImageClassifier/flowers/valid/1/image_06739.jpg commandline-checkpoint.pth
#sample command 2: python predict.py /home/workspace/ImageClassifier/flowers/valid/52/image_04215.jpg commandline-checkpoint.pth
# python predict.py /home/workspace/ImageClassifier/flowers/valid/12/image_03997.jpg commandline-checkpoint.pth
# python predict.py /home/workspace/ImageClassifier/flowers/valid/17/image_03829.jpg commandline-checkpoint.pth

device= torch.device("cuda" if torch.cuda.is_available() and gpu_option else "cpu")
print (device)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained=True)
    model.classifier = nn.Sequential(nn.Linear(25088, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 102),
                                 nn.LogSoftmax(dim=1))
    model.load_state_dict(checkpoint['state_dict'])        
    model.class_to_idx = checkpoint['class_to_idx']
    return model
    
model = load_checkpoint(args.checkpoint_filename)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    
    
    width, height = pil_image.size
    if width < height:
        height = float(height) * 256 / float(width)
        width = 256
    else:
        width = float(width) * 256 / float(height)
        height = 256
    
    pil_image.show()
    
    print(f"box before crop: {pil_image.getbbox()}")
    
    size = 256, 256
    pil_image.thumbnail(size)
    #left, upper, right, lower
    
    print(f"box after resize, before crop: {pil_image.getbbox()} height {height} width {width}")
    
    
    pil_image.show()
    
    left = (width - 224) / 2
    right = left + 224
    upper = (height - 224) / 2
    lower = upper + 224
    
    print(f"left: {left} right: {right} upper:{upper} lower:{lower}")
    
    pil_image = pil_image.crop((left,upper,right,lower))
    print(pil_image.getbbox())

    np_image = np.array(pil_image) / 255 #Color channels of images are encoded 0-255, model expected floats 0-1. Y
    
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std #Subtract the means from each color channel, then divide by the standard deviation.
    
    np_image_transpose = np_image.transpose((2, 0, 1))
    
    toFloatTensor  = torch.from_numpy(np_image_transpose).type(torch.FloatTensor)
    
    #https://knowledge.udacity.com/questions/164455
    return(toFloatTensor)

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    print(f"image_path: {image_path}")
    # TODO: Implement the code to predict the class from an image file
    image = process_image(image_path)
    if device == 'cuda':        
        model = model.cuda()
    image = image.unsqueeze(0)
    #unsqueeze from here https://discuss.pytorch.org/t/how-to-classify-single-image-using-loaded-net/1411
    #image_path = image_path.unsqueeze(0)  
    log_ps = model(image)
    ps = torch.exp(log_ps)
    top_p, top_class = ps.topk(topk)
    print(f"top_p: {top_p}")
    #source https://stackoverflow.com/questions/34097281/how-can-i-convert-a-tensor-into-a-numpy-array-in-tensorflow
    top_p = top_p.cpu()
    top_p = top_p.detach().numpy()
    top_class = top_class.cpu()
    top_class = top_class.detach().numpy()
    return top_p[0], top_class[0]

model = load_checkpoint(checkpoint_filename)

import json
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

probs, classes= predict(args.image_filename, model, topk=args.top_k)
print(f"probs: {probs} classes {classes}")
print([cat_to_name[str(i)] for i in classes])