import argparse

parser = argparse.ArgumentParser(description='Short sample app')

#cd ImageClassifier
#python train.py /home/workspace/ImageClassifier/floers
#python train.py /home/workspace/ImageClassifier/flowers --save_dir /home/workspace/ImageClassifier --learning_rate 0.5 --epochs 1

parser.add_argument('data_dir')
parser.add_argument("--save_dir", default="/home/workspace/ImageClassifier")
parser.add_argument("--learning_rate", default=0.01, type=float)
parser.add_argument("--hidden_units", default=512, type=int)
parser.add_argument("--epochs", default=1, type=int)
parser.add_argument("--arch", default='vgg16')

#--learning_rate 0.01 --hidden_units 512 --epochs 20

args = parser.parse_args()

save_dir = args.save_dir
data_dir = args.data_dir
print(f"learning_rate: {args.learning_rate}, hidden_units: {args.hidden_units}, epochs: {args.epochs}")
arch = args.arch

import torch
from torch import nn
from torch import optim
from workspace_utils import active_session
import json

from torchvision import datasets, transforms, models

train_transform = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_valid_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.ToTensor(),                                           
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

print(train_dir)
image_train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
image_test_dataset = datasets.ImageFolder(test_dir, transform=test_valid_transform)
image_valid_dataset = datasets.ImageFolder(valid_dir, transform=test_valid_transform)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(image_train_dataset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(image_test_dataset, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(image_valid_dataset, batch_size=64, shuffle=True)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

model = eval("models." + args.arch + "(pretrained=True)")

# Only train the classifier parameters, feature parameters are frozen
for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(nn.Linear(25088, args.hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(args.hidden_units, 102),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()


optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

print (device)

model.to(device)

epochs = 1
steps = 0
running_loss = 0
valid_loss = 0
print_every = 50
with active_session():
    for epoch in range(epochs):
        for inputs, labels in trainloader:        
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
    
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Valid loss: {running_loss/print_every:.3f}.. "
                      f"Valid loss: {valid_loss/len(testloader):.3f}.. "
                      f"Valid accuracy: {accuracy/len(testloader):.3f}")

#Move model back to CPU before saving checkpoint https://knowledge.udacity.com/questions/228583
model.to('cpu');

checkpoint = {
    'input_size' : 25088,
    'output_size' : 102,
    'class_to_idx' : image_train_dataset.class_to_idx,
    'classifier' : model.classifier,
    'state_dict' : model.state_dict()
}

torch.save(checkpoint,save_dir+'/commandline-checkpoint.pth')
print('complete')