import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import cv2
from torchvision import transforms 
from torchvision.datasets import ImageFolder
from torch.utils.data import  DataLoader 
from torchvision import models 
from torch import optim 
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

train_path =  "D:\\dataset_sym_Light"
classes = os.listdir(train_path)

class_weight = [] 
plt.subplots_adjust(hspace=0.5, wspace=0.5)

for c in classes : 
    class_path = os.path.join(train_path , c) 
    img_path = os.path.join ( class_path , os.listdir(class_path)[0]) 
    class_weight.append (len(os.listdir(class_path)) )
    img = cv2.imread (img_path) 
    plt.subplot(5 , 5 , classes.index(c)+1 ) 
    plt.title (c)
    plt.imshow(img)
plt.show()

plt.bar(classes, class_weight)    
plt.xlabel("class")
plt.ylabel("number of images" )

train_transform = transforms.Compose([
        transforms.Resize ( (150 , 150) ),
        #transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
test_transform =   transforms.Compose([
        transforms.Resize ( (150 , 150) ),        
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def load():
    path = "D:\\dataset_sym_Light"
    dataset = ImageFolder(path)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_dataset.dataset.transform = train_transform
    test_dataset.dataset.transform = test_transform
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, test_loader

train, test = load()
model = models.vgg19(pretrained=True)
for p in model.parameters():
    p.requires_grad = False

model.classifier = nn.Sequential(
    nn.Linear(25088, 2048),
    nn.ReLU(),
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Dropout(p=0.6),
    nn.Linear(512, 22),
    nn.LogSoftmax(dim=1)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = optim.Adam( model.parameters() , lr = 0.00001)

def accuracy(data):
    with torch.no_grad():
        model.eval()
        t = 0
        c = 0
        for x, y in data:
            x = x.to(device)
            y = y.to(device)

            preds = model(x)

            for i in range(preds.shape[0]):
                if torch.argmax(preds[i]) == y[i]:
                    c += 1
                t += 1
    return c / t

epochs = 5
step = 0
total_loss = 0 
train_acc = []
test_acc = []
for e in range(epochs):
    for x, y in train:

        x = x.to(device)
        y = y.to(device)

        model.zero_grad()

        output = model(x)
        loss = F.nll_loss(output, y)

        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        step += 1

plt.figure() 
plt.plot(train_acc , label = "train")
plt.plot(test_acc , label = "test" )
plt.legend()