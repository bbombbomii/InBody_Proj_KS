# Resnet code from EE488 [Week11]

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import numpy as np 
import random

from tqdm import tqdm
import time 
import scipy.io as sio 
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import mat73

torch.backends.cudnn.deterministic = True # Use cudnn as deterministic mode for reproducibility
torch.backends.cudnn.benchmark = False

################### Data loading
path = "D:/Cloud/OneDrive - kaist.ac.kr/InBody_Project/임상데이터/3. KAIST Korotkoff Sound/4. Dataset"
#path = r"C:\Users\Human\kaist.ac.kr\Bomi Lee - InBody_Project\임상데이터\3. KAIST Korotkoff Sound\4. Dataset"
os.chdir(path)

arr = mat73.loadmat('trainset_valid_v3.mat')
# arr = mat73.loadmat('trainset_diff_smaller_than_7_v2.mat')
train_data = arr['img_tot']
train_labels = arr['label_tot']
train_labels = np.reshape(train_labels, (1,train_labels.shape[0]))

arr = mat73.loadmat('testset_valid_v3.mat')
# arr = mat73.loadmat('testset_diff_smaller_than_7_v2.mat')
test_data = arr['img_tot']
test_labels = arr['label_tot']
test_labels = np.reshape(test_labels, (1,test_labels.shape[0]))

################### Dataset
class MyDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.FloatTensor(x_data)
        self.x_data = self.x_data.permute(2,0,1).unsqueeze(1)
        self.y_data = torch.LongTensor(y_data)
        self.y_data = self.y_data.permute(1,0).squeeze()
        self.len = self.y_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

trainset = MyDataset(train_data, train_labels)
testset = MyDataset(test_data, test_labels)

trainloader = DataLoader(trainset, batch_size=50, shuffle=True)
testloader = DataLoader(testset, batch_size=1, shuffle=False) # batch size 1로 수정.

# Check the format
# print(trainset[0][0].size())
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
# print(images.size())

def train(model, n_epoch, loader, optimizer, criterion, device="cpu"):
  model.train()
  for epoch in tqdm(range(n_epoch)):
    running_loss = 0.0
    for i, data in enumerate(loader, 0):
      images, labels = data
      images = images.to(device)
      labels = labels.to(device)
      optimizer.zero_grad()

      outputs = model(images)
      loss = criterion(input=outputs, target=labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
    print('Epoch {}, loss = {:.3f}'.format(epoch, running_loss/len(loader)))
  print('Training Finished')

def evaluate(model, loader, device="cpu"):
  model.eval()
  total=0
  correct=0
  with torch.no_grad():
    for data in loader:
      images, labels = data
      images = images.to(device)
      labels = labels.to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted==labels).sum().item()
  acc = 100*correct/total
  return acc

def reset_seed(seed):
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)

class ResNet(nn.Module):
  def __init__(self):
    super(ResNet, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
    self.norm1= nn.BatchNorm2d(32) # Batch normalization layer with channel size = 32.
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
    self.norm2= nn.BatchNorm2d(32)

    self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
    self.norm3= nn.BatchNorm2d(64)
    self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
    self.norm4= nn.BatchNorm2d(64)

    self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
    self.norm5= nn.BatchNorm2d(128)
    self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
    self.norm6= nn.BatchNorm2d(128)

    self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.avg_pool = nn.AdaptiveAvgPool2d(output_size = (1, 1))

    self.fc = nn.Linear(in_features=128, out_features=10)

  def forward(self, x):
    ######### Modify the forward function to utilize skip connection and batch normalization #########

    # Task 1. Add residual connections at conv2, conv4, and conv6 with the same number of input channels and output channels.
    # Task 2. Add the forward propagation through batch normalization layer
    x = self.conv1(x)
    x = self.norm1(x)
    x = F.relu(x)
    x_residual = x
    x = self.conv2(x) + x_residual
    x = self.norm2(x)
    x = F.relu(x)
    x = self.max_pool(x)

    x = self.conv3(x)
    x = self.norm3(x)
    x = F.relu(x)
    x_residual = x
    x = self.conv4(x)+ x_residual
    x = self.norm4(x)
    x = F.relu(x)
    x = self.max_pool(x)

    x = self.conv5(x)
    x = self.norm5(x)
    x = F.relu(x)
    x_residual = x
    x = self.conv6(x)+ x_residual
    x = self.norm6(x)
    x = F.relu(x)
    x = self.max_pool(x)

    x = self.avg_pool(x)
    x = x.view(-1, 128)
    x = self.fc(x)
    return x

reset_seed(0)
resnet_model = ResNet().to("cuda")
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=resnet_model.parameters(), lr=0.1, momentum=0.9)
train(model=resnet_model, n_epoch=10, loader=trainloader, optimizer=optimizer, criterion=criterion, device="cuda")


resnet_acc = evaluate(resnet_model, testloader, device="cuda")
print('ResNet Test accuracy: {:.2f}%'.format(resnet_acc))