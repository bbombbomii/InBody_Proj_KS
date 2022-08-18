# CNN code from EE488 [Week10]
# freshman이 수정함.
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import mat73
# from torchsummary import summary
from tqdm import tqdm
# from sklearn.preprocessing import StandardScaler
import time
import matplotlib.pyplot as plt

import scipy.io as sio 
from torch.utils.data import DataLoader, Dataset
import numpy as np
# Convolutional layer can be defined by torch.nn.Conv2d
# CNN needs number of input channel / number of output channel / size of kernel


################### Data loading
path = "D:/Cloud/OneDrive - kaist.ac.kr/InBody_Project/임상데이터/3. KAIST Korotkoff Sound/4. Dataset"
#path = "C:\Users\Human\kaist.ac.kr\Bomi Lee - InBody_Project\임상데이터\3. KAIST Korotkoff Sound\4. Dataset"
os.chdir(path)

# arr = mat73.loadmat('trainset_valid.mat')
arr = mat73.loadmat('trainset_diff_smaller_than_7_v2.mat')
train_data = arr['img_tot']
train_labels = arr['label_tot']
train_labels = np.reshape(train_labels, (1,train_labels.shape[0]))

# arr = mat73.loadmat('testset_valid.mat')
arr = mat73.loadmat('testset_diff_smaller_than_7_v2.mat')
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
testloader = DataLoader(testset, batch_size=50, shuffle=False)

# Check the format
# print(trainset[0][0].size())
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
# print(images.size())

#################### CNN Implementation
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5,5))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5,5))
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5,5))
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=75264, out_features=2)

    def forward(self, x):
        batchsize = x.size(0)
        x = self.conv1(x)
        x = self.pool1(F.relu(x))
        x = self.conv2(x)
        x = self.pool2(F.relu(x))
        x = self.conv3(x)
        x = self.pool3(F.relu(x))
        x = x.view(batchsize, -1)
        out = self.fc1(x)
        return out

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
            correct += (predicted == labels).sum().item()
    acc = 100*correct/total
    return acc


cnn_model = CNN()
optimizer = optim.SGD(params = cnn_model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
train(model=cnn_model, n_epoch=10, loader=trainloader, optimizer=optimizer, criterion=criterion)
acc = evaluate(cnn_model, testloader)
print('Test accuracy: {:.2f}%'.format(acc))



    
# # Functions for visualizing features of CNN.
# def vis_feat(f1, f2, f3, inp):
#     l = [f1, f2, f3]
#     fig, axes = plt.subplots(4, 3, figsize=(10, 5))
#     plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, wspace = 0, hspace = 0.1)
#     ax = axes[0,0]
#     ax.imshow(inp[0], cmap='gray_r')
#     ax.set_title('Original image')    
#     ax.axis('off') 

#     for i in range(2):
#         ax = axes[0,i+1]
#         ax.axis('off')  

#     for i in range(3):
#         r = 1 + i // 3
#         c = i % 3
#         ax = axes[r, c]             
#         ax.imshow(f1[i], cmap='gray_r')
#         if i == 0:
#             ax.set_title('Output from conv1')      
#         ax.axis('off')
#         if i < 3:
#             ax = axes[1, i]
#             ax.axis('off')

#     for i in range(3):
#         r = 2 + i // 3
#         c = i % 3
#         ax = axes[r, c]             
#         ax.imshow(f2[i], cmap='gray_r')
#         if i == 0:
#             ax.set_title('Output from conv2')      
#         ax.axis('off')
#         if i < 3:
#             ax = axes[2, i]
#             ax.axis('off')

#     for i in range(3):
#         r = 3 + i // 3
#         c = i % 3
#         ax = axes[r, c]             
#         ax.imshow(f3[i], cmap='gray_r')
#         if i == 0:
#             ax.set_title('Output from conv3')      
#         ax.axis('off')
#         if i < 3:
#             ax = axes[3, i]
#             ax.axis('off')

#     plt.xticks([]), plt.yticks([])
#     plt.show()

# def vis(model, loader):
#     with torch.no_grad():
#         for i, data in enumerate(loader, 0):
#             images, labels = data
#             f1 = model.conv1(images)
#             f2 = model.conv2(model.pool1(F.relu(f1)))
#             f3 = model.conv3(model.pool2(F.relu(f2)))
#             vis_feat(f1[0], f2[0], f3[0], images[0])
#             break

# vis(cnn_model.cpu(), trainloader)