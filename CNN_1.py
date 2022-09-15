# CNN code from EE488 [Week10]
# freshman이 수정함.
from tkinter import N
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
from tensorflow.keras.models import load_model
# Convolutional layer can be defined by torch.nn.Conv2d
# CNN needs number of input channel / number of output channel / size of kernel
import math

################### Data loading
# path = "D:/Cloud/OneDrive - kaist.ac.kr/InBody_Project/임상데이터/3. KAIST Korotkoff Sound/4. Dataset"
path = r"C:\Users\Human\kaist.ac.kr\Bomi Lee - InBody_Project\임상데이터\3. KAIST Korotkoff Sound\4. Dataset"
os.chdir(path)

arr = mat73.loadmat('trainset_valid_v3.mat') 
# arr = mat73.loadmat('trainset_diff_smaller_than_7_v2.mat')
train_data = arr['img_tot']
train_labels = arr['label_tot']
train_labels = np.reshape(train_labels, (1,train_labels.shape[0]))



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
trainloader = DataLoader(trainset, batch_size=50, shuffle=True)

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
        self.fc1 = nn.Linear(in_features=75264, out_features=1)

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
            with torch.no_grad():
                labels = np.reshape(labels, (labels.shape[0],1)) #input과 target size 맞춰주기

            loss = criterion(input=outputs.to(torch.float32), target=labels.to(torch.float32)) # tensor type float으로 바꿔주기
            loss.backward()
            
            optimizer.step()
            running_loss += loss.item()
        print('Epoch {}, loss = {:.3f}'.format(epoch, running_loss/len(loader)))
    print('Training Finished')
    PATH = r"C:\Users\Human\kaist.ac.kr\Bomi Lee - InBody_Project\임상데이터\3. KAIST Korotkoff Sound\5. Trained Model"
    torch.save(model, PATH + '/model_epoch50.pt')  # 전체 모델 저장


def evaluate(model, loader, device="cpu"): # 여기서 뭔가 문제가 있음. predicted가 계속 축적되는 원인이.
    model.eval()
    total=0
    correct=0
    with torch.no_grad():
        pred = []
        for data in loader: 
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # _, predicted = torch.max(outputs.data, 1) #dimension=1, 최대값의 class index를 return -> 0 return
            temp = np.reshape(outputs.data,(1))
            if temp.item() >= 1:
                pred.append(1) # predicted probability values that exceed 1 are replaced with 1.
            elif temp.item() < 0:
                pred.append(0)
            else:
                pred.append(temp.item())
    return pred


cnn_model = CNN()
optimizer = optim.SGD(params = cnn_model.parameters(), lr=0.002, momentum=0.9) 
criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()

########## Training
# train(model=cnn_model, n_epoch=50, loader=trainloader, optimizer=optimizer, criterion=criterion)
PATH = r"C:\Users\Human\kaist.ac.kr\Bomi Lee - InBody_Project\임상데이터\3. KAIST Korotkoff Sound\5. Trained Model\Save"
model = torch.load(PATH + '/model_epoch50.pt') 
path = r"C:\Users\Human\kaist.ac.kr\Bomi Lee - InBody_Project\임상데이터\3. KAIST Korotkoff Sound\4. Dataset"
os.chdir(path)

def test(index):
    arr = mat73.loadmat('test_'+str(index)+'.mat')
    test_data = arr['img']
    test_labels = arr['label']
    test_time = arr['time']
    test_pressure = arr['pressure']
    sbp_true = arr['sbp']
    dbp_true = arr['dbp']
    tsbp_true = arr['t_sbp']
    tdbp_true = arr['t_dbp']
    t = test_time
    test_labels = np.reshape(test_labels, (1,test_labels.shape[0]))
    testset = MyDataset(test_data, test_labels)
    testloader = DataLoader(testset, batch_size=1, shuffle=False) # batch size 1로 수정.
    predicted = evaluate(model, testloader)
    plt.figure()
    plt.plot(t,predicted,tsbp_true,1,'ro',tdbp_true,1,'go') # 가로축에 t 추가하기!
    plt.ylabel('predicted probability')
    ############ SBP, DBP Estimation
    
    N = len(t)-1
    MSEsbp_reci = []
    MSEdbp_reci = []
    for n in range(5,N-3):
        # MSE for the SBP
        tsbp    = t[n]
        tdbp = t[N]
        sum = 0
        for n1 in range(-5,4):
            # calculate y(=label curve)
            if t[n+n1] < tsbp - 1:
                y = 0
            elif tsbp - 1 <= t[n+n1] < tsbp:
                y = t[n+n1] - tsbp + 1
            elif tsbp <= t[n+n1] < tdbp - 1:
                y = 1
            elif tdbp -1 <= t[n+n1] < tdbp + 1:
                y = -1/2*t[n+n1] + 1/2*tdbp + 1/2
            else:
                y = 0
            sum = sum + (predicted[n+n1]-float(y))**2
        MSEsbp_reci.append(1/(sum/10))
        # MSE for the DBP
        tsbp = t[1]
        tdbp = t[n]
        sum = 0
        for n1 in range(-5,4):
            # calculate y(=label curve)
            if t[n+n1] < tsbp - 1:
                y = 0
            elif tsbp - 1 <= t[n+n1] < tsbp:
                y = t[n+n1] - tsbp + 1
            elif tsbp <= t[n+n1] < tdbp - 1:
                y = 1
            elif tdbp -1 <= t[n+n1] < tdbp + 1:
                y = -1/2*t[n+n1] + 1/2*tdbp + 1/2
            else:
                y = 0
            sum = sum + (predicted[n+n1]-float(y))**2
        MSEdbp_reci.append(1/(sum/10))
    # zero padding (5 in front, 4 behind)
    MSEsbp_reci_plt = np.concatenate(([0,0,0,0,0],MSEsbp_reci,[0,0,0,0]))
    MSEdbp_reci_plt = np.concatenate(([0,0,0,0,0],MSEdbp_reci,[0,0,0,0]))

    plt.figure()
    plt.plot(t,MSEsbp_reci_plt,'r--',t,MSEdbp_reci_plt,'g--',tsbp_true,1,'ro',tdbp_true,1,'go')
    plt.ylabel('1/MSE')
    plt.show()
    max_index = MSEsbp_reci.index(max(MSEsbp_reci))
    sbp_predicted = test_pressure[max_index]
    # tsbp_predicted = t[max_index]
    max_index = MSEdbp_reci.index(max(MSEdbp_reci))
    dbp_predicted = test_pressure[max_index]
    sbp_err = sbp_predicted - sbp_true
    dbp_err = dbp_predicted - dbp_true
    
    return sbp_err, dbp_err

test_indices = [21,10,50,24,34] # 09-14 PM 01:01
sbp_mse = 0
dbp_mse = 0
sbp_mean = 0
dbp_mean = 0
for index in test_indices:
    sbp_err, dbp_err = test(index)
    print("sbp_err: ", sbp_err)
    print("dbp_err: ", dbp_err)
    sbp_mean = sbp_mean + sbp_err
    dbp_mean = dbp_mean + dbp_err
    sbp_mse = sbp_mse + sbp_err**2
    dbp_mse = dbp_mse + dbp_err**2
sbp_mse = sbp_mse/len(test_indices)
dbp_mse = dbp_mse/len(test_indices)
print("sbp_rmse: ", math.sqrt(sbp_mse))
print("dbp_rmse: ", math.sqrt(dbp_mse))
print("sbp_mean: ", sbp_mean/len(test_indices))
print("dbp_mean: ", dbp_mean/len(test_indices))