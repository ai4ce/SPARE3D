from __future__ import print_function, division

from torchvision import datasets, models, transforms
import torch.nn as nn
import torch
import glob
from PIL import Image
import argparse
import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
import time
from matplotlib import pyplot as plt  
from model import *
from Dataloader import *




parser = argparse.ArgumentParser()
parser.add_argument('--Training_dataroot', default="/home/wenyuhan/project/Train_dataset/Task_4_train",required=False, help='path to training dataset')
parser.add_argument('--Validating_dataroot', default="/home/wenyuhan/project/Train_dataset/Task_4_eval",required=False, help='path to validating dataset')
parser.add_argument('--batchSize', type=int, default=70, help='input batch size')
parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.00002, help='learning rate, default=0.00002')
parser.add_argument('--device', default='cuda:0', help='device')
parser.add_argument('--model_type', default='vgg16', help='|vgg16| |resnet50| |Bagnet33|')
parser.add_argument('--outf', default='/home/wenyuhan/final/I2P/', help='folder to output log')

opt = parser.parse_args()


device = opt.device
  
task_4_model=I2P(opt.model_type).to(opt.device)



def train_model():
    epoch_loss = 0
    epoch_acc =0 
    batch_loss = 0
    batch_acc =0
    batch_loss_list=[]
    path=opt.Training_dataroot
    train_data=I2P_data(path)
    data_train = torch.utils.data.DataLoader(train_data,batch_size=opt.batchSize, shuffle=True)
    task_4_model.train()
    for i, (Input,Label) in enumerate(data_train):
        optimizer.zero_grad()
        Input,Label=Input.to(device),Label.to(device)
        y = task_4_model(Input.float())
        Label=Label.reshape(Input.shape[0])
        loss = criterion(y, Label)
        batch_loss += loss.item() * Input.shape[0]
        batch_loss_list.append(loss.item()*Input.shape[0]/len(train_data))
        loss.backward()
        optimizer.step()
        batch_acc += (y.argmax(1) == Label).sum().item()
    epoch_loss = batch_loss / len(train_data)
    epoch_acc =  batch_acc /len(train_data)
    
    return epoch_loss, epoch_acc, np.array(batch_loss_list)


def Eval():
    eval_loss = 0
    eval_acc = 0
    epoch_eval_loss=0
    epoch_eval_acc =0

    data_transforms=False
    path=opt.Validating_dataroot
    eval_data=I2P_data(path)
    data_eval = torch.utils.data.DataLoader(eval_data,batch_size=opt.batchSize, shuffle=True)
    with torch.no_grad():
        task_4_model.eval()
        for i, (Input,Label) in enumerate(data_eval):
            Input,Label=Input.to(device),Label.to(device)
            y = task_4_model(Input.float())
            Label=Label.reshape(Input.shape[0])
            loss = criterion(y, Label)
            eval_loss += loss.item()* Input.shape[0]
            eval_acc += (y.argmax(1) == Label).sum().item()
        epoch_eval_loss = eval_loss / len(eval_data)
        epoch_eval_acc =eval_acc / len(eval_data)
    return epoch_eval_loss, epoch_eval_acc




N_EPOCHS = opt.niter
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(task_4_model.parameters(), lr=opt.lr)


batch_loss_history=[]



log_path=opt.outf
if os.path.exists(log_path)==False:
    os.makedirs(log_path)



file=open(log_path+"/"+opt.model_type+"_Lr_"+str(opt.lr)+".txt","w")


for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss, train_acc,batch_list = train_model()
    valid_loss, valid_acc = Eval()
    batch_loss_history.append(batch_list)

    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60
    file.write('Epoch: %d' %(epoch + 1))
    file.write(" | time in %d minutes, %d seconds\n" %(mins, secs))
    file.write(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)\n')
    file.write(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)\n')
    file.write("\n")
    file.flush()
    # print(" | time in %d minutes, %d seconds\n" %(mins, secs))
    # print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)\n')
    # print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)\n')
    # print("\n")

file.close()
batch_loss_history=np.array(batch_loss_history)
batch_loss_history=np.concatenate(batch_loss_history,axis=0)
    
batch_loss_history=batch_loss_history.reshape(len(batch_loss_history))
np.save(log_path+"/"+opt.model_type+"_Lr_"+str(opt.lr)+".npy",batch_loss_history)
torch.save(task_1_model.state_dict(),log_path+"/"+opt.model_type+"_Lr_"+str(opt.lr)+".pth")

