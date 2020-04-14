from __future__ import print_function, division

import torch.nn as nn
import torch
import glob
import argparse
import os
import torch
import numpy as np
import cv2
import time
from model import *
from Dataloader import *




parser = argparse.ArgumentParser()
parser.add_argument('--Training_dataroot', default="/home/wenyuhan/project/Train_dataset/Task_5_train",required=False, help='path to training dataset')
parser.add_argument('--Validating_dataroot', default="/home/wenyuhan/project/Train_dataset/Task_5_eval",required=False, help='path to validating dataset')
parser.add_argument('--batchSize', type=int, default=70, help='input batch size')
parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.00002, help='learning rate, default=0.00002')
parser.add_argument('--device', default='cuda:0', help='device')
parser.add_argument('--model_type', default='vgg16', help='|vgg16| |resnet50| |Bagnet33|')
parser.add_argument('--outf', default='/home/wenyuhan/final/P2I_BC/', help='folder to output log')

opt = parser.parse_args()


device = opt.device
  
task_5_model=P2I_BC(opt.model_type).to(opt.device)



def train_model():
    epoch_loss = 0
    epoch_acc =0 
    batch_loss = 0
    batch_acc =0
    batch_loss_list=[]
    path=opt.Training_dataroot
    train_data=P2I_BC_data(path)
    data_train = torch.utils.data.DataLoader(train_data,batch_size=opt.batchSize, shuffle=True)
    task_5_model.train()
    for i, (input_1,input_2,input_3,input_4,Label,view_vector) in enumerate(data_train):
        optimizer.zero_grad()
        input_1,input_2,input_3,input_4,Label,view_vector=input_1.to(device),input_2.to(device),input_3.to(device),input_4.to(device),Label.to(device),view_vector.to(device)
        y_1 = task_5_model(input_1.float(),view_vector.float())
        y_2 = task_5_model(input_2.float(),view_vector.float())
        y_3 = task_5_model(input_3.float(),view_vector.float())
        y_4 = task_5_model(input_4.float(),view_vector.float())
        Y=torch.cat((y_1,y_2,y_3,y_4),axis=1)
       # output=nn.functional.softmax(Y,dim=1)
        loss = criterion(Y, Label.float())
        batch_loss += loss.item() * input_1.shape[0]
        batch_loss_list.append(loss.item()*input_1.shape[0]/len(train_data))
        loss.backward()
        optimizer.step()
        batch_acc += (Y.argmax(1) == Label.argmax(1)).sum().item()
    epoch_loss = batch_loss / len(train_data)
    epoch_acc =  batch_acc /len(train_data)
    
    return epoch_loss, epoch_acc, np.array(batch_loss_list)


def Eval():
    eval_loss = 0
    eval_acc = 0
    epoch_eval_loss=0
    epoch_eval_acc =0
    path=opt.Validating_dataroot
    eval_data=P2I_BC_data(path)
    data_eval = torch.utils.data.DataLoader(eval_data,batch_size=opt.batchSize, shuffle=True)
    with torch.no_grad():
        task_5_model.eval()
        for i, (input_1,input_2,input_3,input_4,Label,view_vector) in enumerate(data_eval):
            input_1,input_2,input_3,input_4,Label,view_vector=input_1.to(device),input_2.to(device),input_3.to(device),input_4.to(device),Label.to(device),view_vector.to(device)
            y_1 = task_5_model(input_1.float(),view_vector.float())
            y_2 = task_5_model(input_2.float(),view_vector.float())
            y_3 = task_5_model(input_3.float(),view_vector.float())
            y_4 = task_5_model(input_4.float(),view_vector.float())
            Y=torch.cat((y_1,y_2,y_3,y_4),axis=1)
          #  output=nn.functional.softmax(Y,dim=1)
            loss = criterion(Y, Label.float())
            eval_loss += loss.item()* input_1.shape[0]
            eval_acc += (Y.argmax(1) == Label.argmax(1)).sum().item()
        epoch_eval_loss = eval_loss / len(eval_data)
        epoch_eval_acc =eval_acc / len(eval_data)
    return epoch_eval_loss, epoch_eval_acc




N_EPOCHS = opt.niter
criterion = torch.nn.BCEWithLogitsLoss().to(device)
optimizer = torch.optim.Adam(task_5_model.parameters(), lr=opt.lr)


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

