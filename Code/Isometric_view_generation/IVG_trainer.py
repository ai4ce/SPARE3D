from __future__ import print_function
import argparse
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import glob

import numpy as np
import cv2
from model import *
from Dataloader import *



parser = argparse.ArgumentParser()
parser.add_argument('--Training_dataroot', default="/home/wenyuhan/project/Train_dataset/Task_6_train",required=False, help='path to training dataset')
parser.add_argument('--Validating_dataroot', default="/home/wenyuhan/project/Train_dataset/Task_6_eval",required=False, help='path to validating dataset')
parser.add_argument('--batchSize', type=int, default=5, help='input batch size')
parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.0005, help='learning rate, default=0.0005')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta', type=float, default=0.5, help='beta, default=0.5')
parser.add_argument('--wd', type=float, default=0.05, help='weight decay, default=0.05')

parser.add_argument('--device', default='cuda:0', help='device')
parser.add_argument('--outf', default='/home/wenyuhan/final/IVG/', help='folder to output log')

opt = parser.parse_args()


inputChannelSize = 9
outputChannelSize= 3
ngf=64
ndf=64
BatchSize=opt.batchSize
imageSize=256
poolSize=50       
device = opt.device
#### creating model 
netD = D(inputChannelSize + outputChannelSize, ndf).to(device)
netG = G(inputChannelSize, outputChannelSize, ngf).to(device)

path_train=opt.Training_dataroot
path_eval=opt.Validating_dataroot
dataset_train=IVG_data(path_train,transform=True)
dataset_eval= IVG_data(path_eval,transform=True)
data_train = torch.utils.data.DataLoader(dataset_train,batch_size=BatchSize,shuffle=True)
data_eval=torch.utils.data.DataLoader(dataset_eval,batch_size=BatchSize,shuffle=False)

criterionBCE = nn.BCELoss().to(device)
criterionCAE = nn.L1Loss().to(device)
##### training hyperparameters
lambdaGAN = 1.0
lambdaIMG = 0.1
lrD=opt.lrD
lrG=opt.lrG
beta1=opt.beta
wd=opt.wd
niter=opt.niter
annealStart=0
annealEvery=400
sizePatchGAN=30
log_path=opt.outf
if os.path.exists(log_path)==False:
    os.makedirs(log_path)

# display parameter 
display=5
evalIter=100
trainLogger = open(opt.outf+'train.log' , 'w')

##### variable create
target= torch.FloatTensor(BatchSize, outputChannelSize, imageSize, imageSize)
input = torch.FloatTensor(BatchSize, inputChannelSize, imageSize, imageSize)
val_target= torch.FloatTensor(BatchSize, outputChannelSize, imageSize, imageSize)
val_input = torch.FloatTensor(BatchSize, inputChannelSize, imageSize, imageSize)
label_d = torch.FloatTensor(BatchSize)

imagePool = ImagePool(poolSize)

target, input, label_d = target.to(device), input.to(device), label_d.to(device)
val_target, val_input = val_target.to(device), val_input.to(device)

target = Variable(target)
input = Variable(input)
label_d = Variable(label_d)

val_iter = iter(data_eval)
data_val = val_iter.next()

val_input_cpu,val_target_cpu = data_val

### store validation images

val_target.resize_as_(val_target_cpu).copy_(val_target_cpu)
val_input.resize_as_(val_input_cpu).copy_(val_input_cpu)

vutils.save_image(val_target,opt.outf+ 'real_target.png', normalize=True)

#### optimizer 

optimizerD = optim.Adam(netD.parameters(), lr = lrD, betas = (beta1, 0.999), weight_decay=wd)
optimizerG = optim.Adam(netG.parameters(), lr = lrG, betas = (beta1, 0.999), weight_decay=0.0)


##### training loop 
ganIterations = 0

for epoch in range(niter):
    if epoch > annealStart:
        adjust_learning_rate(optimizerD, lrD, epoch, None, annealEvery)
        adjust_learning_rate(optimizerG, lrG, epoch, None, annealEvery)
    for i, data in enumerate(data_train, 0):
        netG.train()
        netD.train()
        input_cpu, target_cpu = data
        batch_size = target_cpu.size(0)
        target_cpu, input_cpu = target_cpu.to(device), input_cpu.to(device)
        target.data.resize_as_(target_cpu).copy_(target_cpu)
        with torch.no_grad():
            input.resize_as_(input_cpu).copy_(input_cpu)
        for p in netD.parameters(): 
            p.requires_grad = True 
        netD.zero_grad()
        real_label=1
        fake_label=0
        if i%2==1:
        	real_label=0
        	fake_label=1


        with torch.no_grad():
            label_d.resize_((batch_size, 1, sizePatchGAN, sizePatchGAN)).fill_(real_label)
        output = netD(torch.cat([target, input], 1)) # conditional
        errD_real = criterionBCE(output, label_d)
        errD_real.backward()
        D_x = output.data.mean()
        x_hat = netG(input)
        fake = x_hat.detach()
        fake = Variable(imagePool.query(fake.data))
        label_d.data.fill_(fake_label)
        output = netD(torch.cat([fake, input], 1)) # conditional
        errD_fake = criterionBCE(output, label_d)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step() # update parameters
        for p in netD.parameters(): 
            p.requires_grad = False
        netG.zero_grad() # start to update G

                # compute L_L1 (eq.(4) in the paper
        L_img_ = criterionCAE(x_hat, target)
        L_img = lambdaIMG * L_img_
        if lambdaIMG != 0: 
            L_img.backward(retain_graph=True) # in case of current version of pytorch
            #L_img.backward(retain_variables=True)
        # compute L_cGAN (eq.(2) in the paper
        real_label=1
        label_d.data.fill_(real_label)
        output = netD(torch.cat([x_hat, input], 1))
        errG_ = criterionBCE(output, label_d)
        errG = lambdaGAN * errG_ 
        if lambdaGAN != 0:
            errG.backward()
        D_G_z2 = output.data.mean()

        optimizerG.step()
        ganIterations += 1
        if ganIterations % display == 0:
            print('[%d/%d][%d/%d] L_D: %f L_img: %f L_G: %f D(x): %f D(G(z)): %f / %f'
          % (epoch, niter, i, len(data_train),
             errD.item(), L_img.item(), errG.item(), D_x, D_G_z1, D_G_z2))
             
            sys.stdout.flush()
            trainLogger.write('%d\t%f\t%f\t%f\t%f\t%f\t%f\n' % \
                        (i, errD.item(), errG.item(), L_img.item(), D_x, D_G_z1, D_G_z2))
            trainLogger.flush()
        if ganIterations % evalIter == 0:
            val_batch_output = torch.FloatTensor(val_target.size()).fill_(0)
            for idx in range(val_input.size(0)):
                single_img = val_input[idx,:,:,:].unsqueeze(0)
                with torch.no_grad():
                    netG.eval()
                    val_inputv = Variable(single_img)
                    x_hat_val = netG(val_inputv)
                    x_hat_val = x_hat_val.reshape(3,256,256)
             
                val_batch_output[idx,:,:,:].copy_(x_hat_val.data)
            vutils.save_image(val_batch_output, '%s/generated_epoch_%08d_iter%08d.png' % (opt.outf, epoch, ganIterations), normalize=True)
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
trainLogger.close()
