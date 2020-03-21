from __future__ import print_function, division

from torchvision import datasets, models, transforms
import torch.nn as nn
import torch
import glob
from PIL import Image

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
import bagnets.pytorchnet


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        dim=(new_h, new_w)

        Img= cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
      

        return Img


class ThreeV2I_BC(nn.Module):

    def __init__(self,model_type=None):
        super(ThreeV2I_BC, self).__init__()
        if model_type=="Bagnet33":
            self.model = bagnets.pytorchnet.bagnet33(pretrained=False)
            self.model.conv1=torch.nn.Conv2d(in_channels=12, out_channels=64, 
                               kernel_size=1, stride=1,padding=0,bias=False)
            num_ftrs = self.model.fc.in_features
            self.model.fc=torch.nn.Linear(num_ftrs, 1)
        if model_type=="vgg16":
            self.model = models.vgg16(pretrained=False)
            self.model.features[0]=torch.nn.Conv2d(in_channels=12, out_channels=64, 
                               kernel_size=(3,3), stride=(1,1),padding=(1,1))
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6]=torch.nn.Linear(num_ftrs, 1)
        if model_type=="resnet50":
            self.model = models.resnet50(pretrained=False)
            self.model.conv1=torch.nn.Conv2d(in_channels=12, out_channels=64, 
                               kernel_size=(3,3), stride=(1,1),padding=(1,1))
            num_ftrs = self.model.fc.in_features
            self.model.fc=torch.nn.Linear(num_ftrs, 1)
    def forward(self, x):
        y = self.model(x)
        return y


class ThreeV2I_ML(nn.Module):

    def __init__(self,model_type=None):
        super(ThreeV2I_ML, self).__init__()

        #### model for extracting 3v features
        
        if model_type=="vgg16":
            self.model_3V = models.vgg16(pretrained=False)
            self.model_3V.features[0]=torch.nn.Conv2d(in_channels=9, out_channels=64, 
                               kernel_size=(3,3), stride=(1,1),padding=(1,1))
            num_ftrs = self.model_3V.classifier[6].in_features
            self.model_3V.classifier[6]=torch.nn.Linear(num_ftrs, 128)
            ### model for extracting orthogonal view features
            self.model_ortho = models.vgg16(pretrained=False)
            self.model_ortho.features[0]=torch.nn.Conv2d(in_channels=3, out_channels=64, 
                               kernel_size=(3,3), stride=(1,1),padding=(1,1))
            self.model_ortho.classifier[6]=torch.nn.Linear(num_ftrs, 128)

    

        if model_type=="Bagnet33":
            self.model_3V = bagnets.pytorchnet.bagnet33(pretrained=False)
            self.model_3V.conv1=torch.nn.Conv2d(in_channels=9, out_channels=64, 
                               kernel_size=1, stride=1,padding=0,bias=False)
            num_ftrs = self.model_3V.fc.in_features
            self.model_3V.fc=torch.nn.Linear(num_ftrs, 128)
            self.model_ortho = bagnets.pytorchnet.bagnet33(pretrained=False)
            self.model_ortho.conv1=torch.nn.Conv2d(in_channels=3, out_channels=64, 
                               kernel_size=1, stride=1,padding=0,bias=False)
            self.model_ortho.fc=torch.nn.Linear(num_ftrs, 128)
        

        if model_type=="resnet50":
            self.model_3V = models.resnet50(pretrained=False)
            self.model_3V.conv1=torch.nn.Conv2d(in_channels=9, out_channels=64, 
                               kernel_size=(3,3), stride=(1,1),padding=(1,1))
            num_ftrs = self.model_3V.fc.in_features
            self.model_3V.fc=torch.nn.Linear(num_ftrs, 128)

            self.model_ortho = models.resnet50(pretrained=False)
            self.model_ortho.conv1=torch.nn.Conv2d(in_channels=3, out_channels=64, 
                               kernel_size=(3,3), stride=(1,1),padding=(1,1))
            
            self.model_ortho.fc=torch.nn.Linear(num_ftrs, 128)
    def forward(self, x_3V,x_ortho):
        feature_3v = self.model_3V(x_3V)
        feature_ortho = self.model_ortho(x_ortho)
        distance=nn.functional.pairwise_distance(feature_3v,feature_ortho,2,keepdim=True)
        return distance



class I2P(nn.Module):

    def __init__(self,model_type=None):
        super(I2P, self).__init__()
        if model_type=="vgg16":
            self.model = models.vgg16(pretrained=False)
            self.model.features[0]=torch.nn.Conv2d(in_channels=12, out_channels=64, 
                               kernel_size=(3,3), stride=(1,1),padding=(1,1))
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6]=torch.nn.Linear(num_ftrs, 4)
        if model_type=="resnet50":
            self.model = models.resnet50(pretrained=False)
            self.model.conv1=torch.nn.Conv2d(in_channels=12, out_channels=64, 
                               kernel_size=(3,3), stride=(1,1),padding=(1,1))
            num_ftrs = self.model.fc.in_features
            self.model.fc=torch.nn.Linear(num_ftrs, 4)
        
        if model_type=="Bagnet33":
            self.model = bagnets.pytorchnet.bagnet33(pretrained=False)
            self.model.conv1=torch.nn.Conv2d(in_channels=12, out_channels=64, 
                               kernel_size=1, stride=1,padding=0,bias=False)
            num_ftrs = self.model.fc.in_features
            self.model.fc=torch.nn.Linear(num_ftrs, 4)
    def forward(self, x):
        y = self.model(x)
        return y




class P2I_BC(nn.Module):

    def __init__(self,model_type=None):
        super(P2I_BC, self).__init__()
        
        if model_type=="Bagnet33":
            self.model = bagnets.pytorchnet.bagnet33(pretrained=False)
            self.model.conv1=torch.nn.Conv2d(in_channels=12, out_channels=64, 
                               kernel_size=1, stride=1,padding=0,bias=False)
            num_ftrs = self.model.fc.in_features
            self.model.fc=torch.nn.Linear(num_ftrs, 128)
            self.fc = nn.Linear(128+8, 1)

        if model_type=="vgg16":
            self.model = models.vgg16(pretrained=False)
            self.model.features[0]=torch.nn.Conv2d(in_channels=12, out_channels=64, 
                               kernel_size=(3,3), stride=(1,1),padding=(1,1))
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6]=torch.nn.Linear(num_ftrs, 128)
            self.fc = nn.Linear(128+8, 1)

        if model_type=="resnet50":
            self.model = models.resnet50(pretrained=False)
            self.model.conv1=torch.nn.Conv2d(in_channels=12, out_channels=64, 
                               kernel_size=(3,3), stride=(1,1),padding=(1,1))
            num_ftrs = self.model.fc.in_features
            self.model.fc=torch.nn.Linear(num_ftrs, 128)
            self.fc = nn.Linear(128+8, 1)
    def forward(self, x,view_vector):
        y = self.model(x)
        y= self.fc(torch.cat((y,view_vector),axis=1))

        return y


class P2I_ML(nn.Module):

    def __init__(self,model_type=None):
        super(P2I_ML, self).__init__()
        
        #### model for extracting 3v features 
        if model_type=="vgg16":
            self.model_3V = models.vgg16(pretrained=False)
            self.model_3V.features[0]=torch.nn.Conv2d(in_channels=9, out_channels=64, 
                               kernel_size=(3,3), stride=(1,1),padding=(1,1))
            num_ftrs = self.model_3V.classifier[6].in_features
            self.model_3V.classifier[6]=torch.nn.Linear(num_ftrs, 128)
            self.fc_3V = nn.Linear(128+8, 50)
            ### model for extracting orthogonal view features
            self.model_ortho = models.vgg16(pretrained=False)
            self.model_ortho.features[0]=torch.nn.Conv2d(in_channels=3, out_channels=64, 
                               kernel_size=(3,3), stride=(1,1),padding=(1,1))
            self.model_ortho.classifier[6]=torch.nn.Linear(num_ftrs, 50)

        if model_type=="resnet50":
            #### model for extracting 3v features 
            self.model_3V = models.resnet50(pretrained=False)
            self.model_3V.conv1=torch.nn.Conv2d(in_channels=9, out_channels=64, 
                               kernel_size=(3,3), stride=(1,1),padding=(1,1))
            num_ftrs = self.model_3V.fc.in_features
            self.model_3V.fc=torch.nn.Linear(num_ftrs, 128)
            self.fc_3V = nn.Linear(128+8, 50)

        ### model for extracting orthogonal view features
            self.model_ortho = models.resnet50(pretrained=False)
            self.model_ortho.conv1=torch.nn.Conv2d(in_channels=3, out_channels=64,
                                   kernel_size=(3,3), stride=(1,1),padding=(1,1))
            self.model_ortho.fc=torch.nn.Linear(num_ftrs, 50)

        if model_type=="Bagnet33":
            self.model_3V = bagnets.pytorchnet.bagnet33(pretrained=False)
            self.model_3V.conv1=torch.nn.Conv2d(in_channels=9, out_channels=64, 
                               kernel_size=1, stride=1,padding=0,bias=False)
            num_ftrs = self.model_3V.fc.in_features
            self.model_3V.fc=torch.nn.Linear(num_ftrs, 128)
            self.fc_3V = nn.Linear(128+8, 50)

            self.model_ortho = bagnets.pytorchnet.bagnet33(pretrained=False)
            self.model_ortho.conv1=torch.nn.Conv2d(in_channels=3, out_channels=64, 
                               kernel_size=1, stride=1,padding=0,bias=False)
            self.model_ortho.fc=torch.nn.Linear(num_ftrs, 50) 
        
        
    def forward(self, x_3V,x_ortho,view_vector):
        feature_3v = self.model_3V(x_3V)
        feature_3v= self.fc_3V(torch.cat((feature_3v,view_vector),axis=1))
        feature_ortho = self.model_ortho(x_ortho)
        distance=nn.functional.pairwise_distance(feature_3v,feature_ortho,2,keepdim=True)
        return distance





#################### Isometric view generation model 

class ImagePool:
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        if pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, image):
        if self.pool_size == 0:
            return image
        if self.num_imgs < self.pool_size:
            self.images.append(image.clone())
            self.num_imgs += 1
            return image
        else:
            if np.random.uniform(0,1) > 0.5:
                random_id = np.random.randint(self.pool_size, size=1)[0]
                tmp = self.images[random_id].clone()
                self.images[random_id] = image.clone()
                return tmp
            else:
                return image


def adjust_learning_rate(optimizer, init_lr, epoch, factor, every):
  #import pdb; pdb.set_trace()
    lrd = init_lr / every
    old_lr = optimizer.param_groups[0]['lr']
   # linearly decaying lr
    lr = old_lr - lrd
    if lr < 0: lr = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr




###### model module 



def blockUNet(in_c, out_c, name, transposed=False, bn=True, relu=True, dropout=False):
    block = nn.Sequential()
    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    if not transposed:
        block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False))
    else:
        block.add_module('%s_tconv' % name, nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False))
    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if dropout:
        block.add_module('%s_dropout' % name, nn.Dropout2d(0.5, inplace=True))
    return block

class G(nn.Module):
    def __init__(self, input_nc, output_nc, nf):
        super(G, self).__init__()

        # input is 256 x 256
        layer_idx = 1
        name = 'layer%d' % layer_idx
        layer1 = nn.Sequential()
        layer1.add_module(name, nn.Conv2d(input_nc, nf, 4, 2, 1, bias=False))

        # input is 128 x 128
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer2 = blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False)
        # input is 64 x 64
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer3 = blockUNet(nf*2, nf*4, name, transposed=False, bn=True, relu=False, dropout=False)
        # input is 32
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer4 = blockUNet(nf*4, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
        # input is 16
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer5 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
        # input is 8
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer6 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
        # input is 4
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer7 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
        # input is 2 x  2
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer8 = blockUNet(nf*8, nf*8, name, transposed=False, bn=False, relu=False, dropout=False)

        ## NOTE: decoder
        # input is 1
        name = 'dlayer%d' % layer_idx
        d_inc = nf*8
        dlayer8 = blockUNet(d_inc, nf*8, name, transposed=True, bn=True, relu=True, dropout=True)

        #import pdb; pdb.set_trace()
        # input is 2
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        d_inc = nf*8*2
        dlayer7 = blockUNet(d_inc, nf*8, name, transposed=True, bn=True, relu=True, dropout=True)
        # input is 4
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        d_inc = nf*8*2
        dlayer6 = blockUNet(d_inc, nf*8, name, transposed=True, bn=True, relu=True, dropout=True)
        # input is 8
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        d_inc = nf*8*2
        dlayer5 = blockUNet(d_inc, nf*8, name, transposed=True, bn=True, relu=True, dropout=False)
        # input is 16
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        d_inc = nf*8*2
        dlayer4 = blockUNet(d_inc, nf*4, name, transposed=True, bn=True, relu=True, dropout=False)
        # input is 32
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        d_inc = nf*4*2
        dlayer3 = blockUNet(d_inc, nf*2, name, transposed=True, bn=True, relu=True, dropout=False)
        # input is 64
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        d_inc = nf*2*2
        dlayer2 = blockUNet(d_inc, nf, name, transposed=True, bn=True, relu=True, dropout=False)
        # input is 128
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer1 = nn.Sequential()
        d_inc = nf*2
        dlayer1.add_module('%s_relu' % name, nn.ReLU(inplace=True))
        dlayer1.add_module('%s_tconv' % name, nn.ConvTranspose2d(d_inc, output_nc, 4, 2, 1, bias=False))
        dlayer1.add_module('%s_tanh' % name, nn.Tanh())

        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4
        self.layer5 = layer5
        self.layer6 = layer6
        self.layer7 = layer7
        self.layer8 = layer8
        self.dlayer8 = dlayer8
        self.dlayer7 = dlayer7
        self.dlayer6 = dlayer6
        self.dlayer5 = dlayer5
        self.dlayer4 = dlayer4
        self.dlayer3 = dlayer3
        self.dlayer2 = dlayer2
        self.dlayer1 = dlayer1
    def forward(self, x):
      
        out1 = self.layer1(x)
    
        out2 = self.layer2(out1)
     
        out3 = self.layer3(out2)
      
        out4 = self.layer4(out3)
       
        out5 = self.layer5(out4)
      
        out6 = self.layer6(out5)
       
        out7 = self.layer7(out6)
        out8 = self.layer8(out7)
       
        dout8 = self.dlayer8(out8)
       
        dout8_out7 = torch.cat([dout8, out7], 1)
       
        dout7 = self.dlayer7(dout8_out7)
        dout7_out6 = torch.cat([dout7, out6], 1)
        dout6 = self.dlayer6(dout7_out6)
        dout6_out5 = torch.cat([dout6, out5], 1)
        dout5 = self.dlayer5(dout6_out5)
        dout5_out4 = torch.cat([dout5, out4], 1)
        dout4 = self.dlayer4(dout5_out4)
        dout4_out3 = torch.cat([dout4, out3], 1)
        dout3 = self.dlayer3(dout4_out3)
        dout3_out2 = torch.cat([dout3, out2], 1)
        dout2 = self.dlayer2(dout3_out2)
        dout2_out1 = torch.cat([dout2, out1], 1)
        dout1 = self.dlayer1(dout2_out1)
        
        return dout1



class D(nn.Module):
    def __init__(self, nc, nf):
        super(D, self).__init__()

        main = nn.Sequential()
        # 256
        layer_idx = 1
        name = 'layer%d' % layer_idx
        main.add_module('%s_conv' % name, nn.Conv2d(nc, nf, 4, 2, 1, bias=False))

        # 128
        layer_idx += 1 
        name = 'layer%d' % layer_idx
        main.add_module(name, blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False))

        # 64
        layer_idx += 1 
        name = 'layer%d' % layer_idx
        nf = nf * 2
        main.add_module(name, blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False))

        # 32    
        layer_idx += 1 
        name = 'layer%d' % layer_idx
        nf = nf * 2
        main.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
        main.add_module('%s_conv' % name, nn.Conv2d(nf, nf*2, 4, 1, 1, bias=False))
        main.add_module('%s_bn' % name, nn.BatchNorm2d(nf*2))

        # 31
        layer_idx += 1 
        name = 'layer%d' % layer_idx
        nf = nf * 2
        main.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
        main.add_module('%s_conv' % name, nn.Conv2d(nf, 1, 4, 1, 1, bias=False))
        main.add_module('%s_sigmoid' % name , nn.Sigmoid())
        # 30 (sizePatchGAN=30)

        self.main = main

    def forward(self, x):
        output = self.main(x)
        return output
