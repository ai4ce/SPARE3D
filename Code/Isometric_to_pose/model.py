from __future__ import print_function, division

from torchvision import models
import torch.nn as nn
import torch
import os
import numpy as np
import bagnets.pytorchnet



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
