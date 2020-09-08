from __future__ import print_function, division

from torchvision import models
import torch.nn as nn
import torch
import numpy as np
import bagnets.pytorchnet



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

