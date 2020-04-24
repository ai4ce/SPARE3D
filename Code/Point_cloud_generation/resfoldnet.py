#!/usr/bin/env python

import sys
import torch
import torch.nn as nn
import torch.nn.functional as Functional
try:
    import pytorch3d.loss.chamfer_distance as chd
except:
    print('pytorch 3d is not installed. Chamfer distance is calucated using brute force')
    
    def chd(input1, input2):
        # input1, input2: BxNxK, BxMxK, K = 3
        B, N, K = input1.shape
        _, M, _ = input2.shape

        # Repeat (x,y,z) M times in a row
        input11 = input1.unsqueeze(2)           # BxNx1xK
        input11 = input11.expand(B, N, M, K)    # BxNxMxK
        # Repeat (x,y,z) N times in a column
        input22 = input2.unsqueeze(1)           # Bx1xMxK
        input22 = input22.expand(B, N, M, K)    # BxNxMxK
        # compute the distance matrix
        D = input11 - input22                   # BxNxMxK
        D = torch.norm( D, p=2, dim=3 )         # BxNxM

        dist0, _ = torch.min( D, dim=1 )        # BxM
        dist1, _ = torch.min( D, dim=2 )        # BxN

        loss = torch.mean(dist0, 1) + torch.mean(dist1, 1)  # B
        loss = torch.mean(loss)                             # 1
        return loss


class ChamfersDistance(nn.Module):
   
    def forward(self, input1, input2):
        return chd(input1, input2)


def get_and_init_FC_layer(din, dout):
    li = nn.Linear(din, dout)
    #init weights/bias
    nn.init.xavier_uniform_(li.weight.data, gain=nn.init.calculate_gain('relu'))
    li.bias.data.fill_(0.)
    return li


def get_MLP_layers(dims, doLastRelu):
    layers = []
    for i in range(1, len(dims)):
        layers.append(get_and_init_FC_layer(dims[i-1], dims[i]))
        if i==len(dims)-1 and not doLastRelu:
            continue
        layers.append(nn.ReLU())
    return layers


class PointwiseMLP(nn.Sequential):
    '''Nxdin ->Nxd1->Nxd2->...-> Nxdout'''

    def __init__(self, dims, doLastRelu=False):
        layers = get_MLP_layers(dims, doLastRelu)
        super(PointwiseMLP, self).__init__(*layers)


class FoldingNetSingle(nn.Module):
    def __init__(self, dims):
        super(FoldingNetSingle, self).__init__()
        self.mlp = PointwiseMLP(dims, doLastRelu=False)

    def forward(self, X):
        return self.mlp.forward(X)


class Decoder(nn.Module):

    def __init__(self, map_dims, fold_dims, grid, res = False):
        super(Decoder, self).__init__()
        self.fold = FoldingNetSingle(fold_dims)
        self.map = FoldingNetSingle(map_dims)
        self.grid = grid                                # NxD
        self.N = grid.shape[0]
        self.res = res

    def forward(self, codeword):
        codeword = codeword.unsqueeze(1)                # Bx1xK
        codeword = codeword.expand(-1, self.N, -1)      # BxNxK
        
        # expand grid
        B = codeword.shape[0]                           # NxD  
        grid = self.grid.unsqueeze(0)                   # 1xNxD
        grid = grid.expand(B, -1, -1)                   # BxNxD
        grid_mapped = self.map(grid)                    # BxNxK
        f = grid_mapped + codeword                      # BxNxK
        f = self.fold.forward(f)                        # BxNxD -> BxNxdim_f???
        if self.res:
            f = f + grid
        return f


class ResFoldNetOneForAll(nn.Module):
    """using resnet18 as front-end, folding net as back-end
    """
    def __init__(self, code_dim, map_dims, fold_dims, grid, old=False, input_channels=9, res=False):
        super(ResFoldNetOneForAll, self).__init__()
        # assert(code_dim==map_dims[-1])
        # assert(code_dim==fold_dims[0])
        # assert(len(grid.shape)==2)
        # N, D = grid.shape
        # assert(D==map_dims[0])
        # assert(fold_dims[-1]==D)

        # Encoder resnet
        resnet = models.resnet50(pretrained=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.fc = othernet.get_and_init_FC_layer(2048, code_dim)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.grid = grid
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        # Decoder
        self.Decoder = Decoder(map_dims, fold_dims, self.grid, res)
    
    def encode(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)    # BxK
        return x

    def decode(self, codeword):
        return self.Decoder(codeword) # BxNxD

    def forward(self, x):
        codeword = self.encode(x)
        f = self.decode(codeword)
        return f


def resnetfold(grid, dim=512, res=False, input_channels=9):
    return ResFoldNetOneForAll(dim, (3, 32, dim), (dim, dim, dim, 3), grid, input_channels=input_channels, res=res)


if __name__ == '__main__':
    grid = torch.rand(100, 3)
    model = resfoldnetoneforall(grid, 256)
    model = model.cuda()
    img = torch.rand(10, 3, 112, 112)
    img = img.cuda()
    p  = model(img)
    print(p.shape)
