import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import numpy as np
import othernet
from othernet import FoldingNetSingle

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


def resfoldnetoneforall(grid, dim=512):
    return ResFoldNetOneForAll(dim, (3, 32, dim), (dim, dim, dim, 3), grid)


def resnetfoldbaymax(grid, dim=512, res=False, input_channels=9):
    return ResFoldNetOneForAll(dim, (3, 32, dim), (dim, dim, dim, 3), grid, input_channels=input_channels, res=res)

# def resnetfoldbaymaxres(grid, dim=512):
#     return ResFoldNetOneForAll(dim, (3, 32, dim), (dim, dim, dim, 3), grid, input_channels=15, res=True)

def resnetfoldbaymaxold(grid, dim=512):
    return ResFoldNetOneForAll(dim, (3, 32, dim), (dim + len(grid.shape), dim, dim, 3), grid, input_channels=15, old=True)

if __name__ == '__main__':
    grid = torch.rand(100, 3)
    model = resfoldnetoneforall(grid, 256)
    model = model.cuda()
    img = torch.rand(10, 3, 112, 112)
    img = img.cuda()
    p  = model(img)
    print(p.shape)
