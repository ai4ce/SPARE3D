#!/usr/bin/env python

import sys
import torch
import torch.nn as nn
import torch.nn.functional as Functional

from .pointnet import PointNetGlobalMax, get_MLP_layers, PointNetVanilla, PointwiseMLP

# In order to use NNDModule from a "C" implementation
# ./nndistance put side by side of ./python
# sys.path.append('/home/ryw/project/NeuralSoftBot/neuralnet/net/nndistance')
# from modules.nnd import NNDModule


# class ChamfersDistance(nn.Module):
#     '''
#     Use NNDModule as a member.
#     '''
#     def __init__(self):
#         super(ChamfersDistance, self).__init__()
#         self.nnd = NNDModule()

#     def forward(self, input1, input2):          # BxNxK, BxMxK
#         dist0, dist1 = self.nnd.forward(input1, input2)  # BxN, BxM
#         loss = torch.mean(torch.sqrt(dist0), 1) + torch.mean(torch.sqrt(dist1), 1)  # B
#         loss = torch.mean(loss)                             # 1
#         # loss = torch.rand(1,requires_grad=True)
#         return loss


# class ChamfersDistance2(nn.Module):
#     '''
#     Derive a new class from NNDModule
#     '''
#     def forward(self, input1, input2):                                            # BxNxK, BxMxK
#         dist0, dist1 = super( ChamfersDistance3, self ).forward( input1, input2 )  # BxN, BxM
#         loss = torch.mean(torch.sqrt(dist0), 1) + torch.mean(torch.sqrt(dist1), 1)  # B
#         loss = torch.mean(loss)                                                    # 1
#         return loss


# class ChamfersDistance3(nn.Module):
#     '''
#     Extensively search to compute the Chamfersdistance. No reference to external implementation Incomplete
#     '''
#     def forward(self, input1, input2):
#         # input1, input2: BxNxK, BxMxK, K = 3
#         B, N, K = input1.shape
#         _, M, _ = input2.shape

#         # Repeat (x,y,z) M times in a row
#         input11 = input1.unsqueeze(2)           # BxNx1xK
#         input11 = input11.expand(B, N, M, K)    # BxNxMxK
#         # Repeat (x,y,z) N times in a column
#         input22 = input2.unsqueeze(1)           # Bx1xMxK
#         input22 = input22.expand(B, N, M, K)    # BxNxMxK
#         # compute the distance matrix
#         D = input11 - input22                   # BxNxMxK
#         D = torch.norm( D, p=2, dim=3 )         # BxNxM

#         dist0, _ = torch.min( D, dim=1 )        # BxM
#         dist1, _ = torch.min( D, dim=2 )        # BxN

#         loss = torch.mean(dist0, 1) + torch.mean(dist1, 1)  # B
#         loss = torch.mean(loss)                             # 1
#         return loss


class FoldingNetSingle(nn.Module):
    def __init__(self, dims):
        super(FoldingNetSingle, self).__init__()
        self.mlp = PointwiseMLP(dims, doLastRelu=False)

    def forward(self, X):
        return self.mlp.forward(X)


class FoldingNetVanilla(nn.Module):             # PointNetVanilla or nn.Sequential
    def __init__(self, MLP_dims, FC_dims, grid_dims, Folding1_dims,
                 Folding2_dims, MLP_doLastRelu=False):
        assert(MLP_dims[-1]==FC_dims[0])
        super(FoldingNetVanilla, self).__init__()
        # Encoder
        #   PointNet
        self.PointNet = PointNetVanilla(MLP_dims, FC_dims, MLP_doLastRelu)

        # Decoder
        #   Folding
        #     2D grid: (grid_dims(0) * grid_dims(1)) x 2
        # TODO: normalize the grid to align with the input data
        self.N = grid_dims[0] * grid_dims[1]
        u = (torch.arange(0, grid_dims[0], dtype=torch.float32) / grid_dims[0] - 0.5).repeat(grid_dims[1])
        v = (torch.arange(0, grid_dims[1], dtype=torch.float32) / grid_dims[1] - 0.5).expand(grid_dims[0], -1).t().reshape(-1)
        self.grid = torch.stack((u, v), 1)      # Nx2

        #     1st folding
        self.Fold1 = FoldingNetSingle(Folding1_dims)
        #     2nd folding
        self.Fold2 = FoldingNetSingle(Folding2_dims)

    def forward(self, X):
        # encoding
        f = self.PointNet.forward(X)            # BxK
        f = f.unsqueeze(1)                      # Bx1xK
        codeword = f.expand(-1, self.N, -1)     # BxNxK

        # cat 2d grid and feature
        B = codeword.shape[0]                   # extract batch size
        if not X.is_cuda:
            tmpGrid = self.grid                 # Nx2
        else:
            tmpGrid = self.grid.cuda()          # Nx2
        tmpGrid = tmpGrid.unsqueeze(0)
        tmpGrid = tmpGrid.expand(B, -1, -1)     # BxNx2

        # 1st folding
        f = torch.cat((tmpGrid, codeword), 2 )  # BxNx(K+2)
        f = self.Fold1.forward(f)               # BxNx3

        # 2nd folding
        f = torch.cat((f, codeword), 2 )        # BxNx(K+3)
        f = self.Fold2.forward(f)               # BxNx3
        return f
