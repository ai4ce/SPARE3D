import torch
import torch.nn as nn
# from chamferdist import ChamferDist

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


# class ChamfersDistance3(nn.Module):
#      '''
#      Extensively search to compute the Chamfersdistance. No reference to external implementation Incomplete
#      '''
#      def forward(self, input1, input2):
#          # input1, input2: BxNxK, BxMxK, K = 3
#          dist1, dist2, idx1, idx2 = chamferDist(pc1, pc2)
#          # B, N, K = input1.shape
#          # _, M, _ = input2.shape

#          # # Repeat (x,y,z) M times in a row
#          # input11 = input1.unsqueeze(2)           # BxNx1xK
#          # input11 = input11.expand(B, N, M, K)    # BxNxMxK
#          # # Repeat (x,y,z) N times in a column
#          # input22 = input2.unsqueeze(1)           # Bx1xMxK
#          # input22 = input22.expand(B, N, M, K)    # BxNxMxK
#          # # compute the distance matrix
#          # D = input11 - input22                   # BxNxMxK
#          # D = torch.norm( D, p=2, dim=3 )         # BxNxM

#          # dist0, _ = torch.min( D, dim=1 )        # BxM
#          # dist1, _ = torch.min( D, dim=2 )        # BxN

#          loss = torch.mean(dist1, 1) + torch.mean(dist2, 1)  # B
#          loss = torch.mean(loss)                             # 1
#          return loss


class ChamfersDistance3(nn.Module):
     '''
     Extensively search to compute the Chamfersdistance. No reference to external implementation Incomplete
     '''
     def forward(self, input1, input2):
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
         loss = torch.mean(loss) 
                                    # 1
         return loss     

class FoldingNetSingle(nn.Module):
    def __init__(self, dims):
        super(FoldingNetSingle, self).__init__()
        self.mlp = PointwiseMLP(dims, doLastRelu=False)

    def forward(self, X):
        return self.mlp.forward(X)
    

     
