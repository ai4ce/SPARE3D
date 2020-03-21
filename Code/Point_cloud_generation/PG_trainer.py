import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data_collect import Task_6_dataset
import argparse
import glog as logger
import othernet
# from othernet import ChamfersDistance3
import net
from othernet import ChamfersDistance3
from chamfer_distance import ChamferDistance
#from pytorch3d.loss import chamfer_distance
# from chamfer_distance import ChamferDistance

# parser = argparse.ArgumentParser(description='train res-folding net')
# parser.add_argument('datapath', metavar='DATA_PATH', type=str, help='path to data')
# parser.add_argument('outpath', metavar='OUT_PATH', type=str, help='path to save model, loss and optimizer state')
# parser.add_argument('-b', '--batch-size', type=int, default=16, help='batch size')
# parser.add_argument('-l', '--learning-rate', type=float, default=1e-4, help='learning rate')
# parser.add_argument('-e', '--epochs', type=int, default=500, help='number of epochs')
# parser.add_argument('-d', '--dimension', type=int, default=512, help='dimension of latent space, if 0, using original mode')
# parser.add_argument('-s', '--size', type=int, default=0, help='image size')
# parser.add_argument('--old', action='store_true', default=False)
# parser.add_argument('--res', action='store_true', default=False)
# parser.add_argument('--pro', action='store_true', default=False)
# args = parser.parse_args()
def cd(x, y):
       # x = x.permute(0, 2, 1)
        #y = y.permute(0, 2, 1)
        d1, d2 = ChamferDistance()(x, y)
        loss=torch.sum(d1, dim=1) + torch.sum(d2, dim=1)
        print(loss)
        return torch.mean(loss,dim=0)

# if not os.path.exists(args.datapath):
#     logger.error('Data %s is not found.' % args.datapath)
#     exit(1)

outputpath="./log_128_0.005"
if not os.path.exists(outputpath):
    os.mkdir(outputpath)
    os.mkdir(outputpath + '/model')
    os.mkdir(outputpath +'/loss')
#chamfer_loss = chamfer_distance
chamfer_loss = ChamfersDistance3().to("cuda:2")

def train(dataset, model, batch_size, lr, epochs,device,outputpath):
    """train res-fold net
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    model.to(device)
    model.train()
    loss_log = open(outputpath+'/loss/loss.txt', 'w')
    iters = 0
    for ep in range(epochs):
        for batch_idx, batch in enumerate(dataloader):
            # print("name in batch:", batch_idx)
            opt.zero_grad()
            img = batch['img'].type(torch.FloatTensor)
            pcd = batch['pcd'].type(torch.FloatTensor).to(device)
            img = img.permute(0, 3, 1, 2).to(device) #batch size, image channel, image height, image width
            pcd_pred = model(img)
          
            loss = chamfer_loss(pcd_pred, pcd)
            
            loss.backward()
            opt.step()
            iters += 1
            if batch_idx % 10 == 9:
                logger.info('[%d, %5d] loss: %.6f' %
                    (ep + 1, batch_idx + 1, loss.item()))
            print(iters, loss.item(), file=loss_log)
        torch.save(model.state_dict(), os.path.join(outputpath, 'model','ep_%d.pth' % (ep + 1)))
    loss_log.close()
    return 0    


if __name__ == '__main__':
    device="cuda:2"
    path = '/home/wenyuhan/final_paper/task6_dataset_train'
    dataset=Task_6_dataset(path,transform=None)
    u = (torch.arange(0, 100, dtype=torch.float32) / 100 - 0.5).repeat(100)
    v = (torch.arange(0, 100, dtype=torch.float32) / 100 - 0.5).expand(100, -1).t().reshape(-1)
    w = torch.zeros(10000, dtype=torch.float32)
    grid = torch.stack((v, v, w), 1)   
    
    grid = torch.FloatTensor(grid).to(device)
    resfoldnet = net.resnetfoldbaymax(grid, dim=128, res=False)

    ret = train(dataset, resfoldnet, 2, 5e-3, 100,device,outputpath)
