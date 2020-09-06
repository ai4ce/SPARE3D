import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data_collect import Task_6_dataset
import argparse
import glog as logger
from resfoldnet import *
import numpy as np

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


# if not os.path.exists(args.datapath):
#     logger.error('Data %s is not found.' % args.datapath)
#     exit(1)


# if not os.path.exists(args.outpath):
#     os.mkdir(args.outpath)
#     os.mkdir(args.outpath + '/model')
#     os.mkdir(args.outpath + 'result' +'/loss')

# chamer_loss = ChamfersDistance3()


def eval(dataset, model):

    errs_1 = []
    errs_2 = []
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    # model.eval()
    # model.cuda()

    # range_points = torch.FloatTensor(dataset.range_points).cuda()
    # mean_points = torch.FloatTensor(dataset.mean_points).cuda()
    for batch in dataloader:
        img = batch['img'].type(torch.FloatTensor).permute(0, 3, 1, 2).cuda()
        # ind = batch['ind'][0]
        ind = batch['index']
        # points_gt = batch['pcd'][0].type(torch.FloatTensor).cuda()

        points_pred = model(img)

        points_pred = points_pred[0].cpu()

        points_pred = points_pred.detach().numpy()
        # points_pred *= range_points
        # points_pred += mean_points

        # import pdb;pdb.set_trace()
        np.save('/home/siyuan/project/SIQ/task6/test_result/' + ind[0] + '.npy', points_pred)
        # points_gt *= range_points
        # points_gt += mean_points

    #     print("pcd %d predicted." % ind)
    #     d_1, d_2 = eval_single_gpu(points_pred, points_gt)
    #     errs_1.append(d_1)
    #     errs_2.append(d_2)

    # np.savez("errors_gpu_%d_%d.npz" % (args.dimension, args.size) , errs_1=errs_1, errs_2=errs_2)


def main():
    # dataset = BaymaxDataset(args.datapath, args.size)
    u = (torch.arange(0, 100, dtype=torch.float32) / 100 - 0.5).repeat(100)
    v = (torch.arange(0, 100, dtype=torch.float32) / 100 - 0.5).expand(100, -1).t().reshape(-1)
    w = torch.zeros(10000, dtype=torch.float32)
    grid = torch.stack((u, v, w), 1)
    grid = torch.FloatTensor(grid).cuda()
    model = resfold(grid, dim=512)
    N = 0
    print(model.parameters())
    # for p in model.parameters():
    #     N += p.numel()
    # print(N)
    model.eval()
    model.load_state_dict(torch.load('/home/siyuan/project/SIQ/task6/result/model2/ep_100.pth'))
    model.cuda()
    path = '/home/siyuan/project/SIQ/task6/task6_dataset_test'
    dataset = Task_6_dataset(path,transform=None)
    eval(dataset, model)

if __name__ == '__main__':
    main()

