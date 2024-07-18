# from _future_ import print_function, division

import torch
import argparse
import os
import numpy as np
import time
from model import ThreeV2I_ML
from Dataloader import ThreeV2I_ML_data

parser = argparse.ArgumentParser()
parser.add_argument('--Testing_dataroot', required=True, help='path to Test data')
parser.add_argument('--statedic', required=True, help='path to model weights')
parser.add_argument('--batchsize', type=int, default=8, help='input batch size')
parser.add_argument('--device', default='cuda:0', help='device')
parser.add_argument('--model_type', default='vgg16',help='|vgg16| |resnet50| |Bagnet33|')

opt = parser.parse_args()
device = opt.device

model = ThreeV2I_ML(opt.model_type).to(device)

model.load_state_dict(torch.load(opt.statedic))

def test_model():
    test_loss = 0
    test_acc = 0
    data_transforms = False
    path = opt.Testing_dataroot
    test_data = ThreeV2I_ML_data(path)
    data_test = torch.utils.data.DataLoader(test_data, batch_size=opt.batchsize, shuffle=False)
    criterion = torch.nn.MarginRankingLoss(margin=2.0).to(device)

    with torch.no_grad():
        for i, (View, input_1, input_2, input_3, input_4, Label) in enumerate(data_test):
            View, input_1, input_2, input_3, input_4, Label = View.to(device), input_1.to(device), input_2.to(device), input_3.to(device), input_4.to(device), Label.to(device)

            y_1 = model(View.float(), input_1.float())
            y_2 = model(View.float(), input_2.float())
            y_3 = model(View.float(), input_3.float())
            y_4 = model(View.float(), input_4.float())
            Y = torch.cat((y_1, y_2, y_3, y_4), axis = 1)

            total_loss = torch.zeros(1).to(device)
            for j in range(input_1.shape[0]):
                Index = list(range(4))
                index = Label[j].item()
                Index.remove(index)

                loss_1 = criterion(Y[j][index].reshape(1), Y[j][Index[0]].reshape(1), torch.tensor([-1]).to(device))
                loss_2 = criterion(Y[j][index].reshape(1), Y[j][Index[1]].reshape(1), torch.tensor([-1]).to(device))
                loss_3 = criterion(Y[j][index].reshape(1), Y[j][Index[2]].reshape(1), torch.tensor([-1]).to(device))
                total_loss += (loss_1 + loss_2 + loss_3) / 3 / input_1.shape[0]

            test_loss += total_loss.item() * input_1.shape[0]
            test_acc += (Y.argmin(1) == Label.reshape(input_1.shape[0])).sum().item()

        epoch_test_loss = test_loss / len(test_data)
        epoch_test_acc = test_acc / len(test_data)

    return epoch_test_loss, epoch_test_acc

start_time = time.time()
test_loss, test_acc = test_model()
secs = int(time.time() - start_time)
mins = secs / 60
secs = secs % 60

print(f"Test Loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc * 100:.1f}%")
print("Time taken: %d minutes, %d seconds"%(mins, secs))
