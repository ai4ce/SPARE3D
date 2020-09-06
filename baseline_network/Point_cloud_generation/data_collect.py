import os
from torch.utils.data import Dataset
import numpy as np
import torch
import cv2
import glob


class Task_6_dataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.dic=sorted(os.listdir(root_dir))
        self.root_dir=root_dir

    def __getitem__(self, idx):   

        if torch.is_tensor(idx):
            idx = idx.tolist()        
        input_dic = os.path.join(self.root_dir,
                                self.dic[idx], "Dic3V")
        output_dic = os.path.join(self.root_dir,
                                self.dic[idx], "DicAns")
        Front_img=self.convert_input(input_dic, "/*front.png")
        Right_img=self.convert_input(input_dic, "/*right.png")
        Top_img=self.convert_input(input_dic, "/*top.png")
        pcd_GT=self.load_pcd(output_dic, "/*.npy")
        img = torch.cat((Front_img, Right_img, Top_img), 2)
        
        # print("name of file:", input_dic)
        return {'pcd': pcd_GT, 'img': img, 'index':self.dic[idx]}

    def __len__(self):
        return len(self.dic)

    def convert_input(self, Dic, name):
        file_name=glob.glob(Dic+name)
        img=cv2.imread(file_name[0])/255
        img = torch.from_numpy(img)
        return img

    def load_pcd(self, Dic, name):
        file_name = glob.glob(Dic+name)
        pcd_GT = np.load(file_name[0])
        X_range = pcd_GT[:,0].max() - pcd_GT[:,0].min()
        Y_range = pcd_GT[:,1].max() - pcd_GT[:,1].min()
        Z_range = pcd_GT[:,2].max() - pcd_GT[:,2].min()
        max_range = max(X_range, Y_range, Z_range)
        # mean_points = pcd_GT.mean(axis=0).mean(axis=0)
        # range_points = pcd_GT.max(axis=0).max(axis=0) - pcd_GT.min(axis=0).min(axis=0)
        # pcd_GT -= mean_points
        # pcd_GT /= range_points
        
        N = pcd_GT.shape[0]
        if N >= 10000:
            pcd_GT = pcd_GT[np.random.choice(pcd_GT.shape[0], 10000, replace=False)]
        else: 
            pcd_GT = pcd_GT[np.random.choice(pcd_GT.shape[0], 10000, replace=True)]

        pcd_GT = pcd_GT/max_range #normalize by the max range, obtained previously
            
        pcd_GT = torch.from_numpy(pcd_GT)
        return pcd_GT

        