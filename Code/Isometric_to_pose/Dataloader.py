from __future__ import print_function, division

from torchvision import datasets, models, transforms
import torch.nn as nn
import torch
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import cv2

class I2P_data(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dic=sorted(os.listdir(root_dir))
        self.dic.remove('answer.json')
        self.transform=transform
        self.root_dir=root_dir
        with open(os.path.join(self.root_dir, 'answer.json'), 'r') as f:
            self.answer = json.load(f)
    def __getitem__(self, idx):   
        if torch.is_tensor(idx):
            idx = idx.tolist()        
        input_dic = os.path.join(self.root_dir,
                                self.dic[idx])
       
        
        answer = self.answer[self.dic[idx]]
    
        Front_img=self.convert_input(input_dic,"/*f.png")
        Right_img=self.convert_input(input_dic,"/*r.png")
        Top_img=self.convert_input(input_dic,"/*t.png")
       
        Ans=self.convert_input(input_dic,"/answer.png")
        Label=self.convert_answer(answer)
        View=np.concatenate((Front_img,Right_img,Top_img),axis=2)
        group=np.moveaxis(np.concatenate((Ans,View),axis=2),-1,0)
        return group,Label
    def __len__(self):
        return len(self.dic)
    def convert_input(self,Dic,name):
        file_name=glob.glob(Dic+name)
        img=cv2.imread(file_name[0])
        if self.transform:
            img=self.transform(img)
        return img/255
    def convert_answer(self,index):
        
        if index=="0":
            output=torch.tensor([0])
        if index=="1":
            output=torch.tensor([1])
        if index=="2":
            output=torch.tensor([2])
        if index=="3":
            output=torch.tensor([3])
        return output
