from __future__ import print_function, division

import torch
import glob
from torch.utils.data import Dataset
import os
import numpy as np
import cv2

class IVG_data(Dataset):
    def __init__(self, root_dir,transform=None):
        self.dic=sorted(os.listdir(root_dir))
        self.transform=transform
        self.root_dir=root_dir
    def __getitem__(self, idx):   
        if torch.is_tensor(idx):
            idx = idx.tolist()        
        input_dic = os.path.join(self.root_dir,
                                self.dic[idx],"Dic3V")
       
        
        output_dic = os.path.join(self.root_dir,
                                self.dic[idx],"DicAns")
  
       
        Front_img=self.convert_input(input_dic,"/*front.png")
        Right_img=self.convert_input(input_dic,"/*right.png")
        Top_img=self.convert_input(input_dic,"/*top.png")
        Target=self.convert_output(output_dic,"/isometric.png")
        View=np.concatenate((Front_img,Right_img,Top_img),axis=2)
        Input=np.moveaxis(View,-1,0)
        target=np.moveaxis(Target,-1,0)
        
        return Input,target
    def __len__(self):
        return len(self.dic)
    def convert_input(self,Dic,name):
        file_name=glob.glob(Dic+name)
        img=cv2.imread(file_name[0])
        if self.transform:
            img=cv2.resize(img, (256,256))
        return (img/255)*2-1
    def convert_output(self,Dic,name):
        file_name=Dic+name
        
        img=cv2.imread(file_name)
        if self.transform:
            img=cv2.resize(img, (256,256))
        return (img/255)*2-1
