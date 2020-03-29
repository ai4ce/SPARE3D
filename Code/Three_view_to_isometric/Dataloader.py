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
class ThreeV2I_BC_data(Dataset):
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
        Ans_1=self.convert_input(input_dic,"/0.png")
        Ans_2=self.convert_input(input_dic,"/1.png")
        Ans_3=self.convert_input(input_dic,"/2.png")
        Ans_4=self.convert_input(input_dic,"/3.png")
        Label=self.convert_answer(answer)
        View=np.concatenate((Front_img,Right_img,Top_img),axis=2)
        input_1=np.moveaxis(np.concatenate((Ans_1,View),axis=2),-1,0)
        input_2=np.moveaxis(np.concatenate((Ans_2,View),axis=2),-1,0)
        input_3=np.moveaxis(np.concatenate((Ans_3,View),axis=2),-1,0)
        input_4=np.moveaxis(np.concatenate((Ans_4,View),axis=2),-1,0)
        
        return input_1,input_2,input_3,input_4,Label
    def __len__(self):
        return len(self.dic)
    def convert_input(self,Dic,name):
        file_name=glob.glob(Dic+name)
        img=cv2.imread(file_name[0])
        if self.transform:
            img=self.transform(img)
        return img/255
    def convert_answer(self,index):
    
        if index==0:
            output=torch.tensor([1,0,0,0])
        if index==1:
            output=torch.tensor([0,1,0,0])
        if index==2:
            output=torch.tensor([0,0,1,0])
        if index==3:
            output=torch.tensor([0,0,0,1])
        return output



class ThreeV2I_ML_data(Dataset):
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
        Ans_1=self.convert_input(input_dic,"/0.png")
        Ans_2=self.convert_input(input_dic,"/1.png")
        Ans_3=self.convert_input(input_dic,"/2.png")
        Ans_4=self.convert_input(input_dic,"/3.png")
        Label=self.convert_answer(answer)
        View=np.moveaxis((np.concatenate((Front_img,Right_img,Top_img),axis=2)),-1,0)
        input_1=np.moveaxis(Ans_1,-1,0)
        input_2=np.moveaxis(Ans_2,-1,0)
        input_3=np.moveaxis(Ans_3,-1,0)
        input_4=np.moveaxis(Ans_4,-1,0)
       # 
        return View,input_1,input_2,input_3,input_4,Label
    def __len__(self):
        return len(self.dic)
    def convert_input(self,Dic,name):
        file_name=glob.glob(Dic+name)
        img=cv2.imread(file_name[0])
        if self.transform:
            img=self.transform(img)
        return img/255
    def convert_answer(self,index):
       
        if index==0:
            output=torch.tensor([0])
        if index==1:
            output=torch.tensor([1])
        if index==2:
            output=torch.tensor([2])
        if index==3:
            output=torch.tensor([3])
        return output
