import glob
import cv2
import os
import shutil
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-pathread',type=str,default = '/home/yfx/Spare3D/Data_not_touch/test_code1', help='file or folder to be processed.')
parser.add_argument('-pathwrite',type=str,default = '/home/yfx/Spare3D/Data_not_touch/test_out2', help='file or folder to write.')
args = parser.parse_args()

root_dir=args.pathread
des_dir=args.pathwrite


if not os.path.exists(des_dir):
    os.mkdir(des_dir)

answer_file = os.path.join(root_dir, 'answer.json')
shutil.copy(answer_file, des_dir)

dics=sorted(os.listdir(root_dir))
dics.remove('answer.json')
counter=0
for dic in dics:
    same=False
    value=0
    d=[]
    dir=os.path.join(root_dir,dic)
    path = glob.glob(dir + "/1.png")[0]
    img = cv2.imread(path)
    d.append(img)
    path = glob.glob(dir + "/*2.png")[0]
    img = cv2.imread(path)
    d.append(img)
    path = glob.glob(dir + "/3.png")[0]
    img = cv2.imread(path)
    d.append(img)
    path = glob.glob(dir + "/0.png")[0]
    img = cv2.imread(path)
    d.append(img)
    for i in range(3):
        for j in range(i+1,4):
            if (d[i]==d[j]).all():
                value+=1
                same=True
    if same:
        counter+=1
        print(dic,':',value)
    else:
        des_folder = os.path.join(des_dir, dic)
        for root, dirs, files in os.walk(dir):
            if not os.path.exists(des_folder):
                os.mkdir(des_folder)
            for file in files:
                src_file = os.path.join(root, file)
                shutil.copy(src_file, des_folder)
print(counter)

