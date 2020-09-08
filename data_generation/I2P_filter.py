import glob
import cv2
import os
import shutil
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-pathread',type=str,default = '/home/yfx/Spare3D/Data_not_touch/test_out_I2P', help='file or folder to be processed.')
parser.add_argument('-pathwrite',type=str,default = '/home/yfx/Spare3D/Data_not_touch/test_out_I2P_filt', help='file or folder to write.')
parser.add_argument('-rule_out',type=int,default = 500, help='the number of file you want to rule out')
args = parser.parse_args()

root_dir=args.pathread
destination=args.pathwrite

if not os.path.exists(destination):
    os.mkdir(destination)

dics = sorted(os.listdir(root_dir))
dics.remove('answer.json')
potent_data = len(dics) - args.rule_out

counter=0

def get_sym(img1,img2,img3):
    j=np.inf
    if j>np.size(np.nonzero(img1-img2)):
        j=np.size(np.nonzero(img1-img2))
    if j>np.size(np.nonzero(img1-img3)):
        j=np.size(np.nonzero(img1-img3))
    a0 = img2[:, :, 0].T[::-1,:].reshape(np.shape(img2[:, :, 0])[0], np.shape(img2[:, :, 0])[1], 1)
    a1 = img2[:, :, 1].T[::-1,:].reshape(np.shape(img2[:, :, 1])[0], np.shape(img2[:, :, 1])[1], 1)
    a2 = img2[:, :, 2].T[::-1,:].reshape(np.shape(img2[:, :, 2])[0], np.shape(img2[:, :, 2])[1], 1)
    img2r = np.concatenate((a0, a1, a2), axis=2)
    if j>np.size(np.nonzero(img2r-img3)):
        j=np.size(np.nonzero(img2r-img3))
    img1v=img1[::-1,::-1]
    img1v=img1-img1v
    if j>np.size(np.nonzero(img1v)):
        j=np.size(np.nonzero(img1v))
    return j

sym=[]

for i,dic in enumerate(dics):
    dir=os.path.join(root_dir,dic)
    path = glob.glob(dir + "/*f.png")[0]
    imgf = cv2.imread(path)
    path = glob.glob(dir + "/*_r.png")[0]
    imgr = cv2.imread(path)
    path = glob.glob(dir + "/*t.png")[0]
    imgt = cv2.imread(path)
    j = get_sym(imgf, imgr, imgt)*(1+i/len(dics)*1.5)
    sym.append(j)
    
index=np.argsort(sym)

des_folder = os.path.join(destination + '/temperary_data')

if not os.path.exists(des_folder):
    os.mkdir(des_folder)

des_dir = os.path.join(destination + '/temperary_data' + '/sorted_symmetrical')

if not os.path.exists(des_dir):
    os.mkdir(des_dir)

for i in range(len(dics)):
    des_path = os.path.join(des_dir,dics[i])
    if not os.path.exists(des_path):
        os.mkdir(des_path)
    cp_dir = os.path.join(root_dir,dics[index[i]])
    for root, dirs, files in os.walk(cp_dir):
        for file in files:
            src_file = os.path.join(root, file)
            shutil.copy(src_file, des_path)

copy_scr=os.path.join(destination + '/temperary_data' + '/sorted_symmetrical')
des_folder=os.path.join(destination + '/temperary_data' + '/select_good_data')
if not os.path.exists(des_folder):
    os.mkdir(des_folder)
indexs=os.listdir(copy_scr)[-potent_data:]

for index in indexs:
    src_dir = os.path.join(copy_scr, index)
    des_dir = os.path.join(des_folder, index)
    if not os.path.exists(des_dir):
        os.mkdir(des_dir)
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            src_file = os.path.join(root, file)
            shutil.copy(src_file, des_dir)

des_folder = os.path.join(destination + '/temperary_data' + '/select_good_data')
original_dataset_dir = des_folder

des_folder = destination
# des_folder=destination+'/selected_good_data'

# if not os.path.exists(des_folder):
#     os.mkdir(des_folder)

indexs=os.listdir(original_dataset_dir)

for index in indexs:
    src_dir=os.path.join(original_dataset_dir,index)
    for root, dirs, files in os.walk(src_dir):
        fold=files[1][0:8]
        des_dir=os.path.join(des_folder,fold)
        if not os.path.exists(des_dir):
            os.mkdir(des_dir)
        for file in files:
            src_file = os.path.join(root, file)
            shutil.copy(src_file, des_dir)

src_dir = os.path.join(root_dir +'/answer.json')
shutil.copy(src_dir, des_folder)

temperary_dir = os.path.join(destination + '/temperary_data')
shutil.rmtree(temperary_dir)

answer_p_dir = os.path.join(destination + '/answer.p')
shutil.rmtree(answer_p_dir)