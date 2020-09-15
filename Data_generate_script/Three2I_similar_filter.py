import os
import cv2
import glob
import numpy as np
import shutil
import argparse

# this is to filter questions using similar objects

parser = argparse.ArgumentParser()
parser.add_argument('-pathread',type=str,default = '/home/yfx/Spare3D/Data_not_touch/test_out_t2i_filt', help='file or folder to be processed.')
parser.add_argument('-pathwrite',type=str,default = '/home/yfx/Spare3D/Data_not_touch/test_out_t2i_simi', help='file or folder to write.')
args = parser.parse_args()

root = args.pathread
destination = args.pathwrite

if not os.path.exists(destination):
    os.mkdir(destination)

answer_file = os.path.join(root, 'answer.json')
shutil.copy(answer_file, destination)

dics = sorted(os.listdir(root))
dics.remove('answer.json')
dics = np.array(dics)
length = len(dics)
sym = np.zeros((length, length), dtype=int)
list = []
num = []
counter = 0

for i, dic1 in enumerate(dics[:-1]):
    print(i)
    dir1 = os.path.join(root, dic1)
    path1 = glob.glob(dir1 + '/*_f.png')[0]
    imgf1 = cv2.imread(path1)
    path1 = glob.glob(dir1 + '/*_r.png')[0]
    imgr1 = cv2.imread(path1)
    path1 = glob.glob(dir1 + '/*_t.png')[0]
    imgt1 = cv2.imread(path1)
    img1 = np.clip(imgf1 + imgr1 + imgt1, 0, 255)
    t = i + 1
    for k, dic2 in enumerate(dics[t:]):
        j = k + t
        dir2 = os.path.join(root, dic2)
        path2 = glob.glob(dir2 + '/*_f.png')[0]
        imgf2 = cv2.imread(path2)
        path2 = glob.glob(dir2 + '/*_r.png')[0]
        imgr2 = cv2.imread(path2)
        path2 = glob.glob(dir2 + '/*_t.png')[0]
        imgt2 = cv2.imread(path2)
        img2 = np.clip(imgf2 + imgr2 + imgt2, 0, 255)
        s = np.sum((img1 == img2) != 0)
        sym[i, j] = s
        sym[j, i] = s
        if s > 119990:
            counter += 1
    index = np.argsort(sym[i, :])[::-1]
    num.append(sym[i, :][index[0]])
    list.append(dics[index])
    print(dics[index][0])
index = np.argsort(sym[-1, :])[::-1]
num.append(sym[-1, :][index[0]])
list.append(dics[index])


index = np.argsort(num)[::-1]

file = open(destination + "/summary" + ".txt", "w")
file.write('Number of same models: %d\n' % counter)
file.write('\n')
for i, dic in enumerate(dics[index]):
    file.write('{}:'.format(dic))
    file.write('{}\n'.format(list[index[i]]))
    file.write('\n')
file.close()

for i in range(len(dics)):
    des_path = os.path.join(destination, dics[i])
    if not os.path.exists(des_path):
        os.mkdir(des_path)
    cp_dir = os.path.join(root, dics[index[i]])
    for roott, dirs, files in os.walk(cp_dir):
        for file in files:
            src_file = os.path.join(roott, file)
            shutil.copy(src_file, des_path)

