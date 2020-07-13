import glob
import cv2
import os
import shutil
import numpy as np

root_dir="C:/Users/ay162/Desktop/research/New_data/P2I_CSG/csg_png_p2i"
destination="C:/Users/ay162/Desktop/research/New_data/P2I_CSG"
potent_data=8000

dics = sorted(os.listdir(root_dir))
dics.remove('answer.json')
counter=0

def get_sym(img,t):
    img1=img[::-1,:]
    img2=img[:,::-1]
    img3=img1[:,::-1]
    img4=img.T
    img5=img4[::-1,:]
    img6=img4[:,::-1]
    img7=img5[:,::-1]
    if t>np.size(np.nonzero(img-img1)):
        t=np.size(np.nonzero(img-img1))
    if t>np.size(np.nonzero(img-img2)):
        t=np.size(np.nonzero(img-img2))
    if t>np.size(np.nonzero(img-img3)):
        t=np.size(np.nonzero(img-img3))
    if t>np.size(np.nonzero(img-img4)):
        t=np.size(np.nonzero(img-img4))
    if t>np.size(np.nonzero(img-img5)):
        t=np.size(np.nonzero(img-img5))
    if t>np.size(np.nonzero(img-img6)):
        t=np.size(np.nonzero(img-img6))
    if t>np.size(np.nonzero(img-img7)):
        t=np.size(np.nonzero(img-img7))
    return t

sym=[]

for dic in dics:
    blank=False
    j=np.inf
    dir=os.path.join(root_dir,dic)
    path = glob.glob(dir + "/*f.png")[0]
    imgf = cv2.imread(path, 0)
    _, threshf = cv2.threshold(imgf, 254, 0, cv2.THRESH_TOZERO)
    if np.all(threshf==255):
        blank=True
    path = glob.glob(dir + "/*r.png")[0]
    imgr = cv2.imread(path, 0)
    _, threshr = cv2.threshold(imgr, 254, 0, cv2.THRESH_TOZERO)
    if np.all(threshr==255):
        blank=True
    path = glob.glob(dir + "/*t.png")[0]
    imgt = cv2.imread(path, 0)
    _, thresht = cv2.threshold(imgt, 254, 0, cv2.THRESH_TOZERO)
    if np.all(thresht==255):
        blank=True
    if blank:
        j=0
    else:
        j = get_sym(threshf, j)
        j = get_sym(threshr, j)
        j = get_sym(thresht, j)
    if j==0:
        counter+=1
    sym.append(j)
print(counter)
index=np.argsort(sym)

des_folder=destination+"/temperary_data"

if not os.path.exists(des_folder):
    os.mkdir(des_folder)

des_dir=destination+"/temperary_data/phase1_data"

if not os.path.exists(des_dir):
    os.mkdir(des_dir)

for i in range(len(dics)):
    des_path=os.path.join(des_dir,dics[i])
    if not os.path.exists(des_path):
        os.mkdir(des_path)
    cp_dir=os.path.join(root_dir,dics[index[i]])
    for root, dirs, files in os.walk(cp_dir):
        for file in files:
            src_file = os.path.join(root, file)
            shutil.copy(src_file, des_path)

copy_scr=destination+"/temperary_data/phase1_data"
des_folder=destination+'/temperary_data/phase2_data'
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

original_dataset_dir = des_folder

des_folder=destination+'/temperary_data/phase3_data'

if not os.path.exists(des_folder):
    os.mkdir(des_folder)

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

original_dataset_dir = des_folder

total_num = int(len(os.listdir(original_dataset_dir)))

random_idx = np.array(range(total_num))

np.random.shuffle(random_idx)

total_num=int(total_num/10)*10

base_dir = destination+'/temperary_data/phase4_data'
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

sub_dirs = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
test_indexs=[]
for i in range(10):
    test_indexs.append(random_idx[int(total_num * i*0.1):int(total_num * (i+1)*0.1)])
for idx, sub_dir in enumerate(sub_dirs):
    dir = os.path.join(base_dir, sub_dir)
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)
    fnames = [os.listdir(original_dataset_dir)[i] for i in test_indexs[idx]]
    for fname in fnames:
        src_dir=os.path.join(original_dataset_dir,fname)
        des_dir=os.path.join(dir,fname)
        if not os.path.exists(des_dir):
            os.mkdir(des_dir)
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                src_file = os.path.join(root, file)
                shutil.copy(src_file, des_dir)

base_dir=destination+'/Test_data'
if not os.path.exists(base_dir):
    os.mkdir(base_dir)
src_dir=destination+'/temperary_data/phase4_data'
for i in range(10):
    dir = os.path.join(base_dir, 'Test_Group{}'.format(i+1))
    if not os.path.exists(dir):
        os.mkdir(dir)
    train_index=['{}'.format(j+1) for j in range(10) if j!=i]
    test_index=['{}'.format(i+1)]
    stage=['test','train']
    phases=[test_index,train_index]
    for phase_index,phasess in enumerate(stage):
        des_dir=os.path.join(dir,phasess)
        if not os.path.exists(des_dir):
            os.mkdir(des_dir)
        answer_dir=root_dir+'/answer.json'
        shutil.copy(answer_dir, des_dir)
        for phase in phases[phase_index]:
            src_dir1 = os.path.join(src_dir, phase)
            folders=os.listdir(src_dir1)
            for folder in folders:
                src_folder = os.path.join(src_dir1, folder)
                des_folder=os.path.join(des_dir,folder)
                if not os.path.exists(des_folder):
                    os.mkdir(des_folder)
                for root, dirs, files in os.walk(src_folder):
                    for file in files:
                        src_file = os.path.join(root, file)
                        shutil.copy(src_file, des_folder)

