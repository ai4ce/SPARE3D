import random
import os
import shutil
import argparse
from multiprocessing import pool, cpu_count
import glob
from model2svg import * 
from boolean import * 
import numpy as np
import json
from gevent import Timeout 
from gevent import monkey
parser = argparse.ArgumentParser()

parser.add_argument('-pathread',type=str,help='file or folder to be processed.')
parser.add_argument('-pathwrite',type=str,help='file or folder to write.')
parser.add_argument('-n','--n_cores',type=int,default=cpu_count(),help='number of processors.')
parser.add_argument('-W','--width',type=str,default="75mm",help='svg width.')
parser.add_argument('-H','--height',type=str,default="75mm",help='svg height.')
parser.add_argument('-t','--tol',type=float,default=0.04,help='svg discretization tolerance.')
parser.add_argument('-ml','--margin_left',type=int,default=1,help='svg left margin.')
parser.add_argument('-mt','--margin_top',type=int,default=1,help='svg top margin.')
parser.add_argument('-lw','--line_width',type=str,default='0.7',help='svg line width.')
parser.add_argument('-lwh','--line_width_hidden',type=str,default='0.35',help='svg hidden line width.')


args = parser.parse_args()

pathread = args.pathread
pathwrite = args.pathwrite

######## this is CSG 

index=sorted(os.listdir( pathread ))

fnames=[]
dirs=[]
for v in index:
	cad_name=os.path.join(pathread,v)
	v=v.replace(".step","")
	dirs.append(v)
	fnames.append(cad_name)


##### this is ABC
# dirs = sorted(os.listdir( pathread ))
# fnames=[]
# for v in dirs:
#     filename=os.path.join(pathread, v)
#     cad_name=glob.glob(filename+"/*.step")
#     fnames.append(cad_name[0])



answername = np.arange(len(dirs))

random.shuffle(answername)
label=answername%4
i=0
answer = {}
for id in dirs:
    answer[id] = str(label[i])
    i=i+1



def Generate_task(fname,answer_dic=answer,Path_output=pathwrite,args=args):
	converter = Model2SVG(width=args.width, height=args.height, tol=args.tol,
                          margin_left=args.margin_left, margin_top=args.margin_top,
                          line_width=args.line_width, line_width_hidden=args.line_width_hidden)

	path_list = fname.split(os.sep)

	# model_number=path_list[2]

	model_number=path_list[2].replace(".step","")

	MotherDic = Path_output + "/"+model_number+"/"
	
	if not os.path.exists(MotherDic):
		os.makedirs(MotherDic)
	viewpoints=['f','r','t']
	answer_number=["0","1","2","3"]
	answer_number.remove(answer_dic[model_number])
	try:
		seconds = 60 
		timeout = Timeout(seconds)
		timeout.start()
		shp = read_step_file(fname)
		boundbox = get_boundingbox(shp,use_mesh=False)
		sc=49/(max(boundbox[6],boundbox[7],boundbox[8]))
		
		#### generate F R T views 
		simple_shp_1=New_shp(boundbox[0],boundbox[1],boundbox[2],boundbox[3],boundbox[4],boundbox[5],boundbox[6],boundbox[7],boundbox[8])
		shp_1=BRepAlgoAPI_Cut(shp,simple_shp_1).Shape()
		for vp in viewpoints:
			converter.export_shape_to_svg(shape=shp_1, filename=MotherDic+model_number+"_"+vp+".svg", proj_ax=converter.DIRS[vp],scale=sc)
		
        #### generate correct answer 
		
		converter.export_shape_to_svg(shape=shp_1, filename=MotherDic+answer_dic[model_number]+".svg", proj_ax=converter.DIRS["2"],scale=sc)
		
         

        ###  generate wrong answers  

		for item in answer_number:
			simple_shp_2=New_shp(boundbox[0],boundbox[1],boundbox[2],boundbox[3],boundbox[4],boundbox[5],boundbox[6],boundbox[7],boundbox[8])
			shp_2=BRepAlgoAPI_Cut(shp,simple_shp_2).Shape()
			converter.export_shape_to_svg(shape=shp_2, filename=MotherDic+item+".svg", proj_ax=converter.DIRS["2"],scale=sc)
		return 1
	except Exception as re:
		shutil.rmtree(MotherDic)
		print(MotherDic + "has been removed")
		print(fname + ' failed, due to: {}'.format(re))
		return 0

p = pool.Pool(processes=args.n_cores)
f = partial(Generate_task)

t0 = time.time()

mask = p.map(f, fnames)
Mask=np.asarray(mask)
label_valid=np.delete(label, np.where(Mask == 0))

label_valid=[int(i) for i in label_valid]


dirs_valid = np.delete(np.array(dirs), np.where(Mask == 0)) 

Answer = dict(zip(dirs_valid, label_valid))

fname_answer = os.path.join(pathwrite, 'answer.json')
with open(fname_answer, 'w') as ff:
	json.dump(Answer, ff, indent=4)
duration = time.time() - t0
p.close()
n_success = sum(mask)
print('{} done,  {} failed, elapsed time = {}!'.format(n_success, len(mask)-n_success,
                                              time.strftime("%H:%M:%S", time.gmtime(duration))))

