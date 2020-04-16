import os
import sys
import argparse
import random
import shutil
import json
import numpy as np
import glob 
from functools import partial
from multiprocessing import pool, cpu_count
from model2svg import *
from gevent import Timeout

labels = {0:'frt1', 1:'frt2', 2:'frt5', 3:'frt6'}
def generate_iso2pose(folder_name, label, args):
    outdir = os.path.join(args.output_file, folder_name)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    vps = labels[label]
    print('vps: ', vps)
    converter = Model2SVG(width=args.width, height=args.height, tol=args.tol,
                      margin_left=args.margin_left, margin_top=args.margin_top,
                      line_width=args.line_width, line_width_hidden=args.line_width_hidden)
    try:
        # generate frt
        seconds = 10  # the worker can run 60 seconds. 
        timeout = Timeout(seconds) 
        timeout.start()
        fname = glob.glob(os.path.join(args.file, folder_name, '*.step'))[0]
        shp = read_step_file(fname)
        for vp in vps[:-1]:
            fname_out = os.path.join(outdir, '{}_{}.svg'.format(folder_name, vp))
            converter.export_shape_to_svg(shape=shp, filename=fname_out, proj_ax=converter.DIRS[vp])

        # generate answer
        fname_out = os.path.join(outdir, 'answer.svg')
        print("vp-1:", vps[-1])
        converter.export_shape_to_svg(shape=shp, filename=fname_out, proj_ax=converter.DIRS[vps[-1]])
        return True
    
    except Exception as re:
        shutil.rmtree(outdir)
        print('{} failed, due to: {}'.format(fname, re))
        return False


def main(args):
    """
    generate number of folders for task isometric2pose. Default number is 5,000
    """
    if not os.path.exists(args.output_file):
        os.mkdir(args.output_file)
    dirs = sorted(os.listdir(args.file))
    answer_name = np.arange(len(dirs))
    np.random.shuffle(answer_name)
    label = answer_name % 4

    f = partial(generate_iso2pose, args=args)
    t0 = time.time()
    
    p = pool.Pool(processes=args.n_cores)
    mask = p.starmap(f, zip(dirs, label))
    duration = time.time() - t0

    p.close()
    p.join()
    label_valid = [int(i) for i in label[mask]] 
    dirs_valid = np.array(dirs)[mask]
    answer = dict(zip(dirs_valid, label_valid))
    fname_answer = os.path.join(args.output_file, 'answer.json')
    with open(fname_answer, 'w') as ff:
        json.dump(answer, ff, indent=4)
    print('elapsed time = {}!'.format(time.strftime("%H:%M:%S", time.gmtime(duration))))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument('-f','--file',type=str,help='file or folder to be processed.')
    parser.add_argument('-o','--output_file',type=str,help='file or folder to save.')
    parser.add_argument('-v','--vps',type=str,default='ftr12345678',help='viewpoint(s) per file.')
    parser.add_argument('-n','--n_cores',type=int,default=cpu_count(),help='number of processors.')
    parser.add_argument('-W','--width',type=int,default=200,help='svg width.')
    parser.add_argument('-H','--height',type=int,default=200,help='svg height.')
    parser.add_argument('-t','--tol',type=float,default=0.04,help='svg discretization tolerance.')
    parser.add_argument('-ml','--margin_left',type=int,default=1,help='svg left margin.')
    parser.add_argument('-mt','--margin_top',type=int,default=1,help='svg top margin.')
    parser.add_argument('-lw','--line_width',type=float,default='0.7',help='svg line width.')
    parser.add_argument('-lwh','--line_width_hidden',type=float,default='0.35',help='svg hidden line width.')

    args = parser.parse_args(sys.argv[1:])
    main(args)
