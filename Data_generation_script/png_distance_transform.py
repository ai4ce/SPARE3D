import argparse
import numpy as np
import os
import cv2
import glob
import sys
import cairosvg
from scipy import ndimage
from PIL import Image
from cairosvg.surface import PNGSurface
import mahotas

from multiprocessing import pool, cpu_count
from functools import partial

def svg2png_transparent_background(svg_name, args):

    png_name = svg_name.replace(".svg", '.png') 

    with open(svg_name, 'rb') as svg_file:
        PNGSurface.convert(
            bytestring=svg_file.read(),
            write_to=open(png_name, 'wb'),
            
            )
    os.remove(svg_name)

def transparent_background2white(png_name):
    threshold = 200
    im = Image.open(png_name)

    width, height = im.size
    im = im.convert("L") 
    data = im.getdata()
    data = np.matrix(data, dtype=np.float32)
    # print(data.dtype)
    new_data = np.reshape(data, (height, width))
    new_data_binary = np.where(new_data > threshold , True, False)
    np.save("021.npy", new_data_binary)

    dmap = mahotas.distance(new_data_binary)
    new_im = Image.fromarray(dmap)
    new_im.convert("RGB").save(png_name)



def main(args):
    all_png = glob.iglob(os.path.join(args.file, "**", "*.png"), recursive=True)
    p = pool.Pool(processes=args.n_cores)
    p.map(transparent_background2white, all_png)
    p.close()
    p.join()
    # for svg in all_svg:
    #     # import pdb;pdb.set_trace()
    #     svg2png_transparent_background(svg, args)

    # for png in all_png:
    #     transparent_background2white(png)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument('-f','--file',type=str,default='/home/siyuan/project/yfx/SPARE3D/Data_generate_script/distance_transform/i2p', help='file or folder with svg files.')
    parser.add_argument('-W','--width',type=int,default=200,help='svg width.')
    parser.add_argument('-H','--height',type=int,default=200,help='svg height.')
    parser.add_argument('-n','--n_cores',type=int,default=cpu_count(),help='number of processors.')
    args = parser.parse_args(sys.argv[1:])
    main(args)

