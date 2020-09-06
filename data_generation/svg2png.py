import argparse
import os
import glob
import sys
import cairosvg
from PIL import Image
from cairosvg.surface import PNGSurface

from multiprocessing import pool, cpu_count
from functools import partial

# import pdb;pdb.set_trace()

def svg2png_transparent_background(svg_name, args):
    # import pdb;pdb.set_trace()

    png_name = svg_name.replace(".svg", '.png')
    with open(svg_name, 'rb') as svg_file:
        PNGSurface.convert(
            bytestring=svg_file.read(),
            write_to=open(png_name, 'wb'),
            output_width=int(args.width),
            output_height=int(args.height),
            )
    os.remove(svg_name)


def transparent_background2white(png_name):
    im = Image.open(png_name)

    fill_color = (255,255,255)  # new background color

    im = im.convert("RGBA")   # it had mode P after DL it from OP
    if im.mode in ('RGBA', 'LA'):
        background = Image.new(im.mode[:-1], im.size, fill_color)
        background.paste(im, im.split()[-1]) # omit transparency
        im = background

    im.convert("RGB").save(png_name)


def main(args):
    all_svg = glob.iglob(os.path.join(args.file, "**", "*.svg"), recursive=True)
    f1 = partial(svg2png_transparent_background, args=args)
    p = pool.Pool(processes=args.n_cores)
    p.map(f1, all_svg)
    p.close()
    p.join()

    all_png = glob.iglob(os.path.join(args.file, "**", "*.png"), recursive=True)
    # # f2 = partial(transparent_background2white, args=args)
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
    parser.add_argument('-f','--file',type=str,help='file or folder with svg files.')
    parser.add_argument('-W','--width',type=int,default=200,help='svg width.')
    parser.add_argument('-H','--height',type=int,default=200,help='svg height.')
    parser.add_argument('-n','--n_cores',type=int,default=cpu_count(),help='number of processors.')
    args = parser.parse_args(sys.argv[1:])
    main(args)

