'''
model2svg in SPARE3D

author  : cfeng
created : 3/21/20 6:09PM
'''
from math import radians
import os
import sys
import glob
import time
import argparse

from multiprocessing import pool, cpu_count
from functools import partial

from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Pnt2d, gp_Ax2
from OCC.Core.Bnd import Bnd_Box2d
from OCC.Extend.TopologyUtils import (discretize_edge, TopologyExplorer)
from OCC.Core.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
from OCC.Core.HLRAlgo import HLRAlgo_Projector
from OCC.Extend.DataExchange import read_step_file
from boolean import get_boundingbox
try:
    import svgwrite
    import svgwrite.shapes
except ImportError:
    print("svg exporter not available because the svgwrite package is not installed.")
    print("please use '$ conda install -c conda-forge svgwrite'")
    raise ImportError


class Model2SVG(object):

    def __init__(self, tol=0.03, unit="mm", width=800, height=600, margin_left=10, margin_top=10,
                 export_hidden_edges=True, color="black", color_hidden="red", line_width="0.7", line_width_hidden="0.35"):
        '''
        :param tol:
        :param unit:
        :param width: integers, specify the canva size in pixels
        :param height: integers, specify the canva size in pixels
        :param margin_left: integers, in pixel
        :param margin_top: integers, in pixel
        :param export_hidden_edges:
        :param color:
        :param color_hidden:
        :param line_width:
        :param line_width_hidden:
        '''
        self.TOL=tol
        self.UNIT=unit
        self.WIDTH=width
        self.HEIGHT=height
        self.MARGIN_LEFT=margin_left
        self.MARGIN_TOP=margin_top
        self.EXPORT_HIDDEN_EDGES=export_hidden_edges
        self.COLOR=color
        self.COLOR_HIDDEN=color_hidden
        self.LINE_WIDTH=line_width
        self.LINE_WIDTH_HIDDEN=line_width_hidden

        O = gp_Pnt(0,0,0)
        X = gp_Dir(1,0,0)
        Y = gp_Dir(0,1,0)
        nY = gp_Dir(0,-1,0)
        Z = gp_Dir(0,0,1)
        # P1, P1x = gp_Dir( 1,-1, 1), gp_Dir( 1, 1, 0)
        # P2, P2x = gp_Dir( 1, 1, 1), gp_Dir(-1, 1, 0)
        # P3, P3x = gp_Dir(-1, 1, 1), gp_Dir(-1,-1, 0)
        # P4, P4x = gp_Dir(-1,-1, 1), gp_Dir( 1,-1, 0)
        # P5, P5x = gp_Dir( 1,-1,-1), gp_Dir( 1, 1, 0)
        # P6, P6x = gp_Dir( 1, 1,-1), gp_Dir(-1, 1, 0)
        # P7, P7x = gp_Dir(-1, 1,-1), gp_Dir(-1,-1, 0)
        # P8, P8x = gp_Dir(-1,-1,-1), gp_Dir( 1,-1, 0)


        P1, P1x = gp_Dir(-1,-1, 1), gp_Dir( 1,-1, 0)
        P2, P2x = gp_Dir( 1,-1, 1), gp_Dir( 1, 1, 0)
        P3, P3x = gp_Dir( 1, 1, 1), gp_Dir(-1, 1, 0)
        P4, P4x = gp_Dir(-1, 1, 1), gp_Dir(-1,-1, 0)
        P5, P5x = gp_Dir(-1,-1,-1), gp_Dir( 1,-1, 0)
        P6, P6x = gp_Dir( 1,-1,-1), gp_Dir( 1, 1, 0)
        P7, P7x = gp_Dir( 1, 1,-1), gp_Dir(-1, 1, 0)
        P8, P8x = gp_Dir(-1, 1,-1), gp_Dir(-1,-1, 0)
        self.DIRS = {
            '1': gp_Ax2(O, P1, P1x),
            '2': gp_Ax2(O, P2, P2x),
            '3': gp_Ax2(O, P3, P3x),
            '4': gp_Ax2(O, P4, P4x),
            '5': gp_Ax2(O, P5, P5x),
            '6': gp_Ax2(O, P6, P6x),
            '7': gp_Ax2(O, P7, P7x),
            '8': gp_Ax2(O, P8, P8x),
            'f': gp_Ax2(O, nY, X),
            'r': gp_Ax2(O, X, Y),
            't': gp_Ax2(O, Z, X),
        }



    def _get_sorted_hlr_edges(self, topods_shape, ax=gp_Ax2(), export_hidden_edges=True,
                             use_smooth_edges=False, use_sewn_edges=False):
        """ Return hidden and visible edges as two lists of edges
        """
        hlr = HLRBRep_Algo()  
        
        hlr.Add(topods_shape) 

        projector = HLRAlgo_Projector(ax)

        hlr.Projector(projector)
        hlr.Update()
        hlr.Hide()

        hlr_shapes = HLRBRep_HLRToShape(hlr) 

        # visible edges
        visible = []

        visible_sharp_edges_as_compound = hlr_shapes.VCompound()
        if visible_sharp_edges_as_compound:
            visible += list(TopologyExplorer(visible_sharp_edges_as_compound).edges())

        visible_smooth_edges_as_compound = hlr_shapes.Rg1LineVCompound()
        if visible_smooth_edges_as_compound and use_smooth_edges:
            visible += list(TopologyExplorer(visible_smooth_edges_as_compound).edges())

        visible_sewn_edges_as_compound = hlr_shapes.RgNLineVCompound()
        if visible_sewn_edges_as_compound and use_sewn_edges:
           visible += list(TopologyExplorer(visible_sewn_edges_as_compound).edges())

        visible_contour_edges_as_compound = hlr_shapes.OutLineVCompound()
        if visible_contour_edges_as_compound:
            visible += list(TopologyExplorer(visible_contour_edges_as_compound).edges())

        #visible_isoparameter_edges_as_compound = hlr_shapes.IsoLineVCompound()
        #if visible_isoparameter_edges_as_compound:
        #    visible += list(TopologyExplorer(visible_isoparameter_edges_as_compound).edges())

        # hidden edges
        hidden = []
        if export_hidden_edges:
            hidden_sharp_edges_as_compound = hlr_shapes.HCompound()
            if hidden_sharp_edges_as_compound:
                hidden += list(TopologyExplorer(hidden_sharp_edges_as_compound).edges())

            hidden_smooth_edges_as_compound = hlr_shapes.Rg1LineHCompound()
            if hidden_smooth_edges_as_compound and use_smooth_edges:
                hidden += list(TopologyExplorer(hidden_smooth_edges_as_compound).edges())

            hidden_sewn_edges_as_compound = hlr_shapes.RgNLineHCompound()
            if hidden_sewn_edges_as_compound and use_sewn_edges:
                hidden += list(TopologyExplorer(hidden_sewn_edges_as_compound).edges())

            hidden_contour_edges_as_compound = hlr_shapes.OutLineHCompound()
            if hidden_contour_edges_as_compound:
                hidden += list(TopologyExplorer(hidden_contour_edges_as_compound).edges())

        return visible, hidden


    def _edge_to_svg_polyline(self, topods_edge):
        """ Returns a svgwrite.Path for the edge, and the 2d bounding box
        """
        unit = self.UNIT
        tol = self.TOL
        points_3d = discretize_edge(topods_edge, tol)
        points_2d = []
        box2d = Bnd_Box2d()

        for point in points_3d:
            # we tak only the first 2 coordinates (x and y, leave z)
            x_p = point[0]
            y_p = - point[1]
            box2d.Add(gp_Pnt2d(x_p, y_p))
            points_2d.append((x_p, y_p))

        return svgwrite.shapes.Polyline(points_2d, fill="none", class_='vectorEffectClass'), box2d


    def export_shape_to_svg(self, shape, filename=None, proj_ax=gp_Ax2(), scale=1, 
        verbose=True, max_eadge=1):
        """ export a single shape to an svg file and/or string.
        shape: the TopoDS_Shape to export
        filename (optional): if provided, save to an svg file
        proj_ax (optional): projection transformation
        """
        if shape.IsNull():
            raise AssertionError("shape is Null")

        # find all edges
        visible_edges, hidden_edges = self._get_sorted_hlr_edges(
            shape, ax=proj_ax, export_hidden_edges=self.EXPORT_HIDDEN_EDGES)

        # compute polylines for all edges
        # we compute a global 2d bounding box as well, to be able to compute
        # the scale factor and translation vector to apply to all 2d edges so that
        # they fit the svg canva
        global_2d_bounding_box = Bnd_Box2d()

        polylines = []
        polylines_hidden = []
        for visible_edge in visible_edges:
            visible_svg_line, visible_edge_box2d = self._edge_to_svg_polyline(visible_edge)
            polylines.append(visible_svg_line)
            global_2d_bounding_box.Add(visible_edge_box2d)
        if self.EXPORT_HIDDEN_EDGES:
            for hidden_edge in hidden_edges:
                hidden_svg_line, hidden_edge_box2d = self._edge_to_svg_polyline(hidden_edge)
                # # hidden lines are dashed style
                # hidden_svg_line.dasharray([5, 5])
                polylines_hidden.append(hidden_svg_line)
                global_2d_bounding_box.Add(hidden_edge_box2d)

        # translate and scale polylines

        # first compute shape translation and scale according to size/margins 
        x_min, y_min, x_max, y_max = global_2d_bounding_box.Get()
        bb2d_width = x_max - x_min
        bb2d_height = y_max - y_min
        # build the svg drawing
        dwg = svgwrite.Drawing(filename, (self.WIDTH, self.HEIGHT), debug=True)
        # adjust the view box so that the lines fit then svg canvas

        # for abc step file
        # dwg.viewbox(
        #     x_min - 1.4*max_eadge/2 + bb2d_width/2, 
        #     y_min - 1.4*max_eadge/2 + bb2d_height/2,
        #     1.4*max_eadge, 
        #     1.4*max_eadge)

        # for csg step file
        dwg.viewbox(
            x_min - 1.7*max_eadge/2 + bb2d_width/2, 
            y_min - 1.7*max_eadge/2 + bb2d_height/2,
            1.7*max_eadge, 
            1.7*max_eadge)

        # make sure line width stays constant
        # https://github.com/mozman/svgwrite/issues/38
        dwg.defs.add(dwg.style("""
        .vectorEffectClass {
            vector-effect: non-scaling-stroke;
}
"""))

        #draw hidden line first
        if self.EXPORT_HIDDEN_EDGES:
            for polyline in polylines_hidden:
                polyline.stroke(self.COLOR_HIDDEN, width=self.LINE_WIDTH_HIDDEN, linecap="round")
                dwg.add(polyline)
        for polyline in polylines:
            # apply color and style
            polyline.stroke(self.COLOR, width=self.LINE_WIDTH, linecap="round")
            # then adds the polyline to the svg canva
            dwg.add(polyline)

        # export to string or file according to the user choice
        if filename is not None:
            dwg.save()
            if not os.path.isfile(filename):
                raise AssertionError("svg export failed")
            if not verbose:
                print("Shape successfully exported to %s" % filename)
            return True
        return dwg.tostring()


def export_shape_to_svg_by_viewpoints(fname, viewpoints, args):
    converter = Model2SVG(width=args.width, height=args.height, tol=args.tol,
                          margin_left=args.margin_left, margin_top=args.margin_top,
                          line_width=args.line_width, line_width_hidden=args.line_width_hidden)
    print('export_shape_to_svg_by_viewpoints')
    try:

        shp = read_step_file(fname)
        boundbox = get_boundingbox(shp)#3D- return xmax, xmin, ymax, ymin, zmax, zmin, abs(xmax-xmin), abs(ymax-ymin), abs(zmax-zmin)
        max_3d_eadge = max(boundbox[6],boundbox[7],boundbox[8])
        sc=min(args.width, args.height)/max_3d_eadge

        fname_out = os.path.splitext(fname)[0] + '_{}.svg'
        for vp in viewpoints:
            converter.export_shape_to_svg(shape=shp, filename=fname_out.format(vp), proj_ax=converter.DIRS[vp], max_eadge = max_3d_eadge)
        return 1

    except Exception as re:
        print(fname + ' failed, due to: {}'.format(re))
        return 0


def main(args):
    if not os.path.exists(args.file):
        print('{} not exist!'.format(args.file))
        return

    if os.path.isdir(args.file):
        fd = args.file
        fnames = sorted([os.path.join(fd, v) for v in os.listdir(fd) if v.endswith('.step')])
        
    else:
        fnames = [args.file,]
      
  
    print('{} file(s) to be processed.'.format(len(fnames)))

    vps = [v for v in args.vps.lower() if v in ['f', 'r', 't']+['{}'.format(l) for l in range(1,9)]]
    print('viewpoint(s) = ' + ''.join(vps))

    
    p = pool.Pool(processes=args.n_cores)
    f = partial(export_shape_to_svg_by_viewpoints,viewpoints=vps,args=args)

    t0 = time.time()   

    rets = p.map(f, fnames)
    duration = time.time() - t0
    p.close()
    n_success = sum(rets)
    print('{} done,  {} failed, elapsed time = {}!'.format(n_success, len(rets)-n_success,
                                                  time.strftime("%H:%M:%S", time.gmtime(duration))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument('file',type=str, help='file or folder to be processed.')
    parser.add_argument('-v','--vps',type=str,default='ftr12345678',help='viewpoint(s) per file.')
    parser.add_argument('-n','--n_cores',type=int,default=cpu_count(),help='number of processors.')
    parser.add_argument('-W','--width',type=int,default=200,help='svg width.')
    parser.add_argument('-H','--height',type=int,default=200,help='svg height.')
    parser.add_argument('-t','--tol',type=float,default=0.04,help='svg discretization tolerance.')
    parser.add_argument('-ml','--margin_left',type=int,default=1,help='svg left margin.')
    parser.add_argument('-mt','--margin_top',type=int,default=1,help='svg top margin.')
    parser.add_argument('-lw','--line_width',type=str,default='0.7',help='svg line width.')
    parser.add_argument('-lwh','--line_width_hidden',type=str,default='0.35',help='svg hidden line width.')

    args = parser.parse_args(sys.argv[1:])

    args.script_folder = os.path.dirname(os.path.abspath(__file__))
    
    main(args)