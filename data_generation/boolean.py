from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeCone, BRepPrimAPI_MakeSphere
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
import math 
import random
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Pnt2d, gp_Ax2
from OCC.Core.BRepBndLib import brepbndlib_Add

from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
def get_boundingbox(shape, tol=1e-6, use_mesh=False):
    """ return the bounding box of the TopoDS_Shape `shape`
    Parameters
    ----------
    shape : TopoDS_Shape or a subclass such as TopoDS_Face
        the shape to compute the bounding box from
    tol: float
        tolerance of the computed boundingbox
    use_mesh : bool
        a flag that tells whether or not the shape has first to be meshed before the bbox
        computation. This produces more accurate results
    """
    bbox = Bnd_Box()
    bbox.SetGap(tol)
    if use_mesh:
        mesh = BRepMesh_IncrementalMesh()
        mesh.SetParallelDefault(True)
        mesh.SetShape(shape)
        mesh.Perform()
        if not mesh.IsDone():
            raise AssertionError("Mesh not done.")
    brepbndlib_Add(shape, bbox, use_mesh)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    return xmax, xmin, ymax, ymin, zmax, zmin, abs(xmax-xmin), abs(ymax-ymin), abs(zmax-zmin)


def cal_position(Xmax,Xmin,Ymax,Ymin,Zmax,Zmin):
    X_mean=(Xmax+Xmin)/2
    Y_mean=(Ymax+Ymin)/2
    Z_mean=(Zmax+Zmin)/2
    L=abs(Xmax-Xmin)
    W=abs(Ymax-Ymin)
    H=abs(Zmax-Zmin)
    x_position=random.uniform(X_mean-0.25*L,X_mean+0.25*L)
    y_position=random.uniform(Y_mean-0.25*W,Y_mean+0.25*W)
    z_position=random.uniform(Z_mean-0.25*H,Z_mean+0.25*H)
    return x_position,y_position,z_position



def New_shp(Xmax,Xmin,Ymax,Ymin,Zmax,Zmin,X,Y,Z):
    Index=random.randint(1,4)
    position=cal_position(Xmax,Xmin,Ymax,Ymin,Zmax,Zmin)
    A=(X+Y+Z)/5
    if Index == 1:
        X1 = random.uniform(0.5*A,A)
        Y1 = random.uniform(0.5*A,A)
        Z1 = random.uniform(0.5*A,A)
        nshp=BRepPrimAPI_MakeBox(gp_Pnt(-0.5*X1+position[0], -0.5*Y1+position[1], -0.5*Z1+position[2]),X1, Y1, Z1).Shape()
    if Index == 2:
        
        R = random.uniform(0.25*A,0.5*A)
        nshp=BRepPrimAPI_MakeSphere(gp_Pnt(position[0], position[1], position[2]), R).Shape()
    if Index == 3:
        R2 = random.uniform(0.25*A,0.5*A)
        H = random.uniform(0.5*A,A)
        origin = gp_Ax2(gp_Pnt(position[0], position[1], -0.5*H+position[2]), gp_Dir(0.0, 0.0, 1.0))
        nshp=BRepPrimAPI_MakeCone(origin,  R2,0, H).Shape()
    if Index == 4:
        
        R = random.uniform(0.25*A,0.5*A)
        H = random.uniform(0.5*A,A)
        cylinder_origin = gp_Ax2(gp_Pnt(position[0], position[1], -0.5*H+position[2]), gp_Dir(0.0, 0.0, 1.0))
        nshp=BRepPrimAPI_MakeCylinder(cylinder_origin, R, H).Shape()
    return nshp