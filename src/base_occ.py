import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import sys
import pickle
import json
import time
import os
import glob
import shutil
import datetime
import platform
from scipy.spatial import ConvexHull, Delaunay
from optparse import OptionParser
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('parso').setLevel(logging.ERROR)

from PyQt5.QtWidgets import QApplication, qApp
from PyQt5.QtWidgets import QDialog, QCheckBox
from PyQt5.QtWidgets import QFileDialog
# pip install PyQt5
# pip install --upgrade --force-reinstall PyQt5

from src.base import SetDir, plot2d, which, create_tempnum
from src.base_qtOCC import init_QDisplay
from src.OCCGui import init_qtdisplay

from OCC.Display.SimpleGui import init_display
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir
from OCC.Core.gp import gp_Ax1, gp_Ax2, gp_Ax3
from OCC.Core.gp import gp_XYZ
from OCC.Core.gp import gp_Lin, gp_Elips, gp_Pln
from OCC.Core.gp import gp_Mat, gp_GTrsf, gp_Trsf
from OCC.Core.gp import gp_Cylinder, gp_Pnt, gp_Vec, gp_Dir
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Compound, TopoDS_Face, topods_Face, topods_Solid
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TColgp import TColgp_Array1OfPnt, TColgp_Array2OfPnt
from OCC.Core.TColgp import TColgp_HArray1OfPnt, TColgp_HArray2OfPnt
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_SOLID, TopAbs_VERTEX, TopAbs_SHAPE
from OCC.Core.TopAbs import TopAbs_VERTEX
from OCC.Core.TopoDS import TopoDS_Iterator, topods_Vertex
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepProj import BRepProj_Projection
from OCC.Core.BRepFill import BRepFill_Filling
from OCC.Core.BRepFill import BRepFill_CurveConstraint
from OCC.Core.BRepTools import breptools_Read
from OCC.Core.BRepOffset import BRepOffset_MakeOffset, BRepOffset_Skin, BRepOffset_Interval
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_ThruSections, BRepOffsetAPI_MakeOffset, BRepOffsetAPI_MakeEvolved, BRepOffsetAPI_MakePipe, BRepOffsetAPI_MakePipeShell
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere, BRepPrimAPI_MakeBox
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_GTransform
from OCC.Core.BRepIntCurveSurface import BRepIntCurveSurface_Inter
from OCC.Core.BRepLProp import BRepLProp_SurfaceTool, BRepLProp_SLProps
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepBndLib import brepbndlib, brepbndlib_Add, brepbndlib_AddOBB, brepbndlib_AddOptimal
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.Bnd import Bnd_Box, Bnd_OBB
from OCC.Core.Geom import Geom_Plane, Geom_Surface, Geom_BSplineSurface
from OCC.Core.Geom import Geom_Curve, Geom_Line, Geom_Ellipse, Geom_Circle
from OCC.Core.Geom import Geom_RectangularTrimmedSurface, Geom_ToroidalSurface
from OCC.Core.Geom import Geom_Curve, Geom_Plane, Geom_Line
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf, GeomAPI_ProjectPointOnCurve
from OCC.Core.GeomAPI import geomapi
from OCC.Core.GeomAPI import GeomAPI_IntCS, GeomAPI_IntSS
from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface
from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
from OCC.Core.GeomAPI import GeomAPI_Interpolate
from OCC.Core.GeomAbs import GeomAbs_C0, GeomAbs_C1, GeomAbs_C2
from OCC.Core.GeomAbs import GeomAbs_G1, GeomAbs_G2
from OCC.Core.GeomAbs import GeomAbs_Intersection, GeomAbs_Arc
from OCC.Core.GeomFill import GeomFill_BoundWithSurf
from OCC.Core.GeomFill import GeomFill_BSplineCurves
from OCC.Core.GeomFill import GeomFill_StretchStyle, GeomFill_CoonsStyle, GeomFill_CurvedStyle
from OCC.Core.AIS import AIS_Manipulator
from OCC.Core.V3d import V3d_SpotLight, V3d_XnegYnegZpos, V3d_XposYposZpos
from OCC.Core.Graphic3d import Graphic3d_NOM_ALUMINIUM, Graphic3d_NOM_COPPER, Graphic3d_NOM_BRASS
from OCC.Core.Quantity import Quantity_Color, Quantity_NOC_WHITE, Quantity_NOC_CORAL2, Quantity_NOC_BROWN
from OCC.Core.BRepTools import breptools_Write
from OCC.Extend.DataExchange import write_step_file, read_step_file
from OCC.Extend.DataExchange import write_iges_file, read_iges_file
from OCC.Extend.DataExchange import write_stl_file, read_stl_file
from OCC.Extend.ShapeFactory import midpoint
from OCCUtils.Topology import Topo
from OCCUtils.Topology import shapeTypeString, dumpTopology
from OCCUtils.Construct import make_box, make_face, make_line, make_wire, make_edge
from OCCUtils.Construct import make_plane, make_polygon
from OCCUtils.Construct import point_to_vector, vector_to_point
from OCCUtils.Construct import dir_to_vec, vec_to_dir


def _gmsh_path():
    """Find Gmsh."""

    if os.name == "nt":
        gmp = which("gmsh.exe")
    else:
        gmp = which("gmsh")
    if gmp is None:
        print(
            "Could not find Gmsh."
            + "Interactive plotting and shapes module not available."
        )
    return gmp


def pnt_to_xyz(p):
    return p.X(), p.Y(), p.Z()


def float_to_string(number):
    if number == 0 or abs(np.log10(abs(number))) < 100:
        return ' {: 0.10E}'.format(number)
    else:
        return ' {: 0.10E}'.format(number).replace('E', '')


def rotate_xyz(axs=gp_Ax3(), deg=0.0, xyz="x"):
    if xyz == "x":
        ax1 = gp_Ax1(axs.Location(), axs.XDirection())
    elif xyz == "y":
        ax1 = gp_Ax1(axs.Location(), axs.YDirection())
    elif xyz == "z":
        ax1 = gp_Ax1(axs.Location(), axs.Direction())
    else:
        ax1 = gp_Ax1(axs.Location(), axs.Direction())
    axs.Rotate(ax1, np.deg2rad(deg))


def line_from_axs(axs=gp_Ax3(), length=100):
    return make_edge(axs.Location(), pnt_from_axs(axs, length))


def pnt_from_axs(axs=gp_Ax3(), length=100):
    vec = point_to_vector(axs.Location()) + \
        dir_to_vec(axs.Direction()) * length
    return vector_to_point(vec)


def pnt_trf_vec(pnt=gp_Pnt(), vec=gp_Vec()):
    v = point_to_vector(pnt)
    v.Add(vec)
    return vector_to_point(v)


def set_trf(ax1=gp_Ax3(), ax2=gp_Ax3()):
    trf = gp_Trsf()
    trf.SetTransformation(ax2, ax1)
    return trf


def set_loc(ax1=gp_Ax3(), ax2=gp_Ax3()):
    trf = set_trf(ax1, ax2)
    loc = TopLoc_Location(trf)
    return loc


def trsf_scale(axs=gp_Ax3(), scale=1):
    trf = gp_Trsf()
    trf.SetDisplacement(gp_Ax3(), axs)
    return trf


def gen_ellipsoid(axs=gp_Ax3(), rxyz=[10, 20, 30]):
    sphere = BRepPrimAPI_MakeSphere(gp_Ax2(), 1).Solid()
    loc = set_loc(gp_Ax3(), axs)
    mat = gp_Mat(
        rxyz[0], 0, 0,
        0, rxyz[1], 0,
        0, 0, rxyz[2]
    )
    gtrf = gp_GTrsf(mat, gp_XYZ(0, 0, 0))
    ellips = BRepBuilderAPI_GTransform(sphere, gtrf).Shape()
    ellips.Location(loc)
    return ellips


def get_axs(filename, ax=gp_Ax3()):
    dat = np.loadtxt(filename, skiprows=2)
    pnt = gp_Pnt(*dat[0])
    d_x = gp_Dir(*dat[1])
    d_y = gp_Dir(*dat[2])
    d_z = d_x.Crossed(d_y)
    axs = gp_Ax3(pnt, d_z, d_x)
    trf = gp_Trsf()
    trf.SetTransformation(ax, gp_Ax3())
    axs.Transform(trf)
    return axs


def get_diff(axis=gp_Ax3(), beam=gp_Ax3()):
    vx = dir_to_vec(axis.XDirection())
    vy = dir_to_vec(axis.YDirection())
    vz = dir_to_vec(axis.Direction())
    pln_x = Geom_Plane(axis.Location(), axis.YDirection())
    pln_y = Geom_Plane(axis.Location(), axis.XDirection())
    vec = dir_to_vec(beam.Direction())
    vec_p = gp_Pnt((gp_Vec(axis.Location().XYZ()) + vec).XYZ())
    pnt_x = GeomAPI_ProjectPointOnSurf(vec_p, pln_x).Point(1)
    pnt_y = GeomAPI_ProjectPointOnSurf(vec_p, pln_y).Point(1)
    vec_x = gp_Vec(axis.Location(), pnt_x)
    vec_y = gp_Vec(axis.Location(), pnt_y)
    deg_x = vec_x.AngleWithRef(vz, vy)
    deg_y = vec_y.AngleWithRef(vz, vx)
    p0 = axis.Location()
    p1 = beam.Location()
    print(axis.Location().Coord())
    print(beam.Location().Coord())
    txt = ""
    txt += "deg_x = {:+.2f}\n".format(np.rad2deg(deg_x))
    txt += "deg_y = {:+.2f}\n".format(np.rad2deg(deg_y))
    txt += "dist = {:.2f}\n".format(p0.Distance(p1))
    print(txt)


def occ_to_grasp_cor(axs, name="name", filename="pln.cor"):
    pnt = axs.Location()
    v_x = axs.XDirection()
    v_y = axs.YDirection()
    fp = open(filename, "w")
    fp.write(' {:s}\n'.format(name))
    fp.write(' {:s}\n'.format("mm"))
    fp.write(''.join([float_to_string(v) for v in pnt_to_xyz(pnt)]) + '\n')
    fp.write(''.join([float_to_string(v) for v in pnt_to_xyz(v_x)]) + '\n')
    fp.write(''.join([float_to_string(v) for v in pnt_to_xyz(v_y)]) + '\n')
    fp.close()


def occ_to_grasp_cor_ref(axs, ref=gp_Ax3(), name="name", filename="pln.cor"):
    trf = gp_Trsf()
    trf.SetTransformation(gp_Ax3(), ref)
    # trf.Invert()
    ax1 = axs.Transformed(trf)
    pnt = ax1.Location()
    v_x = ax1.XDirection()
    v_y = ax1.YDirection()
    fp = open(filename, "w")
    fp.write(' {:s}\n'.format(name))
    fp.write(' {:s}\n'.format("mm"))
    fp.write(''.join([float_to_string(v) for v in pnt_to_xyz(pnt)]) + '\n')
    fp.write(''.join([float_to_string(v) for v in pnt_to_xyz(v_x)]) + '\n')
    fp.write(''.join([float_to_string(v) for v in pnt_to_xyz(v_y)]) + '\n')
    fp.close()


def occ_to_grasp_rim(axs, pts, filename="pln.rim", name="name", nxy=5):
    pnt = axs.Location()
    trf = gp_Trsf()
    trf.SetTransformation(gp_Ax3(), axs)
    px, py = [], []
    for i in range(len(pts)):
        i0, i1 = i, (i + 1) % len(pts)
        p0, p1 = pts[i0].Transformed(trf), pts[i1].Transformed(trf)
        p_x = np.delete(np.linspace(p0.X(), p1.X(), nxy), -1)
        p_y = np.delete(np.linspace(p0.Y(), p1.Y(), nxy), -1)
        px.extend(p_x), py.extend(p_y)

    fp = open(filename, "w")
    fp.write(' {:s}\n'.format(name))
    fp.write('{:12d}{:12d}{:12d}\n'.format(len(px), 1, 1))
    #fp.write(' {:s}\n'.format("mm"))
    for i in range(len(px)):
        data = [px[i], py[i]]
        fp.write(''.join([float_to_string(val) for val in data]) + '\n')


def grasp_sfc(mesh, surf, sfc_file="surf.sfc"):
    fp = open(sfc_file, "w")
    ny, nx = surf.shape
    xs, xe = mesh[0][0, 0], mesh[0][0, -1]
    ys, ye = mesh[1][0, 0], mesh[1][-1, 0]
    fp.write(" {} data \n".format(sfc_file))
    fp.write(" {:.2e} {:.2e} {:.2e} {:.2e}\n".format(xs, ys, xe, ye))
    fp.write(" {:d} {:d}\n".format(nx, ny))
    for ix in range(nx):
        for iy in range(ny):
            fp.write(" {:.5e} ".format(surf[iy, ix]))
        fp.write("\n")
    fp.close()


def surf_spl_pcd(px, py, pz):
    nx, ny = px.shape
    pnt_2d = TColgp_Array2OfPnt(1, nx, 1, ny)
    for row in range(pnt_2d.LowerRow(), pnt_2d.UpperRow() + 1):
        for col in range(pnt_2d.LowerCol(), pnt_2d.UpperCol() + 1):
            i, j = row - 1, col - 1
            pnt = gp_Pnt(px[i, j], py[i, j], pz[i, j])
            pnt_2d.SetValue(row, col, pnt)

    curv = GeomAPI_PointsToBSplineSurface(
        pnt_2d, 3, 8, GeomAbs_G2, 0.001).Surface()
    surf = BRepBuilderAPI_MakeFace(curv, 1e-6).Face()
    return surf, pnt_2d


def spl_face(px, py, pz, axs=gp_Ax3()):
    nx, ny = px.shape
    pnt_2d = TColgp_Array2OfPnt(1, nx, 1, ny)
    for row in range(pnt_2d.LowerRow(), pnt_2d.UpperRow() + 1):
        for col in range(pnt_2d.LowerCol(), pnt_2d.UpperCol() + 1):
            i, j = row - 1, col - 1
            pnt = gp_Pnt(px[i, j], py[i, j], pz[i, j])
            pnt_2d.SetValue(row, col, pnt)
            #print (i, j, px[i, j], py[i, j], pz[i, j])

    api = GeomAPI_PointsToBSplineSurface(pnt_2d, 3, 8, GeomAbs_G2, 0.001)
    api.Interpolate(pnt_2d)
    #surface = BRepBuilderAPI_MakeFace(curve, 1e-6)
    # return surface.Face()
    face = BRepBuilderAPI_MakeFace(api.Surface(), 1e-6).Face()
    face.Location(set_loc(gp_Ax3(), axs))
    return face


def spl_curv(px, py, pz):
    num = px.size
    pts = []
    p_array = TColgp_Array1OfPnt(1, num)
    for idx, t in enumerate(px):
        x = px[idx]
        y = py[idx]
        z = pz[idx]
        pnt = gp_Pnt(x, y, z)
        pts.append(pnt)
        p_array.SetValue(idx + 1, pnt)
    api = GeomAPI_PointsToBSpline(p_array)
    return p_array, api.Curve()


def spl_curv_pts(pts=[gp_Pnt()]):
    num = len(pts)
    p_array = TColgp_Array1OfPnt(1, num)
    for idx, pnt in enumerate(pts):
        p_array.SetValue(idx + 1, pnt)
    api = GeomAPI_PointsToBSpline(p_array)
    return p_array, api.Curve()


def get_boundxyz_pts(pts, axs=gp_Ax3()):
    poly = make_polygon(pts, closed=True)
    poly.Location(set_loc(gp_Ax3(), axs))
    n_sided = BRepFill_Filling()
    for e in Topo(poly).edges():
        n_sided.Add(e, GeomAbs_C0)
    n_sided.Build()

    face = n_sided.Face()
    solid = BRepOffset_MakeOffset(
        face, 10.0, 1.0E-5, BRepOffset_Skin, False, True, GeomAbs_Arc, True, True)
    shape = solid.Shape()

    bbox = Bnd_Box()
    bbox.Set(axs.Location(), axs.Direction())
    brepbndlib_Add(shape, bbox)

    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    return [xmin, ymin, zmin, xmax, ymax, zmax]


def get_boundxyz_rim(rim, axs=gp_Ax3()):
    n_sided = BRepFill_Filling()
    for e in Topo(rim).edges():
        n_sided.Add(e, GeomAbs_C0)
    n_sided.Build()

    face = n_sided.Face()
    solid = BRepOffset_MakeOffset(
        face, 10.0, 1.0E-5, BRepOffset_Skin, False, True, GeomAbs_Arc, True, True)
    shape = solid.Shape()

    bbox = Bnd_Box()
    bbox.Set(axs.Location(), axs.Direction())
    brepbndlib_Add(shape, bbox)

    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    return [xmin, ymin, zmin, xmax, ymax, zmax]


def get_boundxyz_face(face, axs=gp_Ax3()):
    """
    Args:
        face (TopoDS_Shape): 
        axs (gp_Ax3): Defaults to gp_Ax3().

    Returns:
        [float x6]:
    """
    solid = BRepOffset_MakeOffset(
        face, 10.0, 1.0E-5, BRepOffset_Skin, False, True, GeomAbs_Arc, True, True)
    shape = solid.Shape()

    bbox = Bnd_Box()
    bbox.Set(axs.Location(), axs.XDirection())
    brepbndlib_Add(shape, bbox)

    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    return [xmin, ymin, zmin, xmax, ymax, zmax]


def get_aligned_boundingbox_ratio(shape, tol=1e-6, optimal_BB=True, ratio=1):
    """ return the bounding box of the TopoDS_Shape `shape`

    Parameters
    ----------

    shape : TopoDS_Shape or a subclass such as TopoDS_Face
        the shape to compute the bounding box from

    tol: float
        tolerance of the computed boundingbox

    use_triangulation : bool, True by default
        This makes the computation more accurate

    ratio : float, 1.0 by default.

    Returns
    -------
        if `as_pnt` is True, return a tuple of gp_Pnt instances
         for the lower and another for the upper X,Y,Z values representing the bounding box

        if `as_pnt` is False, return a tuple of lower and then upper X,Y,Z values
         representing the bounding box
    """
    bbox = Bnd_Box()
    bbox.SetGap(tol)

    # note: useTriangulation is True by default, we set it explicitely, but t's not necessary
    if optimal_BB:
        use_triangulation = True
        use_shapetolerance = True
        brepbndlib_AddOptimal(
            shape, bbox, use_triangulation, use_shapetolerance)
    else:
        brepbndlib_Add(shape, bbox)

    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    dx, mx = (xmax - xmin) * ratio, (xmax + xmin) / 2
    dy, my = (ymax - ymin) * ratio, (ymax + ymin) / 2
    dz, mz = (zmax - zmin) * ratio, (zmax + zmin) / 2
    x0, x1 = mx - dx / 2, mx + dx / 2
    y0, y1 = my - dy / 2, my + dy / 2
    z0, z1 = mz - dz / 2, mz + dz / 2
    corner1 = gp_Pnt(x0, y0, z0)
    corner2 = gp_Pnt(x1, y1, z1)
    center = midpoint(corner1, corner2)

    rim0 = make_polygon(
        [gp_Pnt(x0, y0, z0),
         gp_Pnt(x1, y0, z0),
         gp_Pnt(x1, y1, z0),
         gp_Pnt(x0, y1, z0)],
        closed=True
    )

    rim1 = make_polygon(
        [gp_Pnt(x0, y0, z1),
         gp_Pnt(x1, y0, z1),
         gp_Pnt(x1, y1, z1),
         gp_Pnt(x0, y1, z1)],
        closed=True
    )
    api = BRepOffsetAPI_ThruSections(True, False, 1.0E-9)
    api.AddWire(rim0)
    api.AddWire(rim1)
    box_shp = api.Shape()
    #box_shp = BRepPrimAPI_MakeBox(corner1, corner2).Shape()
    return center, [dx, dy, dz], box_shp


def get_oriented_boundingbox_ratio(shape, optimal_OBB=True, ratio=1.0):
    """ return the oriented bounding box of the TopoDS_Shape `shape`

    Parameters
    ----------

    shape : TopoDS_Shape or a subclass such as TopoDS_Face
        the shape to compute the bounding box from

    optimal_OBB : bool, True by default. If set to True, compute the
        optimal (i.e. the smallest oriented bounding box). 
        Optimal OBB is a bit longer.

    ratio : float, 1.0 by default.

    Returns
    -------
        a list with center, x, y and z sizes

        a shape
    """
    obb = Bnd_OBB()
    if optimal_OBB:
        is_triangulationUsed = True
        is_optimal = True
        is_shapeToleranceUsed = False
        brepbndlib_AddOBB(shape, obb, is_triangulationUsed,
                          is_optimal, is_shapeToleranceUsed)
    else:
        brepbndlib_AddOBB(shape, obb)

    # converts the bounding box to a shape
    aBaryCenter = obb.Center()
    aXDir = obb.XDirection()
    aYDir = obb.YDirection()
    aZDir = obb.ZDirection()
    aHalfX = obb.XHSize()
    aHalfY = obb.YHSize()
    aHalfZ = obb.ZHSize()
    dx = aHalfX * ratio
    dy = aHalfY * ratio
    dz = aHalfZ * ratio

    ax = gp_XYZ(aXDir.X(), aXDir.Y(), aXDir.Z())
    ay = gp_XYZ(aYDir.X(), aYDir.Y(), aYDir.Z())
    az = gp_XYZ(aZDir.X(), aZDir.Y(), aZDir.Z())
    p = gp_Pnt(aBaryCenter.X(), aBaryCenter.Y(), aBaryCenter.Z())
    anAxes = gp_Ax2(p, gp_Dir(aZDir), gp_Dir(aXDir))
    anAxes.SetLocation(gp_Pnt(p.XYZ() - ax * dx - ay * dy - az * dz))
    aBox = BRepPrimAPI_MakeBox(anAxes, 2.0 * dx, 2.0 * dy, 2.0 * dz).Shape()
    return aBaryCenter, [dx, dy, dz], aBox


def face_mesh_triangle(comp=TopoDS_Shape(), isR=0.1, thA=0.1):
    # Mesh the shape
    BRepMesh_IncrementalMesh(comp, isR, True, thA, True)
    bild1 = BRep_Builder()
    comp1 = TopoDS_Compound()
    bild1.MakeCompound(comp1)
    bt = BRep_Tool()
    ex = TopExp_Explorer(comp, TopAbs_FACE)
    while ex.More():
        face = topods_Face(ex.Current())
        location = TopLoc_Location()
        facing = bt.Triangulation(face, location)
        tab = facing.Nodes()
        tri = facing.Triangles()
        print(facing.NbTriangles(), facing.NbNodes())
        for i in range(1, facing.NbTriangles() + 1):
            trian = tri.Value(i)
            index1, index2, index3 = trian.Get()
            pnts = [tab.Value(index1), tab.Value(index2), tab.Value(index3)]
            poly = make_polygon(pnts, closed=True)
            ftri = make_face(poly, True)
            bild1.Add(comp1, poly)
            bild1.Add(comp1, ftri)
            # for j in range(1, 4):
            #    if j == 1:
            #        m = index1
            #        n = index2
            #    elif j == 2:
            #        n = index3
            #    elif j == 3:
            #        m = index2
            #    me = BRepBuilderAPI_MakeEdge(tab.Value(m), tab.Value(n))
            #    if me.IsDone():
            #        bild1.Add(comp1, me.Edge())
        ex.Next()
    return comp1


class GenCompound (object):

    def __init__(self):
        self.builder = BRep_Builder()
        self.compound = TopoDS_Compound()
        self.builder.MakeCompound(self.compound)

    def add_shapes(self, shps=[]):
        for shp in shps:
            self.builder.Add(self.compound, shp)


class Viewer (object):

    def __init__(self, disp=True):
        if disp == True:
            from OCC.Display.qtDisplay import qtViewer3d
            #self.app = self.get_app()
            #self.wi = self.app.topLevelWidgets()[0]
            self.vi = self.findChild(qtViewer3d, "qt_viewer_3d")
            #self.vi = self.wi.findChild(qtViewer3d, "qt_viewer_3d")
        self.selected_shape = []

    def get_app(self):
        app = QApplication.instance()
        #app = qApp
        # checks if QApplication already exists
        if not app:
            app = QApplication(sys.argv)
        return app

    def on_select(self):
        self.vi.sig_topods_selected.connect(self._on_select)

    def clear_selected(self):
        self.selected_shape = []

    def _on_select(self, shapes):
        """
        Parameters
        ----------
        shape : TopoDS_Shape
        """
        for shape in shapes:
            print()
            print(shape.Location().Transformation())
            # print(shape.Location().Transformation().TranslationPart().Coord())
            self.selected_shape.append(shape)
            self.DumpTop(shape)

    def make_comp_selcted(self):
        bild = BRep_Builder()
        comp = TopoDS_Compound()
        bild.MakeCompound(comp)
        for shp in self.selected_shape:
            print(shp)
            bild.Add(comp, shp)
        return comp

    def DumpJson(self, shp):
        jsonname = create_tempnum(self.rootname, self.tmpdir, ".json")
        fp = open(jsonname, "w")
        shp.DumpJson(fp)
        fp.close()

    def DumpTop(self, shape, level=0):
        """
        Print the details of an object from the top down
        """
        brt = BRep_Tool()
        s = shape.ShapeType()
        if s == TopAbs_VERTEX:
            pnt = brt.Pnt(topods_Vertex(shape))
            dmp = " " * level
            dmp += "%s - " % shapeTypeString(shape)
            dmp += "%.5e %.5e %.5e" % (pnt.X(), pnt.Y(), pnt.Z())
            print(dmp)
        else:
            dmp = " " * level
            dmp += shapeTypeString(shape)
            print(dmp)
        it = TopoDS_Iterator(shape)
        while it.More():
            shp = it.Value()
            it.Next()
            self.DumpTop(shp, level + 1)


class OCCApp(plot2d, init_QDisplay, Viewer):

    def __init__(self, temp=True, disp=True, touch=False):
        plot2d.__init__(self, temp=temp)
        self.base_axs = gp_Ax3()
        self.disp = disp
        self.touch = touch
        self.colors = ["BLUE1", "RED", "GREEN",
                       "YELLOW", "BLACK", "WHITE", "BROWN"]

        # OCC Viewer
        if disp == True:
            #self.display, self.start_display, self.add_menu, self.add_function, self.wi = init_qtdisplay()
            init_QDisplay.__init__(self)
            if touch == True:
                Viewer.__init__(self, disp=True)
                self.on_select()

            self.SaveMenu()
            self.ViewMenu()
            self.SelectMenu()
            self.SelectMesh()
        else:
            Viewer.__init__(self, disp=False)

        # GMSH
        self.gmsh = _gmsh_path()

    def AddManipulator(self):
        self.manip = AIS_Manipulator(self.base_axs.Ax2())
        ais_shp = self.display.DisplayShape(
            self.base_axs.Location(),
            update=True
        )
        self.manip.Attach(ais_shp)

    def SaveMenu(self):
        self.add_menu("File")
        self.add_function("File", self.import_cadfile)
        self.add_function("File", self.export_cap)
        if self.touch == True:
            self.add_function("File", self.export_stp_selected)
            self.add_function("File", self.export_stl_selected)
            self.add_function("File", self.export_igs_selected)
            self.add_function("File", self.export_brep_selected)
            self.add_function("File", self.clear_selected)
        self.add_function("File", self.open_newtempdir)
        self.add_function("File", self.open_tempdir)
        self.add_function("File", self.exit_win)

    def ViewMenu(self):
        self.add_menu("View")
        self.add_function("View", self.display.FitAll)
        self.add_function("View", self.display.View_Top)  # XY-Plane(+)
        self.add_function("View", self.display.View_Bottom)  # XY-Plane(-)
        self.add_function("View", self.display.View_Rear)  # XZ-Plane(+)
        self.add_function("View", self.display.View_Front)  # XZ-Plane(-)
        self.add_function("View", self.display.View_Right)  # YZ-Plane(+)
        self.add_function("View", self.display.View_Left)  # YZ-Plane(-)
        self.add_function("View", self.View_XZPlane)
        self.add_function("View", self.View_YZPlane)
        self.add_function("View", self.ray_tracing_mode)
        self.add_function("View", self.display.SetRasterizationMode)
        self.add_function("View", self.get_view_dir)
        self.add_function("View", self.save_view_axis)

    def SelectMenu(self):
        self.add_menu("Select")
        self.add_function("Select", self.display.SetSelectionModeVertex)
        self.add_function("Select", self.display.SetSelectionModeEdge)
        self.add_function("Select", self.display.SetSelectionModeFace)
        self.add_function("Select", self.SetSelectionModeShape)
        self.add_function("Select", self.display.SetSelectionModeNeutral)

    def SelectMesh(self):
        self.add_menu("Mesh")
        self.add_function("Mesh", self.gen_aligned_bounded_box)
        self.add_function("Mesh", self.gen_oriented_bounded_box)
        self.add_function("Mesh", self.gen_mesh_face)

    def SetSelectionModeShape(self):
        self.display.SetSelectionMode(TopAbs_SHAPE)

    def get_view_dir(self):
        print()
        x, y, z = self.display.View.At()
        print("Pos: ", x, y, z)
        x, y, z = self.display.View.Eye()
        print("Eye: ", x, y, z)
        x, y, z = self.display.View.Proj()
        print("Prj: ", x, y, z)
        x, y, z = self.display.View.Up()
        print("Up : ", x, y, z)

    def save_view_axis(self):
        print()
        x, y, z = self.display.View.At()
        p = gp_Pnt(x, y, z)
        print("Pos: ", x, y, z)
        x, y, z = self.display.View.Eye()
        dz = gp_Dir(x, y, z)
        print("Eye: ", x, y, z)
        x, y, z = self.display.View.Proj()
        dx = gp_Dir(x, y, z)
        print("Prj: ", x, y, z)
        x, y, z = self.display.View.Up()
        print("Up : ", x, y, z)
        axs = gp_Ax3(p, dz, dx)
        cor_name = create_tempnum(self.rootname, self.tmpdir, ".cor")
        occ_to_grasp_cor(axs, name="axis", filename=cor_name)

    def View_YZPlane(self):
        self.display.View.SetProj(1, 0, 0)
        self.display.View.SetUp(0, 1, 0)

    def View_XZPlane(self):
        self.display.View.SetProj(0, -1, 0)
        self.display.View.SetUp(1, 0, 0)

    def gen_mesh_face(self):
        self.export_cap()
        comp = self.make_comp_selcted()
        mesh = face_mesh_triangle(comp, 0.1, 0.1)
        self.display.DisplayShape(mesh, update=True, transparency=0.9)
        self.export_cap()

    def gen_aligned_bounded_box(self):
        comp = self.make_comp_selcted()
        c, dxyz, box = get_aligned_boundingbox_ratio(comp, ratio=1.0)
        if self.disp == True:
            self.display.DisplayShape(box, transparency=0.9, update=True)
            self.export_cap()
        return c, dxyz, box

    def gen_oriented_bounded_box(self):
        comp = self.make_comp_selcted()
        c, dxyz, box = get_oriented_boundingbox_ratio(comp, ratio=1.0)
        if self.disp == True:
            self.display.DisplayShape(box, transparency=0.9, update=True)
            self.export_cap()
        return c, dxyz, box

    def ray_tracing_mode(self):
        # create one spotlight
        spot_light = V3d_SpotLight(
            gp_Pnt(-1000, -1000, 1000),
            V3d_XnegYnegZpos,
            Quantity_Color(Quantity_NOC_WHITE)
        )
        # display the spotlight in rasterized mode
        self.display.Viewer.AddLight(spot_light)
        self.display.View.SetLightOn()
        self.display.SetRaytracingMode(depth=8)

        # pythonocc-core=7.4.0
        # TKOpenGl | Type: Error | ID: 0 | Severity: High | Message:
        # Ray-tracing requires OpenGL 4.0+ or GL_ARB_texture_buffer_object_rgb32 extension

        # pythonocc-core=7.4.1
        # RuntimeError: Aspect_GraphicDeviceDefinitionErrorOpenGl_Window::CreateWindow: SetPixelFormat failed.
        # Error code: 2000 raised from method Init of class Display3d

    def import_geofile(self, geofile, tol=0.1):
        # msh1, msh2, msh22, msh3, msh4, msh40, msh41, msh,
        # unv, vtk, wrl, mail, stl, p3d, mesh, bdf, cgns, med,
        # diff, ir3, inp, ply2, celum, su2, x3d, dat, neu, m, key
        geo_dir = os.path.dirname(geofile)
        geo_base = os.path.basename(geofile)
        geo_name = create_tempnum(self.rootname, self.tmpdir, ".geo")
        geo_file = os.path.basename(geo_name)
        shutil.copyfile(geofile, geo_name)
        stl_name, _ = os.path.splitext(self.tmpdir + geo_file)
        stl_name += ".stl"
        stl_name = os.path.abspath(stl_name)
        txt_name, _ = os.path.splitext(self.tmpdir + geo_file)
        txt_name += "_gmsh.txt"
        txt_name = os.path.abspath(txt_name)

        gmsh_run = self.gmsh
        gmsh_run += " -tol {:.2f}".format(tol)
        gmsh_run += " " + geo_base
        gmsh_run += " -3 -o"
        gmsh_run += " {} -format stl".format(stl_name)
        gmsh_run += " -log {}".format(txt_name)

        os.chdir(geo_dir)
        gmsh_success = os.system(gmsh_run)
        os.chdir(self.root_dir)
        return stl_name

    def read_cadfile_axs(self, fileName, axs=gp_Ax1()):
        surf = self.read_cadfile(fileName, disp=False)
        trf = gp_Trsf()
        trf.SetTransformation(axs, gp_Ax3())
        surf.Location(TopLoc_Location(trf))
        return surf

    def read_cadfile(self, fileName, disp=True, col=None, trs=None):
        filesize = os.path.getsize(fileName)
        print(fileName, filesize / 1000, " kB")
        base_dir = os.path.dirname(fileName)
        basename = os.path.basename(fileName)
        rootname, extname = os.path.splitext(fileName)
        if extname in [".stp", ".step"]:
            shpe = read_step_file(fileName)
        elif extname in [".igs", ".iges"]:
            shpe = read_iges_file(fileName)
        elif extname in [".stl"]:
            shpe = read_stl_file(fileName)
        elif extname in [".brep"]:
            shpe = TopoDS_Shape()
            builder = BRep_Builder()
            breptools_Read(shpe, fileName, builder)
        elif extname in [".geo"]:
            stlfile = self.import_geofile(fileName, 0.1)
            shpe = read_stl_file(stlfile)
        else:
            print("Incorrect file index")
            # sys.exit(0)

        if disp == True:
            self.display.DisplayShape(
                shpe, update=True, color=col, transparency=trs)
        return shpe

    def import_cadfile(self):
        options = QFileDialog.Options()
        fileNames, _ = QFileDialog.getOpenFileNames(self, 'QFileDialog.getOpenFileName()', '',
                                                    'CAD files (*.stp *.step *.stl *.igs *.iges, *.brep. *.geo)',
                                                    options=options)
        for fileName in fileNames:
            print(fileName)
            self.read_cadfile(fileName, disp=True, trs=None)
            self.export_cap()

    def export_stl(self, shp, filename=None):
        if filename == None:
            filename = create_tempnum(self.rootname, self.tmpdir, ".stl")
        write_stl_file(shp, filename)

    def export_stp(self, shp, filename=None):
        if filename == None:
            filename = create_tempnum(self.rootname, self.tmpdir, ".stp")
        write_step_file(shp, filename)

    def export_igs(self, shp, filename=None):
        if filename == None:
            filename = create_tempnum(self.rootname, self.tmpdir, ".igs")
        write_iges_file(shp, filename)

    def export_cap_name(self, pngname=None):
        if pngname == None:
            pngname = create_tempnum(self.rootname, self.tmpdir, ".png")
        self.display.View.Dump(pngname)

    def export_cap(self):
        pngname = create_tempnum(self.rootname, self.tmpdir, ".png")
        self.display.View.Dump(pngname)

    def export_stp_selected(self):
        comp = self.make_comp_selcted()
        self.export_stp(comp)

    def export_stl_selected(self):
        comp = self.make_comp_selcted()
        stlname = create_tempnum(self.rootname, self.tmpdir, ".stl")
        write_stl_file(comp, stlname)

    def export_igs_selected(self):
        comp = self.make_comp_selcted()
        igsname = create_tempnum(self.rootname, self.tmpdir, ".igs")
        write_iges_file(comp, igsname)

    def export_brep_selected(self):
        comp = self.make_comp_selcted()
        brepname = create_tempnum(self.rootname, self.tmpdir, ".brep")
        breptools_Write(comp, brepname)

    def save_view(self, num="0"):
        self.display.View_Top()
        self.display.FitAll()
        self.display.View.Dump(self.tempname + "_XY" + num + ".png")

        self.display.View_Rear()
        self.display.FitAll()
        self.display.View.Dump(self.tempname + "_XZ" + num + ".png")

        self.display.View_Right()
        self.display.FitAll()
        self.display.View.Dump(self.tempname + "_YZ" + num + ".png")

        self.display.View.SetProj(V3d_XposYposZpos)
        self.display.FitAll()
        self.display.View.Dump(self.tempname + "_XYZ" + num + ".png")

    def exit_win(self):
        self.close()

    def ShowOCC(self):
        self.display.FitAll()
        self.display.View.Dump(self.tempname + ".png")
        self.start_display()


class dispocc (OCCApp):

    def __init__(self, temp=True, disp=True, touch=False):
        OCCApp.__init__(self, temp, disp, touch)

        # self._key_map = {ord('W'): self._display.SetModeWireFrame,
        #                  ord('S'): self._display.SetModeShaded,
        #                  ord('A'): self._display.EnableAntiAliasing,
        #                  ord('B'): self._display.DisableAntiAliasing,
        #                  ord('H'): self._display.SetModeHLR,
        #                  ord('F'): self._display.FitAll,
        #                  ord('G'): self._display.SetSelectionMode}

        # def keyPressEvent(self, event):
        #     code = event.key()
        #     if code in self._key_map:
        #         self._key_map[code]()
        #     elif code in range(256):
        #         log.info('key: "%s"(code %i) not mapped to any function' % (chr(code), code))
        #     else:
        #         log.info('key: code %i not mapped to any function' % code)

    def reload_app(self, temp=True, disp=True, touch=False):
        OCCApp.__init__(self, temp, disp, touch)

    def show_box(self, axs=gp_Ax3(), lxyz=[100, 100, 100]):
        box = make_box(*lxyz)
        ax1 = gp_Ax3(
            gp_Pnt(-lxyz[0] / 2, -lxyz[1] / 2, -lxyz[2] / 2),
            gp_Dir(0, 0, 1)
        )
        trf = gp_Trsf()
        trf.SetTransformation(axs, gp_Ax3())
        trf.SetTransformation(ax1, gp_Ax3())
        box.Location(TopLoc_Location(trf))
        self.display.DisplayShape(axs.Location())
        self.show_axs_pln(axs, scale=lxyz[0])
        self.display.DisplayShape(box, transparency=0.7)

    def show_pnt(self, xyz=[0, 0, 0]):
        self.display.DisplayShape(gp_Pnt(*xyz))

    def show_pts(self, pts=[gp_Pnt()], num=1):
        for p in pts[::num]:
            self.display.DisplayShape(p)
        self.display.DisplayShape(make_polygon(pts))

    def show_ball(self, scale=100, trans=0.5):
        shape = BRepPrimAPI_MakeSphere(scale).Shape()
        self.display.DisplayShape(shape, transparency=trans)

    def show_vec(self, beam=gp_Ax3(), scale=1.0):
        pnt = beam.Location()
        vec = dir_to_vec(beam.Direction()).Scaled(scale)
        print(vec.Magnitude())
        self.display.DisplayVector(vec, pnt)

    def show_ellipsoid(self, axs=gp_Ax3(), rxyz=[10., 10., 10.], trans=0.5):
        shape = gen_ellipsoid(axs, rxyz)
        self.display.DisplayShape(shape, transparency=trans, color="BLUE1")
        return shape

    def show_axs_vec(self, beam=gp_Ax3(), scale=1.0):
        pnt = beam.Location()
        vec = dir_to_vec(beam.Direction()).Scaled(scale)
        self.display.DisplayVector(vec, pnt)

    def show_axs_pln(self, axs=gp_Ax3(), scale=100, name=None):
        pnt = axs.Location()
        dx = axs.XDirection()
        dy = axs.YDirection()
        dz = axs.Direction()
        vx = dir_to_vec(dx).Scaled(1 * scale)
        vy = dir_to_vec(dy).Scaled(1 * scale)
        vz = dir_to_vec(dz).Scaled(1 * scale)

        pnt_x = pnt_trf_vec(pnt, vx)
        pnt_y = pnt_trf_vec(pnt, vy)
        pnt_z = pnt_trf_vec(pnt, vz)
        lx, ly, lz = make_line(pnt, pnt_x), make_line(
            pnt, pnt_y), make_line(pnt, pnt_z)
        self.display.DisplayShape(pnt)
        self.display.DisplayShape(lx, color="RED")
        self.display.DisplayShape(ly, color="GREEN")
        self.display.DisplayShape(lz, color="BLUE1")
        if name != None:
            self.display.DisplayMessage(axs.Location(), name)
        return [lx, ly, lz]

    def show_plane(self, axs=gp_Ax3(), scale=100):
        pnt = axs.Location()
        vec = dir_to_vec(axs.Direction())
        pln = make_plane(pnt, vec, -scale, scale, -scale, scale)
        self.display.DisplayShape(pln)

    def show_wire(self, pts=[], axs=gp_Ax3()):
        poly = make_polygon(pts)
        poly.Location(set_loc(gp_Ax3(), axs))

        n_sided = BRepFill_Filling()
        for e in Topo(poly).edges():
            n_sided.Add(e, GeomAbs_C0)
        # n_sided.Build()
        #face = n_sided.Face()
        self.display.DisplayShape(poly)
        return poly

    def prop_axs(self, axs=gp_Ax3(), scale=100, xyz="z"):
        if xyz == "x":
            vec = dir_to_vec(axs.XDirection()).Scaled(scale)
        elif xyz == "y":
            vec = dir_to_vec(axs.YDirection()).Scaled(scale)
        elif xyz == "z":
            vec = dir_to_vec(axs.Direction()).Scaled(scale)
        else:
            vec = dir_to_vec(axs.Direction()).Scaled(scale)
        return axs.Translated(vec)

    def make_plane_axs(self, axs=gp_Ax3(), rx=[0, 500], ry=[0, 500]):
        pln = BRepBuilderAPI_MakeFace(
            gp_Pln(axs),
            rx[0], rx[1], ry[0], ry[1]
        ).Face()
        return pln

    def make_circle(self, axs=gp_Ax3(), radi=100):
        return make_wire(make_edge(Geom_Circle(axs.Ax2(), radi)))

    def make_torus(self, axs=gp_Ax3(), r0=6000, r1=1500):
        tok_surf = Geom_ToroidalSurface(axs, r0, r1)
        return make_face(tok_surf, 1.0E-9)

    def make_cylinder_surf(self, axs=gp_Ax3(), radii=700, hight=500, rng=[0, 0.1], xyz="y"):
        loc = self.prop_axs(axs, radii, "z")
        if xyz == "y":
            rotate_xyz(loc, deg=90, xyz="y")
        elif xyz == "x":
            rotate_xyz(loc, deg=90, xyz="x")
            rotate_xyz(loc, deg=-90, xyz="z")
        else:
            loc = self.prop_axs(loc, -radii, "z")
            #loc = self.prop_axs(loc, -radii, "x")

        face = BRepBuilderAPI_MakeFace(
            gp_Cylinder(loc, radii),
            rng[0], rng[1],
            -hight / 2, hight / 2
        ).Face()
        return face

    def make_trimmedcylinder(self, axs=gp_Ax1(), radii=700, hight=500, rng=[0, 0.1], xyz="y"):
        loc = self.prop_axs(axs, radii, "z")
        if xyz == "y":
            rotate_xyz(loc, deg=90, xyz="y")
        elif xyz == "x":
            rotate_xyz(loc, deg=90, xyz="x")
            rotate_xyz(loc, deg=-90, xyz="z")
        else:
            loc = self.prop_axs(loc, -radii, "z")
            #loc = self.prop_axs(loc, -radii, "x")

        face = BRepBuilderAPI_MakeFace(
            gp_Cylinder(loc, radii),
            rng[0], rng[1],
            -hight / 2, hight / 2
        ).Face()
        return face

    def make_EllipWire(self, rxy=[1.0, 1.0], shft=0.0, axs=gp_Ax3()):
        rx, ry = rxy
        if rx > ry:
            major_radi = rx
            minor_radi = ry
            axis = gp_Ax2()
            axis.SetXDirection(axis.XDirection())
        else:
            major_radi = ry
            minor_radi = rx
            axis = gp_Ax2()
            axis.SetXDirection(axis.YDirection())
        axis.Rotate(axis.Axis(), np.deg2rad(shft))
        elip = make_edge(gp_Elips(axis, major_radi, minor_radi))
        poly = make_wire(elip)
        poly.Location(set_loc(gp_Ax3(), axs))
        return poly

    def make_PolyWire(self, num=6, radi=1.0, shft=0.0, axs=gp_Ax3(), skin=None):
        lxy = radi
        pnts = []
        angl = 360 / num
        for i in range(num):
            thet = np.deg2rad(i * angl) + np.deg2rad(shft)
            x, y = radi * np.sin(thet), radi * np.cos(thet)
            pnts.append(gp_Pnt(x, y, 0))
        pnts.append(pnts[0])
        poly = make_polygon(pnts)
        poly.Location(set_loc(gp_Ax3(), axs))

        n_sided = BRepFill_Filling()
        for e in Topo(poly).edges():
            n_sided.Add(e, GeomAbs_C0)
        n_sided.Build()
        face = n_sided.Face()
        if skin == None:
            return poly
        elif skin == 0:
            return face
        else:
            solid = BRepOffset_MakeOffset(
                face, skin, 1.0E-5, BRepOffset_Skin, False, True, GeomAbs_Arc, True, True)
            return solid.Shape()

    def make_StarWire(self, num=5, radi=[2.0, 1.0], shft=0.0, axs=gp_Ax3(), skin=None):
        lxy = radi
        pnts = []
        angl = 360 / num
        for i in range(num):
            a_thet = np.deg2rad(i * angl) + np.deg2rad(shft)
            ax, ay = radi[0] * np.sin(a_thet), radi[0] * np.cos(a_thet)
            pnts.append(gp_Pnt(ax, ay, 0))
            b_thet = a_thet + np.deg2rad(angl) / 2
            bx, by = radi[1] * np.sin(b_thet), radi[1] * np.cos(b_thet)
            pnts.append(gp_Pnt(bx, by, 0))
        pnts.append(pnts[0])
        poly = make_polygon(pnts)
        poly.Location(set_loc(gp_Ax3(), axs))

        n_sided = BRepFill_Filling()
        for e in Topo(poly).edges():
            n_sided.Add(e, GeomAbs_C0)
        n_sided.Build()
        face = n_sided.Face()
        if skin == None:
            return poly
        elif skin == 0:
            return face
        else:
            solid = BRepOffset_MakeOffset(
                face, skin, 1.0E-5, BRepOffset_Skin, False, True, GeomAbs_Arc, True, True)
            return solid.Shape()

    def make_Wire_pts(self, dat=[], axs=gp_Ax3()):
        num = dat.shape
        pts = []
        if num[1] == 2:
            for p in dat:
                pts.append(gp_Pnt(p[0], p[1], 0))
        elif num[1] == 3:
            for p in dat:
                pts.append(gp_Pnt(p[0], p[1], p[2]))
        else:
            for p in dat:
                pts.append(gp_Pnt(p[0], p[1], p[2]))
        pts = np.array(pts)
        #cov = ConvexHull(pts, qhull_options='QJ')

        #pts_ord = []
        # print(cov)
        # print(cov.simplices)
        # print(cov.vertices)
        # for idx in cov.vertices:
        #    print(idx, pnt[idx])
        #    pts_ord.append(gp_Pnt(*pnt[idx]))

        #poly = make_polygon(pts_ord)
        poly = make_polygon(pts)
        poly.Location(set_loc(gp_Ax3(), axs))
        #n_sided = BRepFill_Filling()
        # for e in Topo(poly).edges():
        #    n_sided.Add(e, GeomAbs_C0)
        # n_sided.Build()
        #face = n_sided.Face()
        return poly

    def make_skin(self, pts=[], axs=gp_Ax3(), skin=1.0):
        poly = make_polygon(pts, closed=True)
        poly.Location(set_loc(gp_Ax3(), axs))

        n_sided = BRepFill_Filling()
        for e in Topo(poly).edges():
            n_sided.Add(e, GeomAbs_C0)
        n_sided.Build()
        face = n_sided.Face()
        solid = BRepOffset_MakeOffset(
            face, skin, 1.0E-5, BRepOffset_Skin, False, True, GeomAbs_Arc, True, True)
        return solid.Shape()

    def make_skin_wire(self, poly, axs=gp_Ax3(), skin=1.0):
        n_sided = BRepFill_Filling()
        for e in Topo(poly).edges():
            n_sided.Add(e, GeomAbs_C0)
        n_sided.Build()
        face = n_sided.Face()
        if skin == None:
            return poly
        elif skin == 0:
            return face
        else:
            solid = BRepOffset_MakeOffset(
                face, skin, 1.0E-5, BRepOffset_Skin, False, True, GeomAbs_Arc, True, True)
            return solid.Shape()

    def make_FaceByOrder(self, pts=[]):
        dat = []
        for p in pts:
            dat.append([p.X(), p.Y(), p.Z()])

        dat = np.array(dat)
        cov = ConvexHull(dat, qhull_options='QJ')

        #pts_ord = []
        # print(cov)
        # print(cov.simplices)
        # print(cov.vertices)
        # for idx in cov.vertices:
        #    print(idx, pnt[idx])
        #    pts_ord.append(gp_Pnt(*pnt[idx]))

        #poly = make_polygon(pts_ord)
        poly = make_polygon(pts)
        n_sided = BRepFill_Filling()
        for e in Topo(poly).edges():
            n_sided.Add(e, GeomAbs_C0)
        n_sided.Build()
        face = n_sided.Face()
        return dat, face

    def proj_rim_pln(self, wire, surf, axs=gp_Ax3()):
        proj = BRepProj_Projection(wire, surf, axs.Direction())
        return proj.Current()

    def proj_pnt_pln(self, pnt, surf, axs=gp_Ax3()):
        lin = gp_Lin(pnt, axs.Direction())
        api = GeomAPI_IntCS(Geom_Line(lin), BRep_Tool.Surface(surf))
        dst = np.inf
        sxy = pnt
        num = api.NbPoints()
        for i in range(num):
            p0 = api.Point(i + 1)
            dst0 = pnt.Distance(p0)
            if dst0 < dst:
                dst = dst0
                sxy = p0
        return sxy

    def proj_pln_show(self, face, nxy=[10, 10], ux=[0, 1], uy=[0, 1], axs=gp_Ax3()):
        trf = set_trf(gp_Ax3(), axs)
        pln = self.make_plane_axs(axs, [-1000, 1000], [-1000, 1000])
        surf = BRep_Tool.Surface(face)
        for px in np.linspace(ux[0], ux[1], nxy[0]):
            for py in np.linspace(uy[0], uy[1], nxy[1]):
                p0 = surf.Value(px, py)
                p1 = self.proj_pnt_pln(p0, pln, axs)
                self.display.DisplayShape(p0)
                self.display.DisplayShape(p1)

    def proj_pln_showup(self, face, nxy=[10, 10], lx=[-10, 10], ly=[-10, 10], axs=gp_Ax3()):
        trf = set_trf(gp_Ax3(), axs)
        nx, ny = nxy
        xs, xe = lx
        ys, ye = ly
        plnx = np.linspace(xs, xe, nx)
        plny = np.linspace(ys, ye, ny)
        mesh = np.meshgrid(plnx, plny)
        data = np.zeros_like(mesh[0])
        for (ix, iy), x in np.ndenumerate(data):
            px, py = mesh[0][ix, iy], mesh[1][ix, iy]
            p0 = gp_Pnt(px, py, 0).Transformed(trf)
            p1 = self.proj_pnt_pln(p0, face, axs)
            z = p0.Distance(p1)
            data[ix, iy] = z
            self.display.DisplayShape(p0)
            self.display.DisplayShape(p1)
        return mesh, data

    def reflect_beam(self, shpe=TopoDS_Shape(), beam0=gp_Ax3(), tr=0):
        """
        Calculate the reflection/transmission of a beam by shape

        Args:
            shpe (TopoDS_Shape): The reflection shape. Defaults to TopoDS_Shape().
            beam0 (gp_Ax3): Defaults to gp_Ax3().
            tr (int): 
                0 : reflection (Default)
                1 : transmission
                2 : normal on face.

        Returns:
            beam1 [gp_Ax3]: 
        """
        v0 = dir_to_vec(beam0.Direction())
        v1 = dir_to_vec(beam0.XDirection())
        p0 = beam0.Location()
        d0 = beam0.Direction()
        lin = gp_Lin(p0, d0)
        api = BRepIntCurveSurface_Inter()
        api.Init(shpe, lin, 1.0E-9)
        dst = np.inf
        num = 0
        sxy = p0
        uvw = [0, 0, 0]
        fce = None
        while api.More():
            p1 = api.Pnt()
            dst1 = p0.Distance(p1)
            print(api.Transition(), p1)
            if dst1 < dst and api.W() > 1.0E-6:
                dst = dst1
                uvw = [api.U(), api.V(), api.W()]
                sxy = api.Pnt()
                fce = api.Face()
                api.Next()
            else:
                api.Next()

        print(*uvw)
        u, v, w = uvw
        surf = BRepAdaptor_Surface(fce)
        prop = BRepLProp_SLProps(surf, u, v, 2, 1.0E-9)
        p1, vx, vy = prop.Value(), prop.D1U(), prop.D1V()
        vz = vx.Crossed(vy)
        if vz.Dot(v0) > 0:
            vz.Reverse()
        vx.Normalize()
        vy.Normalize()
        vz.Normalize()
        beam1 = gp_Ax3(
            p1,
            vec_to_dir(v0.Reversed()),
            vec_to_dir(v1.Reversed())
        )
        norm1 = gp_Ax3(
            p1,
            vec_to_dir(vz),
            vec_to_dir(vx)
        )
        if tr == 0:
            # Reflection
            beam1.Mirror(norm1.Ax2())
            if beam1.Direction().Dot(norm1.Direction()) < 0:
                beam1.ZReverse()
        elif tr == 1:
            # Transmission
            beam1.ZReverse()
            beam1.XReverse()
        elif tr == 2:
            # Normal Axis
            beam1 = gp_Ax3(norm1.Ax2())
        elif tr == 3:
            # Normal Axis (ZReverse)
            beam1 = gp_Ax3(norm1.Ax2())
            beam1.ZReverse()
        return beam1

    def calc_angle(self, ax0=gp_Ax3(), ax1=gp_Ax3()):
        p0 = ax0.Location()
        v0 = dir_to_vec(ax0.Direction())
        p1 = ax1.Location()
        v1 = dir_to_vec(ax1.Direction())
        return p0.Distance(p1), v0.Dot(v1)

    def calc_tor_angle1(self, ax0=gp_Ax3(), ax1=gp_Ax3()):
        # img/tok_angle_def1.png
        print(dir_to_vec(ax1.Direction()))
        # Base plane on ax0
        pln = self.make_plane_axs(ax0, [-1, 1], [-1, 1])

        # Beam Project to Base plane
        ax1_prj = self.proj_pnt_pln(ax1.Location(), pln, ax0)
        ax1_dst = self.prop_axs(ax1, scale=1).Location()

        # Ref Coord
        # Z Direction: ax1-pnt -> ax0-Z-Ax
        # Y Direction: ax0-ZDir
        ax1_loc = gp_Ax3(ax1.Ax2())
        ax1_loc.SetDirection(vec_to_dir(gp_Vec(ax1_prj, ax0.Location())))
        ax1_loc.SetYDirection(ax0.Direction())
        v_z = dir_to_vec(ax1_loc.Direction())
        v_x = dir_to_vec(ax1_loc.XDirection())
        v_y = dir_to_vec(ax1_loc.YDirection())

        # Ref Coord XY-Plane (Poloidal)
        ax1_axy = gp_Ax3(ax1.Location(),
                         ax1_loc.Direction(),
                         ax1_loc.XDirection())
        ax1_pxy = self.make_plane_axs(ax1_axy, [-1, 1], [-1, 1])
        pnt_pxy = self.proj_pnt_pln(ax1_dst, ax1_pxy, ax1_loc)

        # Ref Coord XZ-Plane (Toroidal)
        ax1_axz = gp_Ax3(ax1.Location(),
                         ax1_loc.YDirection(),
                         ax1_loc.XDirection())
        ax1_pxz = self.make_plane_axs(ax1_axz, [-1, 1], [-1, 1])
        pnt_pxz = self.proj_pnt_pln(ax1_dst, ax1_pxz, ax1_axz)
        lin_pxz = make_edge(ax1_loc.Location(), pnt_pxz)
        vec_pxz = gp_Vec(ax1_loc.Location(), pnt_pxz)
        deg_pxz = v_z.AngleWithRef(vec_pxz, v_y.Reversed())
        txt_pxz = "pnt_pxy: {:.1f}".format(np.rad2deg(deg_pxz))
        print("Tor: ", np.rad2deg(deg_pxz))

        # Ref Coord YZ-Plane (Poloidal)
        ax1_ayz = gp_Ax3(ax1.Location(),
                         ax1_loc.XDirection(),
                         ax1_loc.Direction())
        ax1_pyz = self.make_plane_axs(ax1_ayz, [-1, 1], [-1, 1])
        pnt_pyz = self.proj_pnt_pln(ax1_dst, ax1_pyz, ax1_ayz)
        lin_pyz = make_edge(ax1_loc.Location(), pnt_pyz)
        vec_pyz = gp_Vec(ax1_loc.Location(), pnt_pyz)
        deg_pyz = np.pi / 2 - v_y.AngleWithRef(vec_pyz, v_x)
        txt_pyz = "pnt_pyz: {:.1f}".format(np.rad2deg(deg_pyz))
        print("Pol: ", np.rad2deg(deg_pyz))

        #self.show_axs_pln(ax0, scale=0.5)
        #self.show_axs_pln(ax1_loc, scale=1)
        #self.display.DisplayShape(lin_pxz, color="BLUE1")
        #self.display.DisplayShape(ax1_pxz, color="BLUE1", transparency=0.9)
        #self.display.DisplayShape(lin_pyz, color="GREEN")
        #self.display.DisplayShape(ax1_pyz, color="GREEN", transparency=0.9)
        # self.display.DisplayShape(
        #   make_edge(ax1.Location(), ax1_dst), color="YELLOW")
        # self.display.DisplayVector(dir_to_vec(
        #   ax1.Direction()).Scaled(0.5), ax1_dst)
        return deg_pxz, deg_pyz

    def calc_tor_angle2(self, ax0=gp_Ax3(), ax1=gp_Ax3()):
        # img/tok_angle_def2.png
        print(dir_to_vec(ax1.Direction()))
        # Base plane on ax0
        pln = self.make_plane_axs(ax0, [-1, 1], [-1, 1])

        # Beam Project to Base plane
        ax1_prj = self.proj_pnt_pln(ax1.Location(), pln, ax0)
        ax1_dst = self.prop_axs(ax1, scale=1).Location()
        ax1_dst_prj = self.proj_pnt_pln(ax1_dst, pln, ax0)

        # Ref Coord
        # Z Direction: ax1-dir (projected)
        # Y Direction: ax0-ZDir
        ax1_loc = gp_Ax3(ax1.Ax2())
        ax1_loc.SetDirection(vec_to_dir(gp_Vec(ax1_prj, ax1_dst_prj)))
        ax1_loc.SetYDirection(ax0.Direction())
        v_z = dir_to_vec(ax1_loc.Direction())
        v_x = dir_to_vec(ax1_loc.XDirection())
        v_y = dir_to_vec(ax1_loc.YDirection())

        # Ref Coord XY-Plane (Poloidal)
        ax1_axy = gp_Ax3(ax1.Location(),
                         ax1_loc.Direction(),
                         ax1_loc.XDirection())
        ax1_pxy = self.make_plane_axs(ax1_axy, [-1, 1], [-1, 1])
        pnt_pxy = self.proj_pnt_pln(ax1_dst, ax1_pxy, ax1_loc)

        # Ref Coord XZ-Plane (Toroidal)
        ax1_axz = gp_Ax3(ax1.Location(),
                         ax1_loc.YDirection(),
                         ax1_loc.XDirection())
        ax1_pxz = self.make_plane_axs(ax1_axz, [-1, 1], [-1, 1])
        pnt_pxz = self.proj_pnt_pln(ax1_dst, ax1_pxz, ax1_axz)
        lin_pxz = make_edge(ax1_loc.Location(), pnt_pxz)
        vec_pxz = gp_Vec(ax1_prj, ax0.Location())
        deg_pxz = v_z.AngleWithRef(vec_pxz, v_y)
        txt_pxz = "pnt_pxy: {:.1f}".format(np.rad2deg(deg_pxz))
        print("Tor: ", np.rad2deg(deg_pxz))

        # Ref Coord YZ-Plane (Poloidal)
        ax1_ayz = gp_Ax3(ax1.Location(),
                         ax1_loc.XDirection(),
                         ax1_loc.Direction())
        ax1_pyz = self.make_plane_axs(ax1_ayz, [-1, 1], [-1, 1])
        pnt_pyz = self.proj_pnt_pln(ax1_dst, ax1_pyz, ax1_ayz)
        lin_pyz = make_edge(ax1_loc.Location(), pnt_pyz)
        vec_pyz = dir_to_vec(ax1.Direction())
        deg_pyz = vec_pyz.AngleWithRef(v_z, v_x)
        txt_pyz = "pnt_pyz: {:.1f}".format(np.rad2deg(deg_pyz))
        print("Pol: ", np.rad2deg(deg_pyz))

        #self.show_axs_pln(ax0, scale=0.5)
        #self.show_axs_pln(ax1_loc, scale=1)
        #self.display.DisplayShape(lin_pxz, color="BLUE1")
        #self.display.DisplayShape(ax1_pxz, color="BLUE1", transparency=0.9)
        #self.display.DisplayShape(lin_pyz, color="GREEN")
        #self.display.DisplayShape(ax1_pyz, color="GREEN", transparency=0.9)
        # self.display.DisplayShape(
        #   make_edge(ax1.Location(), ax1_dst), color="YELLOW")
        # self.display.DisplayVector(dir_to_vec(
        #   ax1.Direction()).Scaled(0.5), ax1_dst)
        return deg_pxz, deg_pyz


if __name__ == '__main__':
    obj = dispocc()
    obj.show_axs_pln()
    obj.display.DisplayShape(make_box(100, 100, 100))
    obj.display.FitAll()
    obj.export_cap()
    obj.close()

    print(obj.base_axs)

    obj = dispocc(touch=True)
    obj.show_axs_pln()
    obj.display.DisplayShape(make_box(100, 100, 100),
                             material=Graphic3d_NOM_ALUMINIUM)
    obj.display.DisplayShape(obj.make_plane_axs(),
                             material=Graphic3d_NOM_COPPER)
    obj.ShowOCC()

    print(obj.base_axs)
    obj.reload_app(disp=True, touch=True)
    obj.ShowOCC()
