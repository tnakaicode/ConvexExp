import numpy as np
import sys
import os

from OCC.Display.SimpleGui import init_display
from OCC.Core.gp import gp_Circ, gp_Pnt, gp_Vec, gp_Dir
from OCC.Core.gp import gp_Ax1, gp_Ax2, gp_Ax3
from OCC.Core.gp import gp_Pln, gp_Lin
from OCC.Core.gp import gp_Trsf
from OCC.Core.Geom import Geom_Circle, Geom_TrimmedCurve
from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.BRepLProp import BRepLProp_CLProps
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
from OCC.Core.BRepGProp import brepgprop_VolumeProperties
from OCC.Core.BRepGProp import brepgprop_LinearProperties
from OCC.Core.BRepCheck import BRepCheck_Analyzer
#from OCC.Core.BRepAlgo import BRepAlgo_BooleanOperation
from OCC.Core.BOPAlgo import BOPAlgo_Splitter
from OCC.Core.LocOpe import LocOpe_FindEdges
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Shape, TopoDS_Iterator
from OCC.Core.TopoDS import TopoDS_Edge, TopoDS_Solid, TopoDS_Face, topods, topods_Vertex
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_SOLID, TopAbs_FACE, TopAbs_VERTEX
from OCC.Core.TopTools import TopTools_ListOfShape
from OCC.Core.GProp import GProp_GProps
#from OCC.Core.GEOMAlgo import GEOMAlgo_Splitter
from OCC.Extend.DataExchange import write_step_file, write_stl_file
from OCC.Extend.ShapeFactory import make_face, make_edge
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCCUtils.Construct import make_box
from OCCUtils.Topology import shapeTypeString, dumpTopology
from OCCUtils.Construct import vec_to_dir, dir_to_vec

from PyQt5.QtWidgets import QApplication, qApp
from PyQt5.QtWidgets import QDialog, QCheckBox

sys.path.append(os.path.join("../"))
from src.base_occ import dispocc


def axs1_to_axs3(axs=gp_Ax1()):
    return gp_Ax3(axs.Location(), axs.Direction())


def axs_pln(axs):
    pnt = axs.Location()
    vx = dir_to_vec(axs.XDirection()).Scaled(100)
    vy = dir_to_vec(axs.YDirection()).Scaled(200)
    vz = dir_to_vec(axs.Direction()).Scaled(300)
    lx = make_edge(pnt, gp_Pnt((gp_Vec(pnt.XYZ()) + vx).XYZ()))
    ly = make_edge(pnt, gp_Pnt((gp_Vec(pnt.XYZ()) + vy).XYZ()))
    lz = make_edge(pnt, gp_Pnt((gp_Vec(pnt.XYZ()) + vz).XYZ()))
    return lx, ly, lz


def get_axs_deg(ax0=gp_Ax3(), ax1=gp_Ax3(), ref=gp_Dir()):
    org_angle = ax0.Angle(ax1)
    ref_angle = ax0.Direction().AngleWithRef(ax1.Direction(), ax0.XDirection())

    if np.sign(org_angle) == np.sign(ref_angle):
        return org_angle
    else:
        return np.pi - ref_angle


class CovExp (dispocc):

    def __init__(self, touch=False, file=False):
        dispocc.__init__(self, touch=touch)
        self.prop = GProp_GProps()
        self.base = make_box(100, 100, 100)
        self.base_vol = self.cal_vol(self.base)

        self.splitter = BOPAlgo_Splitter()
        self.splitter.AddArgument(self.base)
        print(self.cal_vol(self.base))

    def init_base(self, shape):
        self.base = shape
        self.base_vol = self.cal_vol(self.base)

        self.splitter = BOPAlgo_Splitter()
        self.splitter.AddArgument(self.base)
        print(self.cal_vol(self.base))

    def fileout(self, dirname="./shp/"):
        num = 0
        stp_file = dirname + "shp_{:04d}.stp".format(num)
        write_step_file(self.base, stp_file)

        sol_exp = TopExp_Explorer(self.splitter.Shape(), TopAbs_SOLID)
        while sol_exp.More():
            num += 1
            stp_file = dirname + "shp_{:04d}.stp".format(num)
            write_step_file(sol_exp.Current(), stp_file)
            sol_exp.Next()

    def split_run(self, num=5):
        for i in range(num):
            pnt = gp_Pnt(*np.random.rand(3) * 100)
            vec = gp_Vec(*np.random.randn(3))
            pln = gp_Pln(pnt, vec_to_dir(vec))
            fce = make_face(pln, -10000, 10000, -10000, 10000)
            self.splitter.AddTool(fce)
        self.splitter.Perform()

    def cal_len(self, shp=TopoDS_Shape()):
        brepgprop_LinearProperties(shp, self.prop)
        return self.prop.Mass()

    def cal_are(self, shp=TopoDS_Shape()):
        brepgprop_SurfaceProperties(shp, self.prop)
        return self.prop.Mass()

    def cal_vol(self, shp=TopoDS_Shape()):
        brepgprop_VolumeProperties(shp, self.prop)
        return self.prop.Mass()

    def prop_edge(self, edge=TopoDS_Edge()):
        edge_adaptor = BRepAdaptor_Curve(edge)
        edge_line = edge_adaptor.Line()
        return edge_line

    def face_tranfer(self, face=TopoDS_Face(), axs=gp_Ax1()):
        axs_3 = gp_Ax3(axs.Location(), axs.Direction())
        trf = gp_Trsf()
        trf.SetTransformation(axs_3, self.tmp_axs3)
        loc_face = TopLoc_Location(trf)
        face.Location(loc_face)
        return face

    def pln_on_face(self, face=TopoDS_Face()):
        face_adaptor = BRepAdaptor_Surface(face)
        face_trf = face_adaptor.Trsf()
        face_pln = face_adaptor.Plane()
        #face_dir = face_adaptor.Direction()

        face_umin = face_adaptor.FirstUParameter()
        face_vmin = face_adaptor.FirstVParameter()
        face_umax = face_adaptor.LastUParameter()
        face_vmax = face_adaptor.LastVParameter()
        face_u = (face_umax + face_umin) / 2
        face_v = (face_vmax + face_vmin) / 2
        face_pnt = face_adaptor.Value(face_u, face_v)
        face_pln.SetLocation(face_pnt)
        return face_pln

    def face_expand(self, face=TopoDS_Face()):
        plan = self.pln_on_face(face)
        find_edge = LocOpe_FindEdges(self.tmp_face, face)
        find_edge.InitIterator()
        edge_n = 0
        while find_edge.More():
            edge = find_edge.EdgeTo()
            line = self.prop_edge(edge)
            
            e_curve, u0, u1 = BRep_Tool.Curve(edge)
            p = e_curve.Value((u0 + u1) / 2)
            i = (edge_n + self.tmp_face_n) % len(self.colors)
            v = gp_Vec(e_curve.Value(u0), e_curve.Value(u1))
            v.Normalize()
            vz = gp_Vec(0,0,1)
            p0 = gp_Pnt(0,0,1)
            e_curve.D1((u0+u1)/2, p0, vz)
            vx = gp_Vec(self.tmp_axis.Location(), p)
            vy = gp_Vec(self.tmp_axis.Direction())
            vx.Normalize()
            vy.Normalize()
            vz = vx.Crossed(vy)
            txt = f"Face{self.tmp_face_n}-Edge{edge_n}"
            self.display.DisplayShape(edge, color=self.colors[i])
            self.display.DisplayMessage(p, txt)
            self.display.DisplayVector(vz.Scaled(10), p)

            plan_axs = plan.Position()
            line_axs = line.Position()
            line_axs.SetLocation(p)
            line_vec = gp_Vec(line_axs.Direction())
            self.display.DisplayVector(line_vec.Scaled(5), p)

            print()
            print("Face: {:d}, Edge: {:d}".format(self.tmp_face_n, edge_n))
            print(self.tmp_axis.Axis())
            print(plan.Position().Axis())
            #print(self.cal_len(edge), self.cal_are(face))

            self.face_rotate(face, line_axs)
            #self.face_tranfer(face, plan.Axis())

            plan = self.pln_on_face(face)
            print(face, self.cal_are(face), plan)
            print(plan, plan.Axis())
            find_edge.Next()

            edge_n += 1

    def face_rotate(self, face=TopoDS_Face(), axs=gp_Ax1()):
        plan = self.pln_on_face(face)
        plan_axs = plan.Position()
        self.display.DisplayShape(plan_axs.Location())

        v0 = dir_to_vec(self.tmp_axis.Direction())
        v1 = dir_to_vec(plan_axs.Direction())
        print(v0.Dot(v1))

        lin_vec = gp_Vec(axs.Location(), plan_axs.Location())
        edg_circl = Geom_Circle(gp_Circ(
            gp_Ax2(axs.Location(),
                   axs.Direction(),
                   vec_to_dir(lin_vec)), 5))
        rim_u0, rim_u1 = edg_circl.FirstParameter(), edg_circl.LastParameter()
        rim_p0 = edg_circl.Value(rim_u0)

        pln_angle = self.tmp_axis.Angle(plan_axs)
        ref_angle = self.tmp_axis.Direction().AngleWithRef(
            plan_axs.Direction(), axs.Direction())
        print(np.rad2deg(pln_angle), np.rad2deg(ref_angle))

        rim_u2 = -ref_angle
        rim_p2 = edg_circl.Value(rim_u2)
        rim_angle = Geom_TrimmedCurve(edg_circl, rim_u0, rim_u2)

        trf = gp_Trsf()
        #trf.SetRotation(axs, 2*np.pi - ref_angle)
        if np.abs(ref_angle) >= np.pi / 2:
            trf.SetRotation(axs, -ref_angle)
        elif 0 < ref_angle < np.pi / 2:
            trf.SetRotation(axs, np.pi - ref_angle)
        elif -np.pi / 2 < ref_angle < 0:
            trf.SetRotation(axs, -ref_angle - np.pi)
        else:
            trf.SetRotation(axs, -ref_angle)
        #trf.SetTransformation(axs3.Rotated(axs, angle), axs3)
        loc_face = TopLoc_Location(trf)
        new_face = face.Moved(loc_face)
        self.display.DisplayShape(new_face, transparency=0.5)
        self.display.DisplayShape(rim_angle)
        self.display.DisplayShape(rim_p0)
        self.display.DisplayShape(rim_p2)
        return new_face

    def face_init(self, face=TopoDS_Face()):
        self.tmp_face = face
        self.tmp_plan = self.pln_on_face(self.tmp_face)
        self.tmp_axis = self.tmp_plan.Position()
        self.tmp_face_n = 0
        self.show_axs_pln(self.tmp_axis, scale=20, name="Fix-Face")
        self.display.DisplayShape(self.tmp_face, color="RED")

    def prop_soild(self, sol=TopoDS_Solid()):
        sol_exp = TopExp_Explorer(sol, TopAbs_FACE)
        sol_top = TopologyExplorer(sol)
        print()
        print(sol, self.cal_vol(sol))
        print(sol_top.number_of_faces())

        self.face_init(sol_exp.Current())
        sol_exp.Next()
        while sol_exp.More():
            face = sol_exp.Current()
            self.face_expand(face)
            sol_exp.Next()
            self.tmp_face_n += 1

        """self.face_init(face)
        sol_exp = TopExp_Explorer(sol, TopAbs_FACE)
        while sol_exp.More():
            face = sol_exp.Current()
            self.face_expand(face)
            #self.face_init(sol_exp.Current())
            sol_exp.Next()"""

    def prop_solids(self):
        sol_exp = TopExp_Explorer(self.splitter.Shape(), TopAbs_SOLID)
        while sol_exp.More():
            self.prop_soild(sol_exp.Current())
            sol_exp.Next()

    def show_split_solid(self):
        colors = ["BLUE", "RED", "GREEN", "YELLOW", "BLACK", "WHITE"]
        num = 0
        sol_exp = TopExp_Explorer(self.splitter.Shape(), TopAbs_SOLID)
        while sol_exp.More():
            num += 1
            self.display.DisplayShape(
                sol_exp.Current(), color=colors[num % len(colors)], transparency=0.5)
            sol_exp.Next()
        self.ShowOCC()


if __name__ == "__main__":
    obj = CovExp(touch=False)
    obj.split_run(4)
    # obj.prop_solids()

    sol_exp = TopExp_Explorer(obj.splitter.Shape(), TopAbs_SOLID)
    print(sol_exp.Depth())
    sol_exp.Next()
    sol_exp.Next()
    sol_exp.Next()
    obj.prop_soild(sol_exp.Current())

    obj.display.DisplayShape(obj.splitter.Shape(),
                             color="BLUE", transparency=0.9)
    obj.display.DisplayShape(sol_exp.Current(), transparency=0.5)
    obj.ShowOCC()

    # print(obj.cal_vol())
    # obj.prop_soild(obj.base)

    # sobj.fileout()
    # obj.ShowDisplay()
