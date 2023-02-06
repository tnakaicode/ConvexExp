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
from OCC.Core.BRepFilletAPI import BRepFilletAPI_MakeFillet
from OCC.Core.BRepCheck import BRepCheck_Analyzer
# from OCC.Core.BRepAlgo import BRepAlgo_BooleanOperation
from OCC.Core.BOPAlgo import BOPAlgo_Splitter
from OCC.Core.LocOpe import LocOpe_FindEdges
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Shape, TopoDS_Iterator
from OCC.Core.TopoDS import TopoDS_Edge, TopoDS_Solid, TopoDS_Face, topods, topods_Vertex
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_SOLID, TopAbs_FACE, TopAbs_VERTEX
from OCC.Core.TopTools import TopTools_ListOfShape
from OCC.Core.GProp import GProp_GProps
# from OCC.Core.GEOMAlgo import GEOMAlgo_Splitter
from OCC.Core.Prs3d import Prs3d_DimensionAspect
from OCC.Core.PrsDim import PrsDim_AngleDimension
from OCC.Core.Quantity import Quantity_Color, Quantity_NOC_RED1, Quantity_NOC_BLACK
from OCC.Extend.DataExchange import write_step_file, write_stl_file
from OCC.Extend.ShapeFactory import make_face, make_edge
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCCUtils.Construct import make_box
from OCCUtils.Construct import vec_to_dir, dir_to_vec, vertex2pnt

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

    def __init__(self, temp=True, disp=True, touch=False):
        dispocc.__init__(self, temp, disp, touch)
        self.prop = GProp_GProps()
        self.base = make_box(100, 100, 100)
        self.base_vol = self.cal_vol(self.base)

        self.splitter = BOPAlgo_Splitter()
        self.splitter.AddArgument(self.base)
        print(self.cal_vol(self.base))
        
        self.context = []

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

    def split_run(self, num=5, seed=11):
        if seed != None:
            np.random.seed(seed)
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

    def face_midpoint(self, face=TopoDS_Face()):
        pts = [vertex2pnt(v) for v in TopologyExplorer(face).vertices()]
        x, y, z = 0, 0, 0
        for p in pts:
            x += p.X()
            y += p.Y()
            z += p.Z()
        return gp_Pnt(x / len(pts), y / len(pts), z / len(pts))

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
        # face_dir = face_adaptor.Direction()

        face_umin = face_adaptor.FirstUParameter()
        face_vmin = face_adaptor.FirstVParameter()
        face_umax = face_adaptor.LastUParameter()
        face_vmax = face_adaptor.LastVParameter()
        face_u = (face_umax + face_umin) / 2
        face_v = (face_vmax + face_vmin) / 2
        face_pnt = face_adaptor.Value(face_u, face_v)
        face_pnt = self.face_midpoint(face)
        face_pln.SetLocation(face_pnt)
        return face_pln

    def face_fillet(self, face=TopoDS_Face()):
        plan = self.pln_on_face(face)
        find_edge = LocOpe_FindEdges(self.fix_face, face)
        find_edge.InitIterator()
        edge_n = 0

        edge = find_edge.EdgeTo()
        self.display.DisplayShape(edge, color="BLUE1")
        self.fill.Add(10, edge)
        self.fill.Build()
        self.display.DisplayShape(self.fill.Shape(), transparency=0.8)
        self.export_stp(self.fill.Shape(), self.tempname + "_fillet.stp")

    def face_expand(self, face=TopoDS_Face()):
        """Scan around fix_face as a reference and TopDS_Edge in common with the TopoDS_Face that constitutes TopDS_Solid.
           If there is a common TopDS_Edge, the face is rotated (self.rotate_face) around the TopDS_Edge.

        Args:
            face (_type_, optional): _description_. Defaults to TopoDS_Face().
        """
        plan = self.pln_on_face(face)  # gp_Pln
        plan_axs = plan.Position()  # gp_Ax3
        find_edge = LocOpe_FindEdges(self.fix_face, face)
        find_edge.InitIterator()
        edge_n = 0
        while find_edge.More():
            i = (edge_n + self.fix_face_n) % len(self.colors)

            # Common TopoDS_Edge of face and fix_face
            edge = find_edge.EdgeTo()  # TopoDS_Edge
            line = self.prop_edge(edge)  # gp_Lin

            e_curve, u0, u1 = BRep_Tool.Curve(edge)
            vz = gp_Vec(0, 0, 1)  # tangent of edge
            p0 = gp_Pnt(0, 0, 1)  # midpoint of edge
            e_curve.D1((u0 + u1) / 2, p0, vz)
            e_vec = gp_Vec(e_curve.Value(u0), e_curve.Value(u1)).Normalized()
            txt = f"Face{self.fix_face_n}-Edge{edge_n}"
            self.display.DisplayShape(edge, color=self.colors[i])
            self.context.append(self.display.DisplayMessage(p0, txt))

            # Axis defined by common edge
            line_axs = line.Position()  # gp_Ax1
            line_axs.SetLocation(p0)

            pz = dir_to_vec(plan_axs.Direction())
            px = gp_Vec(p0, plan_axs.Location()).Normalized()
            py = gp_Vec(p0, self.fix_axis.Location()).Normalized()
            vx = pz.Crossed(vz)
            if px.Dot(vx) < 0:
                vx.Reverse()
            line_axs = gp_Ax3(p0,
                              vec_to_dir(vz),
                              vec_to_dir(vx))
            line_flg = py.Dot(dir_to_vec(line_axs.YDirection()))

            print()
            print(f"Face: {self.fix_face_n}, Edge: {edge_n}")
            print(edge, self.cal_len(edge))
            print(face, self.cal_are(face), plan)
            print("fix face", self.fix_axis.Axis())
            print("tmp face", plan_axs.Axis())
            print("Dir", py.Dot(dir_to_vec(line_axs.YDirection())))

            self.face_rotate(face, line_axs, flg=line_flg)
            # self.face_tranfer(face, plan.Axis())

            find_edge.Next()
            edge_n += 1

    def face_rotate(self, face=TopoDS_Face(), axs=gp_Ax3(), flg=1):
        """face rotate

        Args:
            face (_type_, optional): TopDS_Face with TopDS_Edge in common with the TopDS_Face (self.tmp_face) to be used as a reference
            axs (_type_, optional): gp_Ax1 defining the common TopDS_Edge of the "face" and the TopDS_Face (self.tmp_face) to be referenced

        Returns:
            _type_: _description_
        """
        # Axis of rotated face
        plan = self.pln_on_face(face)
        plan_axs = plan.Position()

        rim_axs = axs.Ax2()
        rim_circl = Geom_Circle(rim_axs, 10)
        rim_u0, rim_u1 = rim_circl.FirstParameter(), rim_circl.LastParameter()
        rim_p0 = rim_circl.Value(rim_u0)

        pln_angle = self.fix_axis.Direction().Angle(plan_axs.Direction())
        #pln_angle = self.fix_axis.Direction().AngleWithRef(plan_axs.Direction())
        print("Angle", np.rad2deg(pln_angle))

        rim_u2 = -pln_angle
        rim_p2 = rim_circl.Value(rim_u2)
        rim_angle = Geom_TrimmedCurve(rim_circl, rim_u0, rim_u2)

        trf = gp_Trsf()
        if flg >= 0:
            ang = rim_u2
        else:
            ang = rim_u2 - np.pi
        trf.SetRotation(axs.Axis(), ang)
        loc_face = TopLoc_Location(trf)
        new_face = face.Moved(loc_face)
        self.display.DisplayShape(new_face, transparency=0.5)
        # self.display.DisplayShape(rim_angle)
        self.display.DisplayShape(plan_axs.Location())
        self.context.append(self.display.DisplayVector(dir_to_vec(plan_axs.Direction()).Scaled(5), plan_axs.Location()))
        #self.display.DisplayShape(rim_p0)
        #self.display.DisplayShape(rim_p2)
        # self.show_axs_pln(axs, scale=5)
        # self.display.DisplayMessage(rim_p0,
        #                            f"rim_p0: {np.rad2deg(rim_u0):.1f}")
        # self.display.DisplayMessage(rim_p2,
        #                            f"rim_p2: {np.rad2deg(rim_u2):.1f}")

        ag_aspect = Prs3d_DimensionAspect()
        ag_aspect.SetCommonColor(Quantity_Color(Quantity_NOC_BLACK))
        ag = PrsDim_AngleDimension(rim_p0,
                                   axs.Location(),
                                   rim_p2)
        ag.SetDimensionAspect(ag_aspect)
        self.context.append (self.display.Context.Display(ag, True))

        return new_face

    def face_init(self, face=TopoDS_Face()):
        self.fix_face = face
        self.fix_plan = self.pln_on_face(self.fix_face)
        self.fix_axis = self.fix_plan.Position()
        self.fix_face_n = 0
        # self.show_axs_pln(self.fix_axis, scale=20, name="Fix-Face")
        self.context.append(self.display.DisplayVector(dir_to_vec(self.fix_axis.Direction()).Scaled(5), self.fix_axis.Location()))
        self.display.DisplayShape(self.fix_face, color="RED")
        self.display.DisplayShape(self.fix_axis.Location())

    def prop_fillet(self, sol=TopoDS_Solid()):
        self.fill = BRepFilletAPI_MakeFillet(sol)
        fce_exp = TopExp_Explorer(sol, TopAbs_FACE)
        sol_top = TopologyExplorer(sol)
        print()
        print(sol, self.cal_vol(sol))
        print(sol_top.number_of_faces())

        self.face_init(fce_exp.Current())
        fce_exp.Next()
        fce_exp.Next()
        fce_exp.Next()

        face = fce_exp.Current()
        self.face_fillet(face)
        fce_exp.Next()
        self.fix_face_n += 1

    def prop_soild(self, sol=TopoDS_Solid(), nfce=0):
        """property of Topo_DS_Solid
           Determine one TopoDS_Face as the basis for deployment.

        Args:
            sol (TopoDS_Solid()): Defaults to TopoDS_Solid().
        """
        fce_exp = TopExp_Explorer(sol, TopAbs_FACE)
        sol_top = TopologyExplorer(sol)
        print()
        print(sol, self.cal_vol(sol))
        print(sol_top.number_of_faces())
        
        if nfce > fce_exp.Depth():
            nfce = fce_exp.Depth()
        for _ in range(nfce):
            fce_exp.Next()
        self.face_init(fce_exp.Current())

        fce_exp = TopExp_Explorer(sol, TopAbs_FACE)
        while fce_exp.More():
            face = fce_exp.Current()
            if self.fix_face.IsEqual(face):
                pass
            else:
                self.face_expand(face)
            fce_exp.Next()
            self.fix_face_n += 1

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
