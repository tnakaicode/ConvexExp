import numpy as np
import sys
import sys
import os
import time

sys.path.append(os.path.join("../"))
from src.base import plot2d, plotocc

from OCC.Display.SimpleGui import init_display
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir
from OCC.Core.gp import gp_Ax1, gp_Ax2, gp_Ax3
from OCC.Core.gp import gp_Pln, gp_Lin
from OCC.Core.gp import gp_Trsf
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.BRepLProp import BRepLProp_CLProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepAlgo import BRepAlgo_BooleanOperation
from OCC.Core.BOPAlgo import BOPAlgo_MakerVolume, BOPAlgo_Builder
from OCC.Core.LocOpe import LocOpe_FindEdges
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Builder
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_CompSolid
from OCC.Core.TopoDS import TopoDS_Edge, TopoDS_Solid, TopoDS_Face
from OCC.Core.TopoDS import TopoDS_Iterator, topods_Vertex
from OCC.Core.TopAbs import TopAbs_VERTEX
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_SOLID, TopAbs_FACE
from OCC.Core.TopTools import TopTools_ListOfShape
from OCC.Core.GProp import GProp_GProps
#from OCC.Core.GEOMAlgo import GEOMAlgo_Splitter
from OCC.Core.BOPAlgo import BOPAlgo_Splitter
from OCC.Extend.DataExchange import write_step_file, write_stl_file
from OCC.Extend.ShapeFactory import make_face, make_edge
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Extend.TopologyUtils import dump_topology_to_string, get_type_as_string
from OCCUtils.Construct import vec_to_dir, dir_to_vec
from OCCUtils.Construct import make_box

from PyQt5.QtWidgets import QApplication, qApp
from PyQt5.QtWidgets import QDialog, QCheckBox


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


class CovExp (plotocc):

    def __init__(self):
        plotocc.__init__(self)
        self.prop = GProp_GProps()
        self.base = make_box(100, 100, 100)
        self.base_vol = self.cal_vol(self.base)

        self.splitter = BOPAlgo_Splitter()
        self.splitter.AddArgument(self.base)
        print(self.cal_vol(self.base))

        from OCC.Display.qtDisplay import qtViewer3d
        self.app = self.get_app()
        self.wi = self.app.topLevelWidgets()[0]
        self.vi = self.wi.findChild(qtViewer3d, "qt_viewer_3d")
        self.on_select()

    def get_app(self):
        app = QApplication.instance()
        #app = qApp
        # checks if QApplication already exists
        if not app:
            app = QApplication(sys.argv)
        return app

    def on_select(self):
        self.vi.sig_topods_selected.connect(self._on_select)

    def _on_select(self, shapes):
        """
        Parameters
        ----------
        shape : TopoDS_Shape
        """
        for shape in shapes:
            self.DumpTop(shape)

    def DumpTop(self, shape, level=0):
        """
        Print the details of an object from the top down
        """
        brt = BRep_Tool()
        s = shape.ShapeType()
        if s == TopAbs_VERTEX:
            pnt = brt.Pnt(topods_Vertex(shape))
            dmp = " " * level
            dmp += "%s - " % get_type_as_string(shape)
            dmp += "%.5e %.5e %.5e" % (pnt.X(), pnt.Y(), pnt.Z())
            print(dmp)
        else:
            dmp = " " * level
            dmp += get_type_as_string(shape)
            print(dmp)
        it = TopoDS_Iterator(shape)
        while it.More():
            shp = it.Value()
            it.Next()
            self.DumpTop(shp, level + 1)

    def ShowDisplay(self):
        colors = ["BLUE", "RED", "GREEN", "YELLOW", "BLACK", "WHITE"]

        num = 0
        sol_exp = TopExp_Explorer(self.splitter.Shape(), TopAbs_SOLID)
        while sol_exp.More():
            num += 1
            self.display.DisplayShape(
                sol_exp.Current(), color=colors[num % len(colors)], transparency=0.5)
            sol_exp.Next()

        self.display.FitAll()
        self.start_display()

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
        brepgprop.LinearProperties(shp, self.prop)
        return self.prop.Mass()

    def cal_are(self, shp=TopoDS_Shape()):
        brepgprop.SurfaceProperties(shp, self.prop)
        return self.prop.Mass()

    def cal_vol(self, shp=TopoDS_Shape()):
        brepgprop.VolumeProperties(shp, self.prop)
        return self.prop.Mass()

    def prop_edge(self, edge=TopoDS_Edge()):
        edge_adaptor = BRepAdaptor_Curve(edge)
        edge_line = edge_adaptor.Line()
        return edge_line

    def face_init(self, face=TopoDS_Face()):
        self.face_num += 1
        self.tmp_face = face
        self.tmp_plan = self.pln_on_face(self.tmp_face)
        self.tmp_axis = self.tmp_plan.Position()
        print(self.tmp_axis)

        if self.show == True:
            self.display.DisplayMessage(
                self.tmp_axis.Location(), "{:04d}".format(self.face_num))
            pass

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

        return face_pln

    def face_expand(self, face=TopoDS_Face()):
        print(face)

        find_edge = LocOpe_FindEdges(self.tmp_face, face)
        find_edge.InitIterator()

        while find_edge.More():
            if face not in self.face_cnt:
                edge = find_edge.EdgeTo()
                line = self.prop_edge(edge)
                plan = self.pln_on_face(face)

                plan_axs = plan.Position()
                line_axs = line.Position()

                print(self.tmp_axis.Axis())
                print(plan.Position().Axis())
                #print(self.cal_len(edge), self.cal_are(face))

                new_face = self.face_rotate(face, line_axs)
                #self.face_tranfer(face, plan.Axis())

                plan = self.pln_on_face(face)
                print(face, self.cal_are(face), plan)
                print(plan, plan.Axis())
                self.face_cnt.append(face)
            find_edge.Next()
        # self.face_init(face)

    def face_tranfer(self, face=TopoDS_Face(), axs=gp_Ax1()):
        axs_3 = gp_Ax3(axs.Location(), axs.Direction())
        trf = gp_Trsf()
        trf.SetTransformation(axs_3, self.tmp_axs3)
        loc_face = TopLoc_Location(trf)
        face.Location(loc_face)
        return face

    def face_rotate(self, face=TopoDS_Face(), axs=gp_Ax1()):
        plan = self.pln_on_face(face)
        plan_axs = plan.Position()

        pln_angle = self.tmp_axis.Angle(plan_axs)
        ref_angle = self.tmp_axis.Direction().AngleWithRef(
            plan_axs.Direction(), axs.Direction())
        print(np.rad2deg(pln_angle), np.rad2deg(ref_angle))

        trf = gp_Trsf()
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
        new_face = face.Located(loc_face)
        # self.sol_builder.Add(new_face)
        self.face_lst.Append(new_face)
        # face.Location(loc_face)
        if self.show == True:
            self.display.DisplayShape(new_face)
        return new_face

    def prop_soild(self, sol=TopoDS_Solid()):
        self.sol_builder = TopoDS_Builder()

        sol_exp = TopExp_Explorer(sol, TopAbs_FACE)
        sol_top = TopologyExplorer(sol)
        #print(self.cal_vol(sol), self.base_vol)
        print(sol_top.number_of_faces())

        self.face_lst = TopTools_ListOfShape()
        self.face_cnt = []
        self.face_num = 0
        self.face_init(sol_exp.Current())
        #self.sol_builder.Add(sol, sol_exp.Current())
        sol_exp.Next()

        while sol_exp.More():
            face = sol_exp.Current()
            self.face_expand(face)
            sol_exp.Next()

        if self.file == True:
            stp_file = "./shp/shp_{:04d}.stp".format(self.sol_num)
            write_step_file(sol, stp_file)

            stp_file = "./shp/shp_{:04d}_exp.stp".format(self.sol_num)
            new_shpe = TopoDS_Compound()
            self.sol_builder.Add(gp_Pnt())
            # self.sol_builder.MakeCompSolid(new_shpe)
            #write_step_file(new_shpe, stp_file)

        # if self.show == True:
        #    self.display.DisplayShape(self.face_cnt)
        """self.face_init(face)
        sol_exp = TopExp_Explorer(sol, TopAbs_FACE)
        while sol_exp.More():
            face = sol_exp.Current()
            self.face_expand(face)
            #self.face_init(sol_exp.Current())
            sol_exp.Next()"""

    def prop_solids(self):
        sol_exp = TopExp_Explorer(self.splitter.Shape(), TopAbs_SOLID)
        self.sol_num = 0
        while sol_exp.More():
            self.sol_num += 1
            self.prop_soild(sol_exp.Current())
            sol_exp.Next()


if __name__ == "__main__":
    obj = CovExp()
    obj.show_axs_pln()
    obj.ShowOCC()
