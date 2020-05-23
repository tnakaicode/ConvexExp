import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import sys
import os
import json
import logging
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)

from OCC.Display.SimpleGui import init_display
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir
from OCC.Core.gp import gp_Ax1, gp_Ax2, gp_Ax3
from OCC.Core.gp import gp_Pln, gp_Lin
from OCC.Core.gp import gp_Trsf
from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.BRepLProp import BRepLProp_CLProps
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
from OCC.Core.BRepGProp import brepgprop_VolumeProperties
from OCC.Core.BRepGProp import brepgprop_LinearProperties
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepAlgo import BRepAlgo_BooleanOperation
from OCC.Core.BOPAlgo import BOPAlgo_MakerVolume, BOPAlgo_Builder
from OCC.Core.LocOpe import LocOpe_FindEdges
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Builder
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_CompSolid
from OCC.Core.TopoDS import TopoDS_Edge, TopoDS_Solid, TopoDS_Face
from OCC.Core.TopoDS import TopoDS_Iterator, topods_Vertex
from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE
from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_COMPOUND
from OCC.Core.TopTools import TopTools_ListOfShape
from OCC.Core.GProp import GProp_GProps
#from OCC.Core.GEOMAlgo import GEOMAlgo_Splitter
from OCC.Extend.DataExchange import write_step_file, write_stl_file
from OCC.Extend.DataExchange import read_step_file
from OCC.Extend.ShapeFactory import make_face, make_edge
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Extend.TopologyUtils import dump_topology_to_string, get_type_as_string
from OCCUtils.Construct import point_to_vector, vector_to_point
from OCCUtils.Construct import dir_to_vec, vec_to_dir
from OCCUtils.Construct import make_box

from PyQt5.QtWidgets import QApplication, qApp
from PyQt5.QtWidgets import QDialog, QCheckBox

from base import plotocc


class CovExp (plotocc):

    def __init__(self, stpfile="./shp/Box.stp", file=False, show=False):
        plotocc.__init__(self)
        self.SaveMenu()
        self.prop = GProp_GProps()
        self.base = read_step_file(stpfile)
        self.base_vol = self.cal_vol(self.base)

    def ShowDisplay(self):
        self.display.DisplayShape(self.base, color="BLUE", transparency=0.5)
        self.show()

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

        # if self.file == True:
        #    stp_file = "./shp/shp_{:04d}.stp".format(self.sol_num)
        #    write_step_file(sol, stp_file)
        #
        #    stp_file = "./shp/shp_{:04d}_exp.stp".format(self.sol_num)
        #    new_shpe = TopoDS_Compound()
        #    self.sol_builder.MakeCompSolid(new_shpe)
        #    write_step_file(new_shpe, stp_file)

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
    print(obj.cal_vol())
    obj.prop_soild(obj.base)
    obj.ShowDisplay()
