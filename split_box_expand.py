import numpy as np

from OCC.Display.SimpleGui import init_display
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir
from OCC.Core.gp import gp_Ax1, gp_Ax2, gp_Ax3
from OCC.Core.gp import gp_Pln, gp_Lin
from OCC.Core.gp import gp_Trsf
from OCC.Core.TopLoc import TopLoc_Location
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
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Shape
from OCC.Core.TopoDS import TopoDS_Edge, TopoDS_Solid, TopoDS_Face
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_SOLID, TopAbs_FACE
from OCC.Core.TopTools import TopTools_ListOfShape
from OCC.Core.GProp import GProp_GProps
from OCC.Core.GEOMAlgo import GEOMAlgo_Splitter
from OCC.Extend.DataExchange import write_step_file, write_stl_file
from OCC.Extend.ShapeFactory import make_box, make_face
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCCUtils.Construct import vec_to_dir


class CovExp (object):

    def __init__(self, file=False, show=False):
        self.prop = GProp_GProps()
        self.base = make_box(1000, 1000, 1000)
        self.base_vol = self.cal_vol(self.base)

        self.splitter = GEOMAlgo_Splitter()
        self.splitter.AddArgument(self.base)
        print(self.cal_vol(self.base))

        self.show = show

        if self.show == True:
            self.display, self.start_display, self.add_menu, self.add_function_to_menu = init_display()

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
            pnt = gp_Pnt(*np.random.rand(3) * 1000)
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
        print(edge_line, edge_line.Position())
        i_min = edge_adaptor.FirstParameter()
        i_max = edge_adaptor.LastParameter()
        #print(i_min, edge_adaptor.Value(i_min))
        #print(i_max, edge_adaptor.Value(i_max))

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
            edge = find_edge.EdgeFrom()
            self.prop_edge(edge)
            plan = self.pln_on_face(face)
            print(face, self.cal_are(face), plan)
            print(plan, plan.Axis())
            pln_angle = self.tmp_axis.Angle(plan.Axis())
            print(np.rad2deg(pln_angle))
            #print(self.cal_len(edge), self.cal_are(face))

            self.face_tranfer(face, plan.Axis())
            if self.show == True:
                self.display.DisplayShape(face, transparency=0.5)

            plan = self.pln_on_face(face)
            print(face, self.cal_are(face), plan)
            print(plan, plan.Axis())
            find_edge.Next()

    def face_tranfer(self, face=TopoDS_Face(), axs=gp_Ax1()):
        axs_3 = gp_Ax3(axs.Location(), axs.Direction())
        trf = gp_Trsf()
        trf.SetTransformation(axs_3, self.tmp_axs3)
        loc_face = TopLoc_Location(trf)
        face.Move(loc_face)
        return face

    def face_init(self, face=TopoDS_Face()):
        self.tmp_face = face
        self.tmp_plan = self.pln_on_face(self.tmp_face)
        self.tmp_axis = self.tmp_plan.Axis()
        self.tmp_axs3 = gp_Ax3(self.tmp_axis.Location(),
                               self.tmp_axis.Direction())
        print(self.tmp_plan.Axis())

    def prop_soild(self, sol=TopoDS_Solid()):
        sol_exp = TopExp_Explorer(sol, TopAbs_FACE)
        sol_top = TopologyExplorer(sol)
        #print(self.cal_vol(sol), self.base_vol)
        print(sol_top.number_of_faces())

        self.face_init(sol_exp.Current())
        sol_exp.Next()

        while sol_exp.More():
            self.face_expand(sol_exp.Current())
            sol_exp.Next()

    def prop_solids(self):
        sol_exp = TopExp_Explorer(self.splitter.Shape(), TopAbs_SOLID)
        while sol_exp.More():
            self.prop_soild(sol_exp.Current())
            sol_exp.Next()


if __name__ == "__main__":
    obj = CovExp(show=True)
    obj.split_run(1)
    obj.prop_solids()

    # print(obj.cal_vol())
    # obj.prop_soild(obj.base)

    # obj.fileout()
    obj.ShowDisplay()
