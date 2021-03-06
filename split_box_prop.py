import numpy as np

from OCC.Display.SimpleGui import init_display
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir
from OCC.Core.gp import gp_Ax3
from OCC.Core.gp import gp_Pln, gp_Lin
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.BRepLProp import BRepLProp_CLProps
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties, brepgprop_VolumeProperties, brepgprop_LinearProperties
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepAlgo import BRepAlgo_BooleanOperation
from OCC.Core.BOPAlgo import BOPAlgo_Splitter, BOPAlgo_MakerVolume, BOPAlgo_Builder
from OCC.Core.LocOpe import LocOpe_FindEdges
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Solid, TopoDS_Shape, TopoDS_Face
from OCC.Core.TopoDS import TopoDS_Edge
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_SHAPE, TopAbs_SOLID, TopAbs_FACE
from OCC.Core.LocOpe import LocOpe_FindEdges, LocOpe_FindEdgesInFace
from OCC.Core.TopTools import TopTools_ListOfShape
from OCC.Core.GProp import GProp_GProps
from OCC.Extend.DataExchange import write_step_file, write_stl_file
from OCC.Extend.ShapeFactory import make_face
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCCUtils.Construct import vec_to_dir
from OCCUtils.Construct import make_box


class CovExp (object):

    def __init__(self, file=False):
        self.prop = GProp_GProps()
        self.base = make_box(1000, 1000, 1000)
        self.base_vol = self.cal_vol(self.base)

        self.splitter = BOPAlgo_Splitter()
        self.splitter.AddArgument(self.base)
        print(self.cal_vol(self.base))

    def fileout(self):
        num = 0
        stp_file = "./shp/shp_{:04d}.stp".format(num)
        write_step_file(self.base, stp_file)

        sol_exp = TopExp_Explorer(self.splitter.Shape(), TopAbs_SOLID)
        while sol_exp.More():
            num += 1
            stp_file = "./shp/shp_{:04d}.stp".format(num)
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
            plan = self.pln_on_face(face)
            self.prop_edge(edge)
            print(face, self.cal_are(face), plan)
            print(plan, plan.Axis())
            #print(self.cal_len(edge), self.cal_are(face))
            find_edge.Next()

    def face_init(self, face=TopoDS_Face()):
        self.tmp_face = face
        self.tmp_plan = self.pln_on_face(self.tmp_face)
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
        self.exp = TopExp_Explorer(self.splitter.Shape(), TopAbs_SOLID)
        while self.exp.More():
            self.prop_soild(self.exp.Current())
            self.exp.Next()


if __name__ == "__main__":
    obj = CovExp()
    obj.split_run(10)
    obj.prop_solids()

    print(obj.cal_vol())
    obj.prop_soild(obj.base)
    obj.fileout()
