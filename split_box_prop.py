import numpy as np

from OCC.Display.SimpleGui import init_display
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir
from OCC.Core.gp import gp_Ax3
from OCC.Core.gp import gp_Pln
from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties, brepgprop_VolumeProperties, brepgprop_LinearProperties
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepAlgo import BRepAlgo_BooleanOperation
from OCC.Core.BOPAlgo import BOPAlgo_MakerVolume, BOPAlgo_Builder
from OCC.Core.LocOpe import LocOpe_FindEdges
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Solid, TopoDS_Shape, TopoDS_Face
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_SHAPE, TopAbs_SOLID, TopAbs_FACE
from OCC.Core.TopTools import TopTools_ListOfShape
from OCC.Core.GProp import GProp_GProps
from OCC.Core.GEOMAlgo import GEOMAlgo_Splitter
from OCC.Extend.DataExchange import write_step_file, write_stl_file
from OCC.Extend.ShapeFactory import make_box, make_face
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCCUtils.Construct import vec_to_dir


class CovExp (object):

    def __init__(self, file=False):
        self.prop = GProp_GProps()
        self.base = make_box(100, 200, 300)
        self.base_vol = self.cal_vol(self.base)

        self.splitter = GEOMAlgo_Splitter()
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
            pnt = gp_Pnt(*np.random.rand(3) * 100)
            vec = gp_Vec(*np.random.randn(3))
            pln = gp_Pln(pnt, vec_to_dir(vec))
            fce = make_face(pln, -1000, 1000, -1000, 1000)
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

    def face_expand(self, face=TopoDS_Face()):
        print(face)

        find_edge = LocOpe_FindEdges(self.tmp_face, face)
        find_edge.InitIterator()

        if find_edge.More():
            edge = find_edge.EdgeFrom()
            print(face, self.cal_are(face))
            #print(self.cal_len(edge), self.cal_are(face))
            find_edge.Next()

    def prop_soild(self, sol=TopoDS_Solid()):
        sol_exp = TopExp_Explorer(sol, TopAbs_FACE)
        sol_top = TopologyExplorer(sol)
        #print(self.cal_vol(sol), self.base_vol)
        print(sol_top.number_of_faces())

        self.tmp_face = sol_exp.Current()
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
    obj.split_run()
    obj.prop_solids()

    print(obj.cal_vol())
    obj.prop_soild(obj.base)
    # obj.fileout()
