import numpy as np

from OCC.Display.SimpleGui import init_display
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir
from OCC.Core.gp import gp_Ax3
from OCC.Core.gp import gp_Pln
from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.BRepGProp import brepgprop_LinearProperties, brepgprop_VolumeProperties
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepAlgo import BRepAlgo_BooleanOperation
from OCC.Core.BOPAlgo import BOPAlgo_MakerVolume, BOPAlgo_Builder
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Solid, TopoDS_Shape
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_SHAPE, TopAbs_SOLID, TopAbs_FACE
from OCC.Core.TopTools import TopTools_ListOfShape
from OCC.Core.TCollection import TCollection_ExtendedString_IsEqual
from OCC.Core.GProp import GProp_GProps
from OCC.Core.GEOMAlgo import GEOMAlgo_Splitter
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf, GeomAPI_ProjectPointOnCurve
from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface, GeomAPI_IntCS
from OCC.Extend.DataExchange import write_step_file, write_stl_file
from OCC.Extend.ShapeFactory import make_box, make_face
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCCUtils.Construct import vec_to_dir


class CovExp (object):

    def __init__(self):
        self.prop = GProp_GProps()
        self.base = make_box(100, 100, 100)
        self.base_vol = self.cal_vol(self.base)

        self.splitter = GEOMAlgo_Splitter()
        self.splitter.AddArgument(self.base)
        print(self.cal_vol(self.base))

    def split_run(self, num=5):
        for i in range(num):
            pnt = gp_Pnt(*np.random.rand(3) * 100)
            vec = gp_Vec(*np.random.randn(3))
            pln = gp_Pln(pnt, vec_to_dir(vec))
            fce = make_face(pln, -1000, 1000, -1000, 1000)
            self.splitter.AddTool(fce)
        self.splitter.Perform()

    def cal_vol(self, shp=TopoDS_Shape()):
        brepgprop_VolumeProperties(shp, self.prop)
        return self.prop.Mass()

    def prop_solids(self):
        self.exp = TopExp_Explorer(self.splitter.Shape(), TopAbs_SOLID)
        while self.exp.More():
            print(self.cal_vol(self.exp.Current()), self.base_vol)
            top = TopologyExplorer(self.exp.Current())
            print(top.number_of_faces())
            print(top.number_of_edges())

            shp_exp = TopExp_Explorer(self.exp.Current(), TopAbs_FACE)
            fce = shp_exp.Current()
            cheker = BRepCheck_Analyzer(fce)
            """shp_exp.Next()
            while shp_exp.More():
                fc1 = shp_exp.Current()
                print(shp_exp.Depth())
                print(shp_exp.Depth(), cheker.IsValid(fc1))
                shp_exp.Next()"""
            self.exp.Next()


if __name__ == "__main__":
    obj = CovExp()
    obj.split_run()
    obj.prop_solids()

    print(obj.cal_vol())
