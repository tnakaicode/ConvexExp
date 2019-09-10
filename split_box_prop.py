import numpy as np

from OCC.Display.SimpleGui import init_display
from OCC.Core.gp import gp_Pln
from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Solid, TopoDS_Shape
from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepGProp import brepgprop_LinearProperties, brepgprop_VolumeProperties
from OCC.Core.GEOMAlgo import GEOMAlgo_Splitter
from OCC.Core.BOPAlgo import BOPAlgo_MakerVolume, BOPAlgo_Builder
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_SHAPE, TopAbs_SOLID
from OCC.Core.TopTools import TopTools_ListOfShape
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.GProp import GProp_GProps
from OCC.Extend.DataExchange import write_step_file, write_stl_file
from OCC.Extend.ShapeFactory import make_box, make_face
from OCCUtils.Construct import vec_to_dir

if __name__ == "__main__":
    num = 0
    props = GProp_GProps()
    box = make_box(100, 100, 100)

    vol = brepgprop_VolumeProperties(box, props)
    print(vol)

    splitter = GEOMAlgo_Splitter()
    splitter.AddArgument(box)

    for i in range(5):
        pnt = gp_Pnt(*np.random.rand(3) * 100)
        vec = gp_Vec(*np.random.randn(3))
        pln = gp_Pln(pnt, vec_to_dir(vec))
        fce = make_face(pln, -1000, 1000, -1000, 1000)
        splitter.AddTool(fce)

    splitter.Perform()
    
    exp = TopExp_Explorer(splitter.Shape(), TopAbs_SOLID)
    shp = []
    vol = 0
    while exp.More():
        props = GProp_GProps()
        num += 1
        
        cov = exp.Current()
        shp.append(cov)

        brepgprop_VolumeProperties(cov, props)
        vol += props.Mass()
        print(props.Mass(), vol)
        exp.Next()
