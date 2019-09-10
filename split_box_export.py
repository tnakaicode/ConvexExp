import numpy as np

from OCC.Display.SimpleGui import init_display
from OCC.Core.gp import gp_Pln
from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.TopoDS import TopoDS_Compound
from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepGProp import brepgprop_LinearProperties
from OCC.Core.GEOMAlgo import GEOMAlgo_Splitter
from OCC.Core.BOPAlgo import BOPAlgo_MakerVolume, BOPAlgo_Builder
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_SHAPE, TopAbs_SOLID
from OCC.Core.TopTools import TopTools_ListOfShape
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.GProp import GProp_GProps
from OCC.Extend.DataExchange import write_step_file, write_stl_file
from OCCUtils.Topology import Topo
from OCCUtils.Construct import make_box, make_face
from OCCUtils.Construct import vec_to_dir

if __name__ == "__main__":
    num = 0
    box = make_box(100, 100, 100)

    stpname = "{}/shp_{:04d}.stp".format("./shp/", num)
    write_step_file(box, stpname)

    splitter = GEOMAlgo_Splitter()
    splitter.AddArgument(box)

    for i in range(7):
        pnt = gp_Pnt(*np.random.rand(3) * 100)
        vec = gp_Vec(*np.random.randn(3))
        pln = gp_Pln(pnt, vec_to_dir(vec))
        fce = make_face(pln, -1000, 1000, -1000, 1000)
        splitter.AddTool(fce)

    splitter.Perform()
    #shp_list = splitter.Generated()
    
    exp = TopExp_Explorer(splitter.Shape(), TopAbs_SOLID)
    shp = []
    while exp.More():
        num += 1

        props = GProp_GProps()
        brepgprop_LinearProperties(exp.Current(), props)
        vol = props.Mass()
        print(vol)

        stpname = "./shp/shp_{:04d}_{:05.0f}.stp".format(num, vol)
        write_step_file(exp.Current(), stpname)
        shp.append(exp.Current())
        exp.Next()
