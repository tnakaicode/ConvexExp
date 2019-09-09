import numpy as np

from OCC.Display.SimpleGui import init_display
from OCC.Core.gp import gp_Pln
from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.TopoDS import TopoDS_Compound
from OCC.Core.BOPAlgo import BOPAlgo_MakerVolume, BOPAlgo_Builder
from OCC.Core.BRep import BRep_Builder
from OCC.Core.GEOMAlgo import GEOMAlgo_Splitter
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_SHAPE, TopAbs_SOLID
from OCC.Core.TopExp import TopExp_Explorer
from OCCUtils.Topology import Topo
from OCCUtils.Construct import make_box, make_face
from OCCUtils.Construct import vec_to_dir

if __name__ == "__main__":
    display, start_display, add_menu, add_function_to_menu = init_display()

    box = make_box(100, 100, 100)

    p1, v1 = gp_Pnt(50, 50, 50), gp_Vec(0, 0, -1)
    fc1 = make_face(gp_Pln(p1, vec_to_dir(v1)), -1000, 1000, -
                    1000, 1000)  # limited, not infinite plane

    p2, v2 = gp_Pnt(50, 50, 50), gp_Vec(0, 1, -1)
    fc2 = make_face(gp_Pln(p2, vec_to_dir(v2)), -1000, 1000, -
                    1000, 1000)  # limited, not infinite plane

    p3, v3 = gp_Pnt(50, 50, 25), gp_Vec(1, 1, -1)
    fc3 = make_face(gp_Pln(p3, vec_to_dir(v3)), -1000, 1000, -
                    1000, 1000)  # limited, not infinite plane

    bo = BOPAlgo_Builder()
    bo.AddArgument(box)
    bo.AddArgument(fc1)
    bo.AddArgument(fc2)
    bo.AddArgument(fc3)

    splitter = GEOMAlgo_Splitter()
    splitter.AddArgument(box)
    splitter.AddTool(fc1)
    splitter.AddTool(fc2)
    splitter.AddTool(fc3)
    splitter.Perform()
    #display.DisplayShape(splitter.Shape())
    
    exp = TopExp_Explorer(splitter.Shape(), TopAbs_SOLID)
    shp = []
    while exp.More():
        shp.append(exp.Current())
        exp.Next()
    display.DisplayShape(shp[0])

    """bo.Perform()
    print("error status: {}".format(bo.ErrorStatus()))

    colos = ["BLUE", "RED", "GREEN", "YELLOW", "BLACK", "WHITE", "BLUE", "RED"]

    top = Topo(bo.Shape())
    for i, sol in enumerate(top.solids()):
        display.DisplayShape(sol, color=colos[i], transparency=0.5)"""

    display.FitAll()
    start_display()
