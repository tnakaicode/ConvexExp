import numpy as np
import random

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

colos = ["BLUE", "RED", "GREEN", "YELLOW", "BLACK", "WHITE"]

if __name__ == "__main__":
    display, start_display, add_menu, add_function_to_menu = init_display()

    box = make_box(100, 100, 100)

    bo = BOPAlgo_Builder()
    bo.AddArgument(box)
    for i in range(5):
        pnt = gp_Pnt(*np.random.rand(3) * 100)
        vec = gp_Vec(*np.random.randn(3))
        pln = gp_Pln(pnt, vec_to_dir(vec))
        fce = make_face(pln, -1000, 1000, -1000, 1000)
        bo.AddArgument(fce)
        print(pnt, vec)

    bo.Perform()
    print("error status: {}".format(bo.ErrorStatus()))

    top = Topo(bo.Shape())
    for i, sol in enumerate(top.solids()):
        display.DisplayShape(
            sol, color=colos[i % len(colos)], transparency=0.5)

    display.FitAll()
    start_display()
