import numpy as np
import sys

import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('parso').setLevel(logging.ERROR)

from OCC.Display.SimpleGui import init_display
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir
from OCC.Core.gp import gp_Ax1, gp_Ax2, gp_Ax3
from OCC.Core.gp import gp_Pln, gp_Lin
from OCC.Core.gp import gp_Trsf
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_SOLID
from OCC.Extend.DataExchange import write_step_file, write_stl_file
from OCC.Extend.ShapeFactory import make_face, make_edge
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCCUtils.Construct import make_box
from OCCUtils.Topology import shapeTypeString, dumpTopology
from OCCUtils.Construct import vec_to_dir, dir_to_vec

from src.convex import CovExp


if __name__ == "__main__":
    obj = CovExp(touch=False)
    
    snum = 3
    seed = 11
    
    obj.init_base(obj.base)
    obj.split_run(snum, seed)
    obj.nsol = 2
    obj.nfce = 1
    
    sol_exp = TopExp_Explorer( obj.splitter.Shape(), TopAbs_SOLID)
    sol_top = TopologyExplorer(obj.splitter.Shape())

    if sol_top.number_of_solids() < obj.nsol:
        obj.nsol = 1

    sol_num = 1
    sol = sol_exp.Current()
    while sol_exp.More() and obj.nsol > sol_num:
        sol = sol_exp.Current()
        sol_num += 1
        sol_exp.Next()

    obj.prop_soild(sol)
    obj.display.DisplayShape(sol, transparency=0.5)
    obj.display.DisplayShape(obj.base, transparency=0.8)
    obj.ShowOCC()

    # print(obj.cal_vol())
    # obj.prop_soild(obj.base)

    # obj.fileout()
    # obj.ShowDisplay()
