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
    obj = CovExp(touch=True)
    obj.split_run(2)
    # obj.prop_solids()

    sol_exp = TopExp_Explorer(obj.splitter.Shape(), TopAbs_SOLID)
    obj.prop_soild(sol_exp.Current())
    obj.display.DisplayShape(sol_exp.Current(), transparency=0.5)
    obj.show()

    # print(obj.cal_vol())
    # obj.prop_soild(obj.base)

    # obj.fileout()
    # obj.ShowDisplay()
