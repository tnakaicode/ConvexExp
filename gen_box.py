import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
from optparse import OptionParser

import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)

sys.path.append(os.path.join("./"))
from base import plot2d, plotocc

from OCC.Core.gp import gp_Ax3, gp_Pnt, gp_Vec, gp_Dir
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCCUtils.Construct import make_box

if __name__ == '__main__':
    argvs = sys.argv
    parser = OptionParser()
    parser.add_option("--dir", dest="dir", default="./")
    parser.add_option("--pxyz", dest="point",
                      default=[0.0, 0.0, 0.0], type="float", nargs=3)
    opt, argc = parser.parse_args(argvs)
    print(opt, argc)

    obj = plotocc()
    obj.SaveMenu()
    box1 = make_box(gp_Pnt(0, 0, 0), gp_Pnt(100, 100, 100))
    obj.export_stp(box1, stpname="./shp/box_001.stp")

    box2 = make_box(gp_Pnt(50, 50, 50), gp_Pnt(200, 200, 200))
    obj.export_stp(box2, stpname="./shp/box_002.stp")
