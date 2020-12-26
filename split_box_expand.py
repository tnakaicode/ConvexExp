import numpy as np
import sys

import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('parso').setLevel(logging.ERROR)

from src.convex import CovExp


if __name__ == "__main__":
    obj = CovExp(touch=True, file=True)
    obj.split_run(3)
    obj.prop_solids()

    """sol_exp = TopExp_Explorer(obj.splitter.Shape(), TopAbs_SOLID)
    obj.prop_soild(sol_exp.Current())

    obj.display.DisplayShape(sol_exp.Current(), transparency=0.5)

    obj.display.FitAll()
    obj.start_display()"""

    # print(obj.cal_vol())
    # obj.prop_soild(obj.base)

    obj.fileout()
    obj.show_split_solid()
