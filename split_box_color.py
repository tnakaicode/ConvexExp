import numpy as np
import random

import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('parso').setLevel(logging.ERROR)

from src.convex import CovExp


if __name__ == "__main__":
    obj = CovExp(touch=True)
    obj.split_run(5)
    obj.show_split_solid()
