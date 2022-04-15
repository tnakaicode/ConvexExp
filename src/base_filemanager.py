import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
import subprocess
import argparse
from linecache import getline, clearcache

sys.path.append(os.path.join("./"))

import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)

if __name__ == '__main__':
    argvs = sys.argv
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", dest="dir", default="./")
    parser.add_argument("--pxyz", dest="pxyz",
                      default=[0.0, 0.0, 0.0], type=float, nargs=3)
    opt = parser.parse_args()
    print(opt, argvs)

    print(sys.platform)
    path = "."
    if sys.platform == "win32":
        subprocess.run('explorer.exe {}'.format(path))
    elif sys.platform == "linux":
        subprocess.check_call(['xdg-open', path])
    else:
        subprocess.run('explorer.exe {}'.format(path))
