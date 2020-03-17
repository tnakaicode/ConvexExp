import numpy as np

from base import BoxSplit

if __name__ == "__main__":
    obj = BoxSplit()
    obj.split_run()
    obj.create_tempdir(-1)
    obj.fileout()
