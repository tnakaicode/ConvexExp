import numpy as np

from src.base import BoxSplit

if __name__ == "__main__":
    obj = BoxSplit()
    obj.split_run()
    obj.create_tempdir(flag=-1)
    obj.fileout()
