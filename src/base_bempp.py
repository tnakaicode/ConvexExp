import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
import subprocess
import tempfile
from scipy.spatial import ConvexHull, Delaunay
import argparse
from linecache import getline, clearcache

import bempp.api

sys.path.append(os.path.join("./"))
from base import plot2d, plot3d
# from base_occ import dispocc

import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)


class plotBEM (plot2d):

    k = 15

    def __init__(self):
        plot2d.__init__(self)
        self.grid = self.reference_triangle()

    def reference_triangle(self):
        """Return a grid consisting of only the reference triangle."""

        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).T
        elements = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]).T
        return bempp.api.Grid(vertices, elements)

    def __generate_grid_from_geo_string(self, geo_string):
        """Create a grid from a gmsh geo string."""

        def msh_from_string(geo_string):
            """Create a mesh from a string."""
            gmsh_command = bempp.api.GMSH_PATH
            if gmsh_command is None:
                raise RuntimeError("Gmsh is not found. Cannot generate mesh")
            f, geo_name, msh_name = self.get_gmsh_file()
            f.write(geo_string)
            f.close()

            fnull = open(os.devnull, "w")
            cmd = gmsh_command + " -2 " + geo_name + " -format msh2"
            try:
                subprocess.check_call(
                    cmd, shell=True, stdout=fnull, stderr=fnull)
            except:
                print("The following command failed: " + cmd)
                fnull.close()
                raise
            # os.remove(geo_name)
            fnull.close()
            # self.open_filemanager(geo_name)
            return msh_name

        msh_name = msh_from_string(geo_string)
        grid = bempp.api.import_grid(msh_name)
        # os.remove(msh_name)
        # self.open_filemanager(msh_name)
        return grid

    def get_gmsh_file(self):
        """
        Create a new temporary gmsh file.

        Return a 3-tuple (geo_file,geo_name,msh_name), where
        geo_file is a file descriptor to an empty .geo file, geo_name is
        the corresponding filename and msh_name is the name of the
        Gmsh .msh file that will be generated.

        """

        geo, geo_name = tempfile.mkstemp(
            suffix=".geo", dir=self.tmpdir, text=True)
        geo_file = os.fdopen(geo, "w")
        msh_name = os.path.splitext(geo_name)[0] + ".msh"
        return (geo_file, geo_name, msh_name)

    def export_bempp_msh(self):
        bempp.api.export(self.tempname + ".msh", self.grid, write_binary=False)

    def convex_3d(self):
        self.new_3Dfig()

        self.axs.scatter(
            self.grid.vertices[0, :],
            self.grid.vertices[1, :],
            self.grid.vertices[2, :],
            s=0.5
        )
        self.SavePng(self.tempname + "_pnt.png")

        for idx in range(self.grid.edges.shape[1]):
            xi = self.grid.vertices[:, self.grid.edges[0, idx]]
            yi = self.grid.vertices[:, self.grid.edges[1, idx]]
            self.axs.plot([xi[0], yi[0]], [xi[1], yi[1]],
                          [xi[2], yi[2]], "k", lw=0.5)
        self.SavePng(self.tempname + "_edg.png")


if __name__ == '__main__':
    argvs = sys.argv
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", dest="dir", default="./")
    parser.add_argument("--pxyz", dest="pxyz",
                      default=[0.0, 0.0, 0.0], type=float, nargs=3)
    opt = parser.parse_args()
    print(opt, argvs)

    obj = plotBEM()
    bempp.api.export(obj.tempname + ".msh", obj.grid, write_binary=False)
    obj.convex_3d()
