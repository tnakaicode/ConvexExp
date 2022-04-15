import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import scipy as sp
import sys
import pickle
import json
import time
import os
import glob
import shutil
import datetime
import platform
import subprocess
from scipy import ndimage
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize, minimize_scalar, OptimizeResult
import argparse
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('parso').setLevel(logging.ERROR)


def which(program):
    """Run the Unix which command in Python."""
    import os

    def is_exe(fpath):
        """Check if file is executable."""
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, _ = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


def sys_flush(n):
    sys.stdout.write("\r " + " / ".join(map(str, n)))
    sys.stdout.flush()


def split_filename(filename="./temp_20200408000/not_ignore.txt"):
    name = os.path.basename(filename)
    rootname, ext_name = os.path.splitext(name)
    return name, rootname


def create_tempdir(name="temp", flag=1, d="./"):
    print(datetime.date.today(), time.ctime())
    datenm = "{0:%Y%m%d}".format(datetime.date.today())
    dirnum = len(glob.glob(d + "{}_{}*/".format(name, datenm)))
    if flag == -1 or dirnum == 0:
        tmpdir = d + "{}_{}{:03}/".format(name, datenm, dirnum)
        os.makedirs(tmpdir)
        fp = open(tmpdir + "not_ignore.txt", "w")
        fp.close()
    else:
        tmpdir = d + "{}_{}{:03}/".format(name, datenm, dirnum - 1)
    return tmpdir


def create_tempnum(name, tmpdir="./", ext=".tar.gz"):
    num = len(glob.glob(tmpdir + name + "*" + ext)) + 1
    filename = '{}{}_{:03}{}'.format(tmpdir, name, num, ext)
    #print(num, filename)
    return filename


def create_tempdate(name, tmpdir="./", ext=".tar.gz"):
    print(datetime.date.today())
    datenm = "{0:%Y%m%d}".format(datetime.date.today())
    num = len(glob.glob(tmpdir + name + "_{}*".format(datenm) + ext)) + 1
    filename = '{}{}_{}{:03}{}'.format(tmpdir, name, datenm, num, ext)
    return filename


class SetDir (object):

    def __init__(self, temp=True):
        self.root_dir = os.getcwd()
        self.tempname = ""
        self.rootname = ""
        self.date_text = datetime.date.today().strftime("%Y.%m.%d")

        pyfile = sys.argv[0]
        self.filename = os.path.basename(pyfile)
        self.rootname, ext_name = os.path.splitext(self.filename)

        if temp == True:
            self.create_tempdir()
            self.tempname = self.tmpdir + self.rootname
            print(self.rootname)
        else:
            print(self.tmpdir)

    def init(self):
        self.tempname = self.tmpdir + self.rootname

    def create_tempdir(self, name="temp", flag=1, d="./"):
        self.tmpdir = create_tempdir(name, flag, d)
        self.tempname = self.tmpdir + self.rootname
        print(self.tmpdir)

    def create_dir(self, name="temp"):
        os.makedirs(name, exist_ok=True)
        if os.path.isdir(name):
            os.makedirs(name, exist_ok=True)
            fp = open(name + "not_ignore.txt", "w")
            fp.close()
            print("make {}".format(name))
        else:
            print("already exist {}".format(name))
        return name

    def create_dirnum(self, name="./temp", flag=+1):
        dirnum = len(glob.glob("{}_*/".format(name))) + flag
        if dirnum < 0:
            dirnum = 0
        dirname = name + "_{:03}/".format(dirnum)
        os.makedirs(dirname, exist_ok=True)
        fp = open(dirname + "not_ignore.txt", "w")
        fp.close()
        print("make {}".format(dirname))
        return dirname

    def add_tempdir(self, dirname="./", name="temp", flag=1):
        self.tmpdir = dirname
        self.tmpdir = create_tempdir(self.tmpdir + name, flag)
        self.tempname = self.tmpdir + self.rootname
        print(self.tmpdir)

    def add_dir(self, name="temp"):
        dirnum = len(glob.glob("{}/{}/".format(self.tmpdir, name)))
        if dirnum == 0:
            tmpdir = "{}/{}/".format(self.tmpdir, name)
            os.makedirs(tmpdir)
            fp = open(tmpdir + "not_ignore.txt", "w")
            fp.close()
            print("make {}".format(tmpdir))
        else:
            tmpdir = "{}/{}/".format(self.tmpdir, name)
            print("already exist {}".format(tmpdir))
        return tmpdir

    def add_dir_num(self, name="temp", flag=-1):
        if flag == -1:
            num = len(glob.glob("{}/{}_*".format(self.tmpdir, name))) + 1
        else:
            num = len(glob.glob("{}/{}_*".format(self.tmpdir, name)))
        tmpdir = "{}/{}_{:03}/".format(self.tmpdir, name, num)
        os.makedirs(tmpdir, exist_ok=True)
        fp = open(tmpdir + "not_ignore.txt", "w")
        fp.close()
        print("make {}".format(tmpdir))
        return tmpdir

    def open_filemanager(self, path="."):
        abspath = os.path.abspath(path)
        if sys.platform == "win32":
            subprocess.run('explorer.exe {}'.format(abspath))
        elif sys.platform == "linux":
            subprocess.check_call(['xdg-open', abspath])
        else:
            subprocess.run('explorer.exe {}'.format(abspath))

    def open_tempdir(self):
        self.open_filemanager(self.tmpdir)

    def open_newtempdir(self):
        self.create_tempdir("temp", -1)
        self.open_tempdir()

    def exit_app(self):
        sys.exit()


class PlotBase(SetDir):

    def __init__(self, aspect="equal", dim=2, temp=True, *args, **kwargs):
        if temp == True:
            SetDir.__init__(self, temp)
        self.dim = dim
        self.new_fig(aspect, *args, **kwargs)

    def new_fig(self, aspect="equal", dim=None, *args, **kwargs):
        if dim == None:
            self.new_fig(aspect=aspect, dim=self.dim, *args, **kwargs)
        elif self.dim == 2:
            self.new_2Dfig(aspect=aspect, *args, **kwargs)
        elif self.dim == 3:
            self.new_3Dfig(aspect=aspect, *args, **kwargs)
        else:
            self.new_2Dfig(aspect=aspect, *args, **kwargs)

    def new_2Dfig(self, aspect="equal", *args, **kwargs):
        self.fig, self.axs = plt.subplots(*args, **kwargs)
        self.axs.set_aspect(aspect)
        self.axs.xaxis.grid()
        self.axs.yaxis.grid()

    def new_3Dfig(self, aspect="equal", *args, **kwargs):
        self.fig = plt.figure(*args, **kwargs)
        self.axs = self.fig.add_subplot(111, projection='3d')
        #self.axs = self.fig.gca(projection='3d')
        # self.axs.set_aspect('equal')

        self.axs.set_xlabel('x')
        self.axs.set_ylabel('y')
        self.axs.set_zlabel('z')

        self.axs.xaxis.grid()
        self.axs.yaxis.grid()
        self.axs.zaxis.grid()

    def SavePng(self, pngname=None, *args, **kwargs):
        if pngname == None:
            pngname = self.tmpdir + self.rootname + ".png"
        self.fig.savefig(pngname, *args, **kwargs)

    def SavePng_Serial(self, pngname=None, *args, **kwargs):
        if pngname == None:
            pngname = self.rootname
            dirname = self.tmpdir
        else:
            if os.path.dirname(pngname) == "":
                dirname = "./"
            else:
                dirname = os.path.dirname(pngname) + "/"
            basename = os.path.basename(pngname)
            pngname, extname = os.path.splitext(basename)
        pngname = create_tempnum(pngname, dirname, ".png")
        self.fig.savefig(pngname, *args, **kwargs)

    def Show(self):
        try:
            plt.show()
        except AttributeError:
            pass

    def plot_close(self):
        plt.close("all")


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


class plot2d (PlotBase):

    def __init__(self, aspect="equal", temp=True, *args, **kwargs):
        PlotBase.__init__(self, aspect, 2, temp, *args, **kwargs)
        # self.dim = 2
        # self.new_2Dfig(aspect=aspect)
        # self.new_fig(aspect=aspect)

    def add_axs(self, row=1, col=1, num=1, aspect="auto"):
        self.axs.set_axis_off()
        axs = self.fig.add_subplot(row, col, num)
        axs.set_aspect(aspect)
        axs.xaxis.grid()
        axs.yaxis.grid()
        return axs

    def add_twin(self, aspect="auto", side="right", out=0):
        axt = self.axs.twinx()
        # axt.axis("off")
        axt.set_aspect(aspect)
        axt.xaxis.grid()
        axt.yaxis.grid()
        axt.spines[side].set_position(('axes', out))
        make_patch_spines_invisible(axt)
        axt.spines[side].set_visible(True)
        return axt

    def div_axs(self):
        self.div = make_axes_locatable(self.axs)
        # self.axs.set_aspect('equal')

        self.ax_x = self.div.append_axes(
            "bottom", 1.0, pad=0.5, sharex=self.axs)
        self.ax_x.xaxis.grid(True, zorder=0)
        self.ax_x.yaxis.grid(True, zorder=0)

        self.ax_y = self.div.append_axes(
            "right", 1.0, pad=0.5, sharey=self.axs)
        self.ax_y.xaxis.grid(True, zorder=0)
        self.ax_y.yaxis.grid(True, zorder=0)

    def contourf_sub(self, mesh, func, sxy=[0, 0], pngname=None):
        self.new_fig()
        self.div_axs()
        nx, ny = mesh[0].shape
        sx, sy = sxy
        xs, xe = mesh[0][0, 0], mesh[0][0, -1]
        ys, ye = mesh[1][0, 0], mesh[1][-1, 0]
        mx = np.searchsorted(mesh[0][0, :], sx) - 1
        my = np.searchsorted(mesh[1][:, 0], sy) - 1

        self.ax_x.plot(mesh[0][mx, :], func[mx, :])
        self.ax_x.set_title("y = {:.2f}".format(sy))
        self.ax_y.plot(func[:, my], mesh[1][:, my])
        self.ax_y.set_title("x = {:.2f}".format(sx))
        im = self.axs.contourf(*mesh, func, cmap="jet")
        self.fig.colorbar(im, ax=self.axs, shrink=0.9)
        self.fig.tight_layout()
        self.SavePng(pngname)

    def contourf_tri(self, x, y, z):
        self.new_fig()
        self.axs.tricontourf(x, y, z, cmap="jet")

    def contourf_div(self, mesh, func, loc=[0, 0], txt="", title="name", pngname="./tmp/png", level=None):
        sx, sy = loc
        nx, ny = func.shape
        xs, ys = mesh[0][0, 0], mesh[1][0, 0]
        xe, ye = mesh[0][0, -1], mesh[1][-1, 0]
        dx, dy = mesh[0][0, 1] - mesh[0][0, 0], mesh[1][1, 0] - mesh[1][0, 0]
        mx, my = int((sy - ys) / dy), int((sx - xs) / dx)
        tx, ty = 1.1, 0.0

        self.new_2Dfig()
        self.div_axs()
        self.ax_x.plot(mesh[0][mx, :], func[mx, :])
        self.ax_x.set_title("y = {:.2f}".format(sy))

        self.ax_y.plot(func[:, my], mesh[1][:, my])
        self.ax_y.set_title("x = {:.2f}".format(sx))

        self.fig.text(tx, ty, txt, transform=self.ax_x.transAxes)
        im = self.axs.contourf(*mesh, func, cmap="jet", levels=level)
        self.axs.set_title(title)
        self.fig.colorbar(im, ax=self.axs, shrink=0.9)

        plt.tight_layout()
        plt.savefig(pngname + ".png")

    def contourf_div_auto(self, mesh, func, loc=[0, 0], txt="", title="name", pngname="./tmp/png", level=None):
        sx, sy = loc
        nx, ny = func.shape
        xs, ys = mesh[0][0, 0], mesh[1][0, 0]
        xe, ye = mesh[0][0, -1], mesh[1][-1, 0]
        dx, dy = mesh[0][0, 1] - mesh[0][0, 0], mesh[1][1, 0] - mesh[1][0, 0]
        mx, my = int((sy - ys) / dy), int((sx - xs) / dx)
        tx, ty = 1.1, 0.0

        self.new_2Dfig()
        self.div_axs()
        self.axs.set_aspect('auto')
        self.ax_x.plot(mesh[0][mx, :], func[mx, :])
        self.ax_x.set_title("y = {:.2f}".format(sy))

        self.ax_y.plot(func[:, my], mesh[1][:, my])
        self.ax_y.set_title("x = {:.2f}".format(sx))

        self.fig.text(tx, ty, txt, transform=self.ax_x.transAxes)
        im = self.axs.contourf(*mesh, func, cmap="jet", levels=level)
        self.axs.set_title(title)
        self.fig.colorbar(im, ax=self.axs, shrink=0.9)

        plt.tight_layout()
        plt.savefig(pngname + ".png")


class plot3d (PlotBase):

    def __init__(self, aspect="equal", temp=True, *args, **kwargs):
        PlotBase.__init__(self, aspect, 3, temp, *args, **kwargs)
        #self.dim = 3
        # self.new_fig()

    def set_axes_equal(self, axis="xyz"):
        '''
        Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

        Input
          ax: a matplotlib axis, e.g., as output from plt.gca().
        '''

        x_limits = self.axs.get_xlim3d()
        y_limits = self.axs.get_ylim3d()
        z_limits = self.axs.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])

        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5 * max([x_range, y_range, z_range])

        for i in axis:
            if i == "x":
                self.axs.set_xlim3d(
                    [x_middle - plot_radius, x_middle + plot_radius])
            elif i == "y":
                self.axs.set_ylim3d(
                    [y_middle - plot_radius, y_middle + plot_radius])
            elif i == "z":
                self.axs.set_zlim3d(
                    [z_middle - plot_radius, z_middle + plot_radius])
            else:
                self.axs.set_zlim3d(
                    [z_middle - plot_radius, z_middle + plot_radius])

    def plot_ball(self, rxyz=[1, 1, 1]):
        u = np.linspace(0, 1, 10) * 2 * np.pi
        v = np.linspace(0, 1, 10) * np.pi
        uu, vv = np.meshgrid(u, v)
        x = rxyz[0] * np.cos(uu) * np.sin(vv)
        y = rxyz[1] * np.sin(uu) * np.sin(vv)
        z = rxyz[2] * np.cos(vv)

        self.axs.plot_wireframe(x, y, z)
        self.set_axes_equal()
        #self.axs.set_xlim3d(-10, 10)
        #self.axs.set_ylim3d(-10, 10)
        #self.axs.set_zlim3d(-10, 10)


class plotpolar (plot2d):

    def __init__(self, aspect="equal", *args, **kwargs):
        plot2d.__init__(self, *args, **kwargs)
        self.dim = 2
        self.new_polar(aspect)

    def new_polar(self, aspect="equal"):
        self.new_fig(aspect=aspect)
        self.axs.set_axis_off()
        self.axs = self.fig.add_subplot(111, projection='polar')

    def plot_polar(self, px, py, arrow=True, **kwargs):
        plt.polar(px, py, **kwargs)

        if arrow == True:
            num = np.linspace(1, len(px) - 1, 6)
            for idx, n in enumerate(num):
                n = int(n)
                plt.arrow(
                    px[-n], py[-n],
                    (px[-n + 1] - px[-n]) / 100.,
                    (py[-n + 1] - py[-n]) / 100.,
                    head_width=0.1,
                    head_length=0.2,
                )
                plt.text(px[-n], py[-n], "n={:d}".format(idx))


class LineDrawer(object):

    def __init__(self, dirname="./tmp/", txtname="plot_data"):
        self.trajectory = None
        self.xx = []
        self.yy = []
        self.id = 0
        self.fg = 0

        self.dirname = dirname
        self.txtname = dirname + txtname
        self.fp = open(self.txtname + ".txt", "w")

        self.init_fig()

    def run_base(self):
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('key_press_event', self.onkey)
        animation.FuncAnimation(
            self.fig, self.anim_animate, init_func=self.anim_init, frames=30, interval=100, blit=True)

    def init_fig(self):
        self.fig, self.axs = plt.subplots()
        self.axs.set_aspect('equal')
        self.axs.xaxis.grid()
        self.axs.yaxis.grid()
        self.divider = make_axes_locatable(self.axs)

        self.traj_line, = self.axs.plot([], [], 'o', markersize=4, mew=4)
        self.record_line, = self.axs.plot(
            [], [], 'o', markersize=4, mew=4, color='m')
        self.empty, = self.axs.plot([], [])

    def onclick(self, event):
        txt = ""

        # get mouse position and scale appropriately to convert to (x,y)
        if event.xdata is not None:
            self.trajectory = np.array([event.xdata, event.ydata])
            txt += "event.x_f {:.3f} ".format(event.xdata)
            txt += "event.y_f {:.3f} ".format(event.ydata)

        if event.button == 1:
            self.id += 1
            txt += "event {:d} ".format(event.button)
            txt += "event.x_d {:d} ".format(event.x)
            txt += "event.y_d {:d} ".format(event.y)
            txt += "flag {:d} {:d}".format(self.id % 2, self.id)
            self.xx.append(event.xdata)
            self.yy.append(event.ydata)
        elif event.button == 3:
            dat_txt = "data-{:d} num {:d}\n".format(self.fg, self.id)
            for i in range(self.id):
                dat_txt += "{:d} ".format(i)
                dat_txt += "{:.3f} ".format(self.xx[i])
                dat_txt += "{:.3f} ".format(self.yy[i])
                dat_txt += "\n"

            self.fp.write(dat_txt)
            self.fp.write("\n\n")

            self.fq = open(self.txtname + "-{:d}.txt".format(self.fg), "w")
            self.fq.write(dat_txt)
            self.fq.close()

            self.xx = []
            self.yy = []
            self.id = 0
            self.fg += 1

        print(txt)

    def onkey(self, event):
        print("onkey", event, event.xdata)
        # Record
        if event.key == 'r':
            traj = np.array([self.xx, self.yy])
            with open('traj.pickle', 'w') as f:
                pickle.dump(traj, f)
                # f.close()

    def anim_init(self):
        self.traj_line.set_data([], [])
        self.record_line.set_data([], [])
        self.empty.set_data([], [])

        return self.traj_line, self.record_line, self.empty

    def anim_animate(self, i):
        if self.trajectory is not None:
            self.traj_line.set_data(self.trajectory)

        if self.xx is not None:
            self.record_line.set_data(self.xx, self.yy)

        self.empty.set_data([], [])

        return self.traj_line, self.record_line, self.empty

    def show(self):
        try:
            plt.show()
        except AttributeError:
            pass


if __name__ == '__main__':
    obj = SetDir()
    obj.create_tempdir(flag=-1)
    # for i in range(5):
    #    name = "temp{:d}".format(i)
    #    obj.add_dir(name)
    # obj.add_dir(name)
    # obj.tmpdir = obj.add_dir("temp")
    # obj.add_dir("./temp/{}/".format(name))
