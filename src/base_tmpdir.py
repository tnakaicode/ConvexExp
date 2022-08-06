import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import time
import os
import glob
import shutil
import datetime
import argparse
from datetime import date, datetime


def create_tempdir(name="temp", flag=1, d="./"):
    print(date.today(), time.ctime())
    datenm = date.today().strftime("%Y%m%d")
    dirnum = len(glob.glob(d + "{}_{}*/".format(name, datenm)))
    if flag == -1 or dirnum == 0:
        tmpdir = d + "{}_{}{:03}/".format(name, datenm, dirnum)
        os.makedirs(tmpdir)
        fp = open(tmpdir + "not_ignore.txt", "w")
        fp.close()
    else:
        tmpdir = d + "{}_{}{:03}/".format(name, datenm, dirnum - 1)
    return tmpdir


def logistic_1d(px, x0=np.pi / 2, k=0.1):
    return np.pi / (1 + np.exp(-k * (px - x0)))


def sit_t(px, t=np.pi):
    return (t / np.pi) * np.sin(px) - (np.sin(px / 2) / np.sin(t / 2))**2 * np.sin(t)


if __name__ == '__main__':
    argvs = sys.argv
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", dest="name", default="temp")
    parser.add_argument("--flag", dest="flag", default=1, type=int)
    opt = parser.parse_args()
    print(opt, argvs)

    tmpdir = create_tempdir(name=opt.name, flag=opt.flag)
    print(tmpdir)

    px = np.linspace(-1, 0.5, 100) * np.pi
    py = np.sin(px)

    plt.figure()
    plt.grid()
    plt.plot(px, py)
    plt.plot(px, np.sin(py))
    plt.plot(py, np.sin(py))
    plt.savefig(tmpdir + "plot_sin_sin.png")

    fig, axs = plt.subplots()
    plt.grid()
    axs.set_aspect("equal")
    plt.plot(np.cos(px), np.sin(px))
    plt.savefig(tmpdir + "plot_xy.png")

    px = np.linspace(-10, 10, 100)

    plt.figure()
    plt.grid()
    plt.plot(px, logistic_1d(px, x0=0.0, k=0.5), label="k=0.5")
    plt.plot(px, logistic_1d(px, x0=0.0, k=1.0), label="k=1.0")
    plt.plot(px, logistic_1d(px, x0=0.0, k=2.0), label="k=2.0")
    plt.plot(px, logistic_1d(px, x0=0.0, k=5.0), label="k=5.0")
    plt.legend()
    plt.savefig(tmpdir + "plot_logistic.png")

    px = np.linspace(-1, 1, 500) * 5

    plt.figure()
    plt.grid()
    plt.plot(px, np.tanh(px / 0.5), label="tanh=0.5")
    plt.plot(px, np.tanh(px / 1.0), label="tanh=1.0")
    plt.plot(px, np.tanh(px / 2.0), label="tanh=2.0")
    plt.plot(px, np.tanh(px / 5.0), label="tanh=5.0")
    plt.savefig(tmpdir + "plot_tanh.png")
