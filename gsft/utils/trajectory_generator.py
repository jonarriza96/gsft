import numpy as np
import casadi as cs
from scipy.interpolate import CubicSpline


def backflip(a=0.25, w=6):
    T = cs.SX.sym("T")

    x = 0 * T
    y = a * cs.sin(w * T)
    z = -a * cs.cos(w * T)

    xLd = cs.vertcat(x, y, z)
    path = {"T": T, "xLd": xLd}

    return path


def lissajous(A, B, C, a, b, c, w):
    T = cs.SX.sym("T")

    x = A * cs.cos(a * w * T)
    y = B * cs.sin(b * w * T)
    z = C * cs.sin(c * w * T)

    xLd = cs.vertcat(x, y, z)
    path = {"T": T, "xLd": xLd}

    return path


def straight_line(l, direction=np.array([1, 0, 0])):
    T = cs.SX.sym("T")

    xLd = l * T * cs.DM(direction)
    path = {"T": T, "xLd": xLd}

    return path


def generate_interpolating_functions(mJTraj):
    f_pdx = CubicSpline(mJTraj["t"], mJTraj["p"][0])
    f_pdy = CubicSpline(mJTraj["t"], mJTraj["p"][1])
    f_pdz = CubicSpline(mJTraj["t"], mJTraj["p"][2])
    f_pd = lambda t: np.array([f_pdx(t), f_pdy(t), f_pdz(t)])

    f_vdx = CubicSpline(mJTraj["t"], mJTraj["v"][0])
    f_vdy = CubicSpline(mJTraj["t"], mJTraj["v"][1])
    f_vdz = CubicSpline(mJTraj["t"], mJTraj["v"][2])
    f_vd = lambda t: np.array([f_vdx(t), f_vdy(t), f_vdz(t)])

    f_adx = CubicSpline(mJTraj["t"], mJTraj["a"][0])
    f_ady = CubicSpline(mJTraj["t"], mJTraj["a"][1])
    f_adz = CubicSpline(mJTraj["t"], mJTraj["a"][2])
    f_ad = lambda t: np.array([f_adx(t), f_ady(t), f_adz(t)])

    f_jdx = CubicSpline(mJTraj["t"], mJTraj["j"][0])
    f_jdy = CubicSpline(mJTraj["t"], mJTraj["j"][1])
    f_jdz = CubicSpline(mJTraj["t"], mJTraj["j"][2])
    f_jd = lambda t: np.array([f_jdx(t), f_jdy(t), f_jdz(t)])

    f_sdx = CubicSpline(mJTraj["t"], mJTraj["s"][0])
    f_sdy = CubicSpline(mJTraj["t"], mJTraj["s"][1])
    f_sdz = CubicSpline(mJTraj["t"], mJTraj["s"][2])
    f_sd = lambda t: np.array([f_sdx(t), f_sdy(t), f_sdz(t)])

    f_wdx = CubicSpline(mJTraj["t"], mJTraj["ang_vel"][0])
    f_wdy = CubicSpline(mJTraj["t"], mJTraj["ang_vel"][1])
    f_wdz = CubicSpline(mJTraj["t"], mJTraj["ang_vel"][2])
    f_wd = lambda t: np.array([f_wdx(t), f_wdy(t), f_wdz(t)])

    f_rdx = CubicSpline(mJTraj["t"], mJTraj["ang_acc"][0])
    f_rdy = CubicSpline(mJTraj["t"], mJTraj["ang_acc"][1])
    f_rdz = CubicSpline(mJTraj["t"], mJTraj["ang_acc"][2])
    f_rd = lambda t: np.array([f_rdx(t), f_rdy(t), f_rdz(t)])

    return f_pd, f_vd, f_ad, f_jd, f_sd, f_wd, f_rd
