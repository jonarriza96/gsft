#!/usr/bin/env python
# coding=utf-8

import numpy as np
import casadi as cs
import time

import matplotlib.pyplot as plt
from matplotlib import cm

from gsft.external.traj_gen.python.scripts.traj_gen import poly_trajectory as pt

from gsft.utils.rotations import vec_cross, vec_dot, vee
from gsft.utils.visualize import axis_equal
from gsft.utils.panda import panda_limits


def min_jerk_qp(Xs, Xds, Xdds, Xddds, Xdddds, order=10, knot_end=1):
    """Minimum jerk trajectory generation using QP and uniform knots"""
    # knots = Xs_knots
    knots = np.linspace(0, knot_end, Xs.shape[0])
    optimTarget = "end-derivative"
    objWeights = np.array([0, 0, 1, 1])
    # objWeights = np.array([0, 0, 1, 0])
    dim = 3  # x,y,z
    maxConti = (
        6  # + 2  # continuous until snap + 2 (4th derivative--> qdd + 2 (hessian) cont)
    )
    pTraj = pt.PolyTrajGen(knots, order, optimTarget, dim, maxConti)

    ts = knots.copy()

    verbose = False
    # equality constraints
    for i in range(Xs.shape[0]):
        # position eq. constraint
        pin_ = {"t": ts[i], "d": 0, "X": Xs[i]}
        pTraj.addPin(pin_)

        # velocity
        if not np.isnan(Xds[i]).all():
            pin_ = {"t": ts[i], "d": 1, "X": Xds[i]}
            pTraj.addPin(pin_)
            if verbose:
                print(pin_)

        # acceleration  eq. constraint
        if not np.isnan(Xdds[i]).all():
            pin_ = {"t": ts[i], "d": 2, "X": Xdds[i]}
            pTraj.addPin(pin_)
            if verbose:
                print(pin_)

        # jerk eq. constraint
        if not np.isnan(Xddds[i]).all():
            pin_ = {"t": ts[i], "d": 3, "X": Xddds[i]}
            pTraj.addPin(pin_)
            if verbose:
                print(pin_)

        # snap eq. constraint
        if not np.isnan(Xdddds[i]).all():
            pin_ = {"t": ts[i], "d": 4, "X": Xdddds[i]}
            pTraj.addPin(pin_)
            if verbose:
                print(pin_)

        # inequality constraints
        ######################## WHY NOT WORKING? WTF?
        # j_lim = 20
        # j_cube = np.array([[-j_lim, j_lim], [-j_lim, j_lim], [-j_lim, j_lim]])
        # pin_ = {"t": ts[i], "d": 3, "X": j_cube}
        # pTraj.addPin(pin_)

        # s_lim = 50
        # s_cube = np.array([[-s_lim, s_lim], [-s_lim, s_lim], [-s_lim, s_lim]])
        # pin_ = {"t": ts[i], "d": 4, "X": s_cube}
        # pTraj.addPin(pin_)
        ########################

    # solve
    pTraj.setDerivativeObj(objWeights)
    # print("solving")
    # time_start = time.time()
    pTraj.solve()
    # time_end = time.time()
    # print(time_end - time_start)

    return pTraj


def scale_time(pTraj, limits, dt, t_start=1, iterate=True):
    """Scale time to satisfy velocity, acceleration and jerk limits"""
    coeffs = cs.SX.sym("c", pTraj.N + 1)  # pTraj.polyCoeffSet[0]
    T = cs.SX.sym("T")
    t = cs.SX.sym("t")

    p = 0
    for i in range(pTraj.N + 1):
        p += coeffs[i] * ((t / T) ** i)

    v = cs.jacobian(p, t)
    a = cs.jacobian(v, t)
    j = cs.jacobian(a, t)
    s = cs.jacobian(j, t)
    f_poly = cs.Function(
        "f_poly",
        [t, T, coeffs],
        [p, v, a, j, s],
        ["t", "T", "coeffs"],
        ["p", "v", "a", "j", "s"],
    )

    T = t_start
    # N = 100
    coeffs = pTraj.polyCoeffSet
    t_evaluation = np.arange(0, 1 + dt, dt)
    N = t_evaluation.shape[0]
    p_eval = np.zeros((3, N))
    v_eval = np.zeros((3, N))
    a_eval = np.zeros((3, N))
    j_eval = np.zeros((3, N))
    s_eval = np.zeros((3, N))
    print("Time scaling polynomial ...")
    while True:
        n_segment = 0
        # t_eval = np.linspace(0, T, N)
        t_eval = t_evaluation * T
        for i in range(t_eval.shape[0]):
            if t_eval[i] > T * pTraj.Ts[n_segment + 1]:
                n_segment += 1
            tt = t_eval[i] - pTraj.Ts[n_segment] * T

            for k in range(3):
                (
                    p_eval[k, i],
                    v_eval[k, i],
                    a_eval[k, i],
                    j_eval[k, i],
                    s_eval[k, i],
                ) = f_poly(tt, T, coeffs[k, :, n_segment])
        if not iterate:
            break
        if (
            np.all(np.abs(v_eval) <= limits["v"])
            and np.all(np.abs(a_eval) <= limits["a"])
            and np.all(np.abs(j_eval) <= limits["j"])
        ):
            break
        else:
            print(T)
            T += 0.2
    print("Done!")

    mJTraj = {
        "p": p_eval,
        "v": v_eval,
        "a": a_eval,
        "j": j_eval,
        "s": s_eval,
        "t": t_eval,
    }

    return mJTraj


def differential_flatness(acc, jerk, snap, yaw=0, additional=False):
    axQ = acc.copy()
    daxQ = jerk.copy()
    d2axQ = snap.copy()

    mQ = 1
    g = 9.81
    e3 = np.array([0, 0, 1])[:, None]

    b1d = np.array(
        [np.cos(yaw), np.sin(yaw), 0]
    )  # TODO: let user know if singularity  (b3 and b1d being parallel)
    db1d = np.zeros((3, 1))
    d2b1d = np.zeros((3, 1))

    fb3 = mQ * (axQ + g * e3)
    norm_fb3 = np.linalg.norm(fb3, axis=0)
    f = norm_fb3
    b3 = fb3 / norm_fb3
    b3_b1d = vec_cross(b3, b1d)
    norm_b3_b1d = np.linalg.norm(b3_b1d, axis=0)
    b1 = -vec_cross(b3, b3_b1d) / norm_b3_b1d
    b2 = vec_cross(b3, b1)
    R = np.array([b1, b2, b3]).transpose(1, 0, 2)

    dfb3 = mQ * daxQ
    dnorm_fb3 = vec_dot(fb3, dfb3) / norm_fb3
    db3 = (dfb3 * norm_fb3 - fb3 * dnorm_fb3) / norm_fb3**2
    db3_b1d = vec_cross(db3, b1d) + vec_cross(b3, db1d)
    dnorm_b3_b1d = vec_dot(b3_b1d, db3_b1d) / norm_b3_b1d
    db1 = (
        -vec_cross(db3, b3_b1d) - vec_cross(b3, db3_b1d) - b1 * dnorm_b3_b1d
    ) / norm_b3_b1d
    db2 = vec_cross(db3, b1) + vec_cross(b3, db1)
    dR = np.array([db1, db2, db3]).transpose(1, 0, 2)
    # R [3 x 3 x N], dR [3 x 3 x N]
    R_T = np.transpose(R, (2, 1, 0))  # [N x 3 x 3]
    dR_T = np.transpose(dR, (2, 0, 1))  # [N x 3 x 3]
    # R.T @ dR using batch matrix multiplication
    R_T_dR = np.einsum("ijk,ikl->ijl", R_T, dR_T)  # [N x 3 x 3]
    R_T_dR_T = np.transpose(R_T_dR, (1, 2, 0))  # [3 x 3 x N]
    Omega = vee(R_T_dR_T)

    d2fb3 = mQ * d2axQ
    d2norm_fb3 = (
        vec_dot(dfb3, dfb3) + vec_dot(fb3, d2fb3) - dnorm_fb3 * dnorm_fb3
    ) / norm_fb3
    d2b3 = (
        (d2fb3 * norm_fb3 + dfb3 * dnorm_fb3 - dfb3 * dnorm_fb3 - fb3 * d2norm_fb3)
        * norm_fb3**2
        - db3 * norm_fb3**2 * 2 * norm_fb3 * dnorm_fb3
    ) / norm_fb3**4
    d2b3_b1d = (
        vec_cross(d2b3, b1d)
        + vec_cross(db3, db1d)
        + vec_cross(db3, db1d)
        + vec_cross(b3, d2b1d)
    )
    d2norm_b3_b1d = (
        (vec_dot(db3_b1d, db3_b1d) + vec_dot(b3_b1d, d2b3_b1d)) * norm_b3_b1d
        - vec_dot(b3_b1d, db3_b1d) * dnorm_b3_b1d
    ) / norm_b3_b1d**2
    d2b1 = (
        (
            -vec_cross(d2b3, b3_b1d)
            - vec_cross(db3, db3_b1d)
            - vec_cross(db3, db3_b1d)
            - vec_cross(b3, d2b3_b1d)
            - db1 * dnorm_b3_b1d
            - b1 * d2norm_b3_b1d
        )
        * norm_b3_b1d
        - db1 * norm_b3_b1d * dnorm_b3_b1d
    ) / norm_b3_b1d**2
    d2b2 = (
        vec_cross(d2b3, b1)
        + vec_cross(db3, db1)
        + vec_cross(db3, db1)
        + vec_cross(b3, d2b1)
    )
    d2R = np.array([d2b1, d2b2, d2b3]).transpose(1, 0, 2)
    dR_T = np.transpose(dR, (2, 1, 0))  # [N x 3 x 3]
    dR_T_ = np.transpose(dR, (2, 0, 1))  # [N x 3 x 3]
    dR_T_dR = np.einsum("ijk,ikl->ijl", dR_T, dR_T_)  # [N x 3 x 3]
    d2R_T = np.transpose(d2R, (2, 0, 1))  # [N x 3 x 3]
    R_T_d2R = np.einsum("ijk,ikl->ijl", R_T, d2R_T)  # [N x 3 x 3]
    sum_ = (dR_T_dR + R_T_d2R).transpose(1, 2, 0)  # [3 x 3 x N]
    dOmega = vee(sum_)

    # R = R.transpose(1, 0, 2)
    if additional:
        return R, Omega, dOmega, daxQ, d2axQ
    return R, Omega, dOmega


def visualize_mJTraj(
    Xs, mJTraj, time_scale, Xds=None, Xdds=None, Xddds=None, Xdddds=None, t_end=None
):
    p_eval = mJTraj["p"]
    v_eval = mJTraj["v"]
    a_eval = mJTraj["a"]
    j_eval = mJTraj["j"]
    s_eval = mJTraj["s"]
    t_eval = mJTraj["t"]
    omega_eval = mJTraj["ang_vel"]
    domega_eval = mJTraj["ang_acc"]
    if t_end is None:
        knots = np.linspace(0, 1, Xs.shape[0]) * t_eval[-1]
    else:
        knots = np.linspace(0, 1, Xs.shape[0]) * t_end

    fig = plt.figure()
    for k in range(3):
        letter = ["x", "y", "z"][k]

        ax = fig.add_subplot(3, 7, 7 * k + 1)
        ax.plot(t_eval, p_eval[k, :], label="p")
        ax.plot(knots, Xs[:, k], "ro")
        ax.set_ylabel("p" + letter)

        ax = fig.add_subplot(3, 7, 7 * k + 2)
        ax.plot(t_eval, v_eval[k, :], label="v")
        if time_scale:
            ax.plot([t_eval[0], t_end], [panda_limits["v"], panda_limits["v"]], "k--")
            ax.plot([t_eval[0], t_end], [-panda_limits["v"], -panda_limits["v"]], "k--")
        if Xds is not None:
            ax.plot(knots, Xds[:, k], "ro")
        else:
            ax.plot([t_eval[0], t_end], [0, 0], "ro")
        ax.set_ylabel("v" + letter)

        ax = fig.add_subplot(3, 7, 7 * k + 3)
        ax.plot(t_eval, a_eval[k, :], label="a")
        if time_scale:
            ax.plot([t_eval[0], t_end], [panda_limits["a"], panda_limits["a"]], "k--")
            ax.plot([t_eval[0], t_end], [-panda_limits["a"], -panda_limits["a"]], "k--")
        # ax.plot([t_eval[0], t_end], [0, 0], "ro")
        if Xdds is not None:
            ax.plot(knots, Xdds[:, k], "ro")
        else:
            ax.plot([t_eval[0], t_end], [0, 0], "ro")
        ax.set_ylabel("a" + letter)

        ax = fig.add_subplot(3, 7, 7 * k + 4)
        ax.plot(t_eval, j_eval[k, :], label="j")
        if Xddds is not None:
            ax.plot(knots, Xddds[:, k], "ro")
        else:
            ax.plot([t_eval[0], t_end], [0, 0], "ro")
        ax.set_ylabel("j" + letter)

        ax = fig.add_subplot(3, 7, 7 * k + 5)
        ax.plot(t_eval, s_eval[k, :], label="s")
        if Xdddds is None:
            ax.plot([t_eval[0], t_end], [0, 0], "ro")
        else:
            ax.plot(knots, Xdddds[:, k], "ro")

        ax = fig.add_subplot(3, 7, 7 * k + 6)
        ax.plot(t_eval, omega_eval[k, :])
        ax.set_ylabel(r"$\omega$" + letter)

        ax = fig.add_subplot(3, 7, 7 * k + 7)
        ax.plot(t_eval, domega_eval[k, :])
        ax.set_ylabel(r"$\dot{\omega}$" + letter)

    plt.suptitle("Task space")
    # ---------------------- Compute slosh-free orientation ---------------------- #
    # g = np.array([0, 0, 9.81])
    # b3 =  np.squeeze([(acc + g) / np.linalg.norm(acc + g) for acc in a_eval.T])
    b3 = mJTraj["R"][:, 2, :].T

    # # ---------------------------------------------------------------------------- #
    # # Visualize
    v_n2 = np.linalg.norm(v_eval, axis=0)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        p_eval[0],
        p_eval[1],
        p_eval[2],
        c=v_n2,
        cmap=cm.turbo,
        marker=".",
        vmin=min(v_n2),
        vmax=max(v_n2),
    )
    axis_equal(p_eval[0], p_eval[1], p_eval[2], ax=ax)

    scale = 0.1
    for k in range(len(t_eval)):
        ax.plot(
            [p_eval[0, k], p_eval[0, k] + scale * b3[k, 0]],
            [p_eval[1, k], p_eval[1, k] + scale * b3[k, 1]],
            [p_eval[2, k], p_eval[2, k] + scale * b3[k, 2]],
            "b",
        )

    ax.plot(Xs[1:-1, 0], Xs[1:-1, 1], Xs[1:-1, 2], "ko", alpha=0.5)
    ax.plot(Xs[0, 0], Xs[0, 1], Xs[0, 2], "go", alpha=0.5)
    ax.plot(Xs[-1, 0], Xs[-1, 1], Xs[-1, 2], "ro", alpha=0.5)
    plt.suptitle(
        "Minimum Jerk - Slosh free --> Nav time: " + str(round(t_eval[-1], 4)) + "s"
    )
    # plt.show()

    return ax


def min_jerk_trajectory(
    Xs,
    limits,
    order,
    dt,
    yaw,
    t_start=1,
    visualize=False,
    time_scale=True,
    Xds=None,
    Xdds=None,
    Xddds=None,
    Xdddds=None,
    t_scaling=True,
    knot_end=1,
    fast=False,
):
    """
    Generates a minimum jerk trajectory, that interpolates points Xs and fulfills
    velocity, acceleration and jerk limits
    """

    # Solve QP by minimizing Jerk
    pTraj = min_jerk_qp(
        Xs=Xs,
        Xds=Xds,
        Xdds=Xdds,
        Xddds=Xddds,
        Xdddds=Xdddds,
        order=order,
        knot_end=knot_end,
    )

    # Time scaling
    if t_scaling:
        mJTraj = scale_time(
            pTraj=pTraj, limits=limits, t_start=t_start, dt=dt, iterate=time_scale
        )
    else:
        t_evaluation = np.arange(0, knot_end + dt, dt)
        if fast:
            t_evaluation = t_evaluation[:10]
        mJTraj = {
            "t": t_evaluation,
            "p": pTraj.eval(t_evaluation, 0),
            "v": pTraj.eval(t_evaluation, 1),
            "a": pTraj.eval(t_evaluation, 2),
            "j": pTraj.eval(t_evaluation, 3),
            "s": pTraj.eval(t_evaluation, 4),
        }

    # Differential flatness
    mJTraj["R"], mJTraj["ang_vel"], mJTraj["ang_acc"] = differential_flatness(
        acc=mJTraj["a"], jerk=mJTraj["j"], snap=mJTraj["s"], yaw=yaw
    )

    # Visualize
    if visualize:
        if fast:
            t_end = knot_end
        else:
            t_end = None
        ax = visualize_mJTraj(
            mJTraj=mJTraj,
            Xs=Xs,
            Xds=Xds,
            Xdds=Xdds,
            Xddds=Xddds,
            Xdddds=Xdddds,
            time_scale=time_scale,
            t_end=t_end,
        )
        return mJTraj, ax
    else:
        return mJTraj, None


# ---------------------------------------------------------------------------- #
#                             Safety check for diff                            #
# ---------------------------------------------------------------------------- #

###############################################
# # verify that rotation matrix is correct
# R = mJTraj["R"]
# p = mJTraj["p"]
# ax = plot_frames(
#     e1=R[:, 0, :].T,
#     e2=R[:, 1, :].T,
#     e3=R[:, 2, :].T,
#     r=p.T,
#     scale=0.1,
#     interval=0.99,
#     ax_equal=True,
# )

# # verify that angular velocity and acceleration are correct
# ang_vel = np.zeros((3, len(mJTraj["t"])))
# ang_vel[:, 0] = mJTraj["ang_vel"][:, 0]
# for k in range(1, len(mJTraj["t"])):
#     ang_vel[:, k] = ang_vel[:, k - 1] + f_rd(mJTraj["t"][k]) * (
#         mJTraj["t"][k] - mJTraj["t"][k - 1]
#     )

# R = np.zeros((3, 3, len(mJTraj["t"])))
# R[:, :, 0] = mJTraj["R"][:, :, 0]
# for k in range(1, len(mJTraj["t"])):
#     wd = f_wd(mJTraj["t"][k])
#     # wd = np.array([-wd[-2], wd[-1], wd[-3]])  # TODO!!!!!

#     R_dot = R[:, :, k - 1] @ hat(wd)
#     R[:, :, k] = R[:, :, k - 1] + R_dot * (mJTraj["t"][k] - mJTraj["t"][k - 1])

# # set matplotlib colors to red green blue
# plt.rcParams["axes.prop_cycle"] = plt.cycler(color=["r", "g", "b"])
# plt.figure()
# plt.plot(mJTraj["t"], ang_vel.T)
# plt.plot(mJTraj["t"], mJTraj["ang_vel"].T, "--")

# plt.figure()
# plt.subplot(311)
# plt.plot(mJTraj["t"], R[:, 0, :].T)
# plt.plot(mJTraj["t"], mJTraj["R"][:, 0, :].T, "--")

# plt.subplot(312)
# plt.plot(mJTraj["t"], R[:, 1, :].T)
# plt.plot(mJTraj["t"], mJTraj["R"][:, 1, :].T, "--")

# plt.subplot(313)
# plt.plot(mJTraj["t"], R[:, 2, :].T)
# plt.plot(mJTraj["t"], mJTraj["R"][:, 2, :].T, "--")

# plt.show()


# exit()
###############################################
