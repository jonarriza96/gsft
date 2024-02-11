import numpy as np
import casadi as cs
from scipy.interpolate import CubicSpline
from scipy.spatial import distance
import matplotlib.pyplot as plt
from matplotlib import cm

import math
import argparse
import time
import pickle
from sys import exit

import roboticstoolbox as rtb
import swift
import spatialmath as sm
import spatialgeometry as sg
import qpsolvers as qp

from gsft.trajectory_generation.polynomial_optimization import (
    min_jerk_trajectory,
    min_jerk_qp,
    differential_flatness,
)
from gsft.control.task_space_control import geometric_control_se3

from gsft.control.joint_space_control import QP, QP_v, QP_a
from gsft.control.task_space_control import acc_p_servo

from gsft.utils.trajectory_generator import backflip, lissajous, straight_line
from gsft.utils.panda import panda_limits, panda_integrate
from gsft.utils.visualize import axis_equal, plot_frames
from gsft.utils.rotations import hat, vee
from gsft.utils.utils_paper import (
    paper_figure_visualization,
    video_scene_visualization,
    save_data_paper,
    save_data_experiments,
)


def get_desired_ee_pose(t, t_end, pd, vd, ad, jd, sd, yaw, case, offset=None):
    ad = ad[:, None]
    jd = jd[:, None]
    sd = sd[:, None]
    if offset is not None:
        pd += offset

    R, _, _ = differential_flatness(acc=ad, jerk=jd, snap=sd, yaw=yaw)

    R = np.squeeze(R)

    if case == "l":
        e1 = R[:, 0]
        e2 = R[:, 1]
        e3 = R[:, 2]
        R = np.vstack([e3, -e2, e1]).T

    Ted = np.array(
        [
            [R[0, 0], R[0, 1], R[0, 2], pd[0]],
            [R[1, 0], R[1, 1], R[1, 2], pd[1]],
            [R[2, 0], R[2, 1], R[2, 2], pd[2]],
            [0, 0, 0, 1],
        ]
    )  # * sm.SE3.Rz(-np.pi / 2)

    return Ted


def gsfcrm_case_study(trajectory, controller, visualization):
    # ---------------------------------------------------------------------------- #
    #                             Reference trajectory                             #
    # ---------------------------------------------------------------------------- #

    geom_offset = np.zeros(3)
    # ----------------------------- Compute load path ---------------------------- #
    if trajectory["case"] == "l":  # lissajous
        t_start = trajectory["t_start"]
        xi_total = 5
        ampl = 0.2
        path = lissajous(
            A=1 * ampl, B=1 * ampl, C=0.75 * ampl, a=1, b=1, c=2, w=2 * np.pi / xi_total
        )
        n_jerk = 10
        geom_offset = np.array([-0.2, 0, 0])
    elif trajectory["case"] == "b":  # backflip
        t_start = trajectory["t_start"]
        xi_total = 0.5
        path = backflip(a=0.25, w=12)
        n_jerk = 10
    elif trajectory["case"] == "s":  # straight line
        t_start = trajectory["t_start"]
        xi_total = -0.5
        path = straight_line(l=xi_total, direction=np.array([0, 1, 0]))
        n_jerk = 2
    elif trajectory["case"] == "wp":  # waypoint
        t_start = 2
        ptraj = np.array([[0, 0, 0], [-0.25, 0.25, -0.25]])
    elif trajectory["case"] == "h":  # helix
        t_start = trajectory["t_start"]
        T = cs.SX.sym("T")
        xi_total = 5
        xLd = cs.vertcat(-0.12 * T, 0.15 * cs.sin(cs.pi * T), 0.15 * cs.cos(cs.pi * T))
        path = {"T": T, "xLd": xLd}
        n_jerk = 10
        geom_offset = np.array([0.3, -0.4, -0.1])

    # Discretaize the path by evaluating it at the given points
    if trajectory["case"] in ["l", "b", "s", "h"]:
        f_trajectory = cs.Function("f_trajectory", [path["T"]], [path["xLd"]])
        xi = np.linspace(0, xi_total, n_jerk)
        ptraj = np.squeeze([f_trajectory(XI).T for XI in xi])

    # ----------------------------- Compute time law ----------------------------- #
    vtraj = np.ones_like(ptraj) * np.nan
    atraj = np.ones_like(ptraj) * np.nan
    jtraj = np.ones_like(ptraj) * np.nan
    straj = np.ones_like(ptraj) * np.nan
    vtraj[0] = np.zeros(3)
    atraj[0] = np.zeros(3)
    jtraj[0] = np.zeros(3)
    straj[0] = np.zeros(3)
    vtraj[-1] = np.zeros(3)
    atraj[-1] = np.zeros(3)
    jtraj[-1] = np.zeros(3)
    straj[-1] = np.zeros(3)

    # Minimum jerk trajectory interpolating sampled points
    mJTraj, _ = min_jerk_trajectory(
        Xs=ptraj,
        Xds=vtraj,
        Xdds=atraj,
        Xddds=jtraj,
        Xdddds=straj,
        order=10,
        dt=trajectory["dt"],
        limits=panda_limits,
        t_start=t_start,
        visualize=visualization["task_space"],
        yaw=trajectory["yaw"],
    )
    dt = mJTraj["t"][1]
    if visualization["task_space"]:
        plt.show()
    # exit()

    # ---------------------------------------------------------------------------- #
    #                                  Simulation                                  #
    # ---------------------------------------------------------------------------- #

    # Initialize robot
    panda = rtb.models.Panda()
    panda.q = panda.qr

    Ted = get_desired_ee_pose(
        t=mJTraj["t"][0].copy(),
        t_end=mJTraj["t"][-1].copy(),
        pd=mJTraj["p"][:, 0].copy() + geom_offset,
        vd=mJTraj["v"][:, 0].copy(),
        ad=mJTraj["a"][:, 0].copy(),
        jd=mJTraj["j"][:, 0].copy(),
        sd=mJTraj["s"][:, 0].copy(),
        yaw=trajectory["yaw"],
        case=trajectory["case"],
    )
    offset = panda.fkine(panda.q).A[:3, 3] - Ted[:3, 3]

    Ted = get_desired_ee_pose(
        t=mJTraj["t"][0].copy(),
        t_end=mJTraj["t"][-1].copy(),
        pd=mJTraj["p"][:, 0].copy(),
        vd=mJTraj["v"][:, 0].copy(),
        ad=mJTraj["a"][:, 0].copy(),
        jd=mJTraj["j"][:, 0].copy(),
        sd=mJTraj["s"][:, 0].copy(),
        yaw=trajectory["yaw"],
        offset=offset.copy(),
        case=trajectory["case"],
    )

    q0, success, _, _, _ = panda.ik_LM(Tep=Ted, q0=panda.q)
    if not success:
        print("IK failed in initial position")
        exit()
    panda.q = q0
    R0 = panda.fkine(q0).A[:3, :3]

    # Simulation loop
    T = [0]
    Q = [panda.q.copy()]
    Qd = [panda.qd.copy()]
    Qdd = [panda.qdd.copy()]
    TEd = [Ted]
    error = []
    m = []
    e_sf = []
    e_sf_socp = []
    Sl = []
    J_cond = []
    Jd_cond = []
    Ad = []
    E = []
    Acc_EE = []
    Sd = []
    Jd = []

    cont = 0
    p_gc = panda.fkine(panda.q).A[:3, 3]
    R_gc = panda.fkine(panda.q).A[:3, :3]
    v_gc = panda.jacob0(panda.q)[:3, :] @ Qd[0]
    av_gc = panda.jacob0(panda.q)[3:, :] @ Qd[0]
    t_loop_start = time.time()
    t_offset = 0  # needed if replanning
    t_original = mJTraj["t"]  # needed if replanning
    p_original = mJTraj["p"] + offset[:, None]  # needed if replanning
    ptraj_original = ptraj.copy() + offset[None, :]
    ind_min_old = 0  # needed if replanning
    failed = False

    #######################################
    check_trajectory = False
    if check_trajectory:
        # panda = rtb.models.Panda()
        # panda.q = panda.qr
        # Create swift instance
        env = swift.Swift()
        env.launch(realtime=True)
        env.set_camera_pose([1.3, 0, 0.6], [0, 0, -0.3])

        # Add panda to the environment
        env.add(panda)
        goal_axes = sg.Axes(length=0.1)
        env.add(goal_axes)

        # Add trajectory axes
        n_axes = 100
        axes_counter = np.linspace(0, 1, n_axes + 1)
        ind_axes = (axes_counter * (len(mJTraj["t"]) - 1)).astype(int)
        for k in range(n_axes):
            goal_axes = sg.Axes(length=0.1)
            Ted = get_desired_ee_pose(
                t=mJTraj["t"][ind_axes[k]].copy(),
                t_end=mJTraj["t"][-1].copy(),
                pd=mJTraj["p"][:, ind_axes[k]].copy(),
                vd=mJTraj["v"][:, ind_axes[k]].copy(),
                ad=mJTraj["a"][:, ind_axes[k]].copy(),
                jd=mJTraj["j"][:, ind_axes[k]].copy(),
                sd=mJTraj["s"][:, ind_axes[k]].copy(),
                yaw=trajectory["yaw"],
                offset=offset.copy(),
                case=trajectory["case"],
            )
            goal_axes.T = Ted
            # goal_axes.T = panda.fkine(Q[k]).A
            env.add(goal_axes)

            env.step(0)
        exit()
    #######################################
    while True:
        # ---------------------------------- Joints ---------------------------------- #

        # Current joint states
        q = Q[-1]
        qd = Qd[-1]
        qdd = Qdd[-1]

        # Jacobian, pseuodinverse, derivative and Hessian of the panda
        J = panda.jacob0(q)
        H = panda.hessian0(q)
        J_pinv = np.linalg.pinv(J)
        J_dot = np.tensordot(H, qd, axes=(0, 0))  # H[0] * qd[0] + ... +  H[n] * qd[n]

        J_cond += [np.linalg.cond(J)]
        Jd_cond += [np.linalg.cond(J_dot)]

        # ---------------------------- End effector state ---------------------------- #

        # Current end-effector pose
        Te = panda.fkine(q).A
        p = Te[:3, 3]
        R = Te[:3, :3]
        vel = J[:3, :] @ qd
        ang_vel = R.T @ (J[3:, :] @ qd)  # TODO: Understand why R.T (transposed)

        # ------------------------ Desired end effector state ------------------------ #
        if trajectory["replan"]:
            pos = panda.fkine(q).A[:3, 3]
            Rot = panda.fkine(q).A[:3, :3]
            vel = J[:3, :] @ qd
            ang_vel = J[3:, :] @ qd
            acc = J_dot[:3, :] @ qd + J[:3, :] @ qdd
            ang_acc = J_dot[3:, :] @ qd + J[3:, :] @ qdd
            pos_ref = mJTraj["p"][:, cont] + offset
            e_pos = np.linalg.norm(pos - pos_ref)

            # print(e_pos)
            if e_pos > 0.05:
                print("Replanning DF")

                # pos_start = mJTraj["p"][:, cont] + offset
                pos_start = pos
                if T[-1] < t_original[-1] / 2:
                    dists = distance.cdist(pos_start[None, :], ptraj_original[:-1, :])
                    ind_min = np.minimum(
                        np.argmin(dists) + 1, ptraj_original.shape[0] - 1
                    )
                else:
                    dists = distance.cdist(pos_start[None, :], ptraj_original[1:, :])
                    ind_min = np.minimum(
                        np.argmin(dists) + 2, ptraj_original.shape[0] - 1
                    )
                ptraj = np.vstack([pos_start, ptraj_original[ind_min:, :]])
                # ptraj = np.vstack([mJTraj["p"][:, cont] + offset, ptraj_original[1:, :]])

                vtraj = np.ones_like(ptraj) * np.nan
                atraj = np.ones_like(ptraj) * np.nan
                jtraj = np.ones_like(ptraj) * np.nan
                straj = np.ones_like(ptraj) * np.nan
                vref = mJTraj["v"][:, cont]
                aref = mJTraj["a"][:, cont]
                jref = mJTraj["j"][:, cont]
                sref = mJTraj["s"][:, cont]
                if trajectory["replan_fb"]:
                    # Rref, omegaref, domegaref, t_dot, t_ddot = differential_flatness(
                    #     acc=aref[:, None],
                    #     jerk=jref[:, None],
                    #     snap=sref[:, None],
                    #     yaw=0,
                    #     additional=True,
                    # )
                    # omegaref = ang_vel.copy()  # np.squeeze(omegaref)
                    # t_dot = np.squeeze(t_dot)
                    # t_ddot = np.squeeze(t_ddot)
                    # z_b = Rot[:, 2]  # np.squeeze(Rref[:3, 2])
                    # jerk = np.cross(ang_vel, acc)  # + t_dot * z_b
                    # snap = (
                    #     np.cross(ang_vel, np.cross(ang_vel, acc))
                    #     + t_ddot * z_b
                    #     + 2 * np.cross(omegaref, t_dot * z_b)
                    #     + np.cross(
                    #         omegaref,
                    #         aref,
                    #     )
                    # )
                    vtraj[0, :] = vel
                    atraj[0, :] = acc
                    jtraj[0, :] = jref
                    straj[0, :] = sref

                else:
                    vtraj[0, :] = vref
                    atraj[0, :] = aref
                    jtraj[0, :] = jref
                    straj[0, :] = sref

                if T[-1] < t_original[-1] / 2:
                    ind_original = np.argmin(
                        distance.cdist(pos[None, :], p_original.T[:-100, :])
                    )
                else:
                    ind_original = np.argmin(distance.cdist(pos[None, :], p_original.T))
                t_replan = t_original[-1] - t_original[ind_original]

                mJTraj, ax = min_jerk_trajectory(
                    Xs=ptraj,
                    Xds=vtraj,
                    Xdds=atraj,
                    Xddds=jtraj,
                    Xdddds=straj,
                    order=10,
                    dt=trajectory["dt"],
                    limits=panda_limits,
                    t_start=t_replan,
                    visualize=1,  # SET TO TRUE TO DEBUG
                    fast=0,  # SET TO FALSE TO DEBUG
                    yaw=trajectory["yaw"],
                    t_scaling=False,
                    knot_end=t_replan,
                )
                if ax is not None:
                    ax.plot(p_original[0, :], p_original[1, :], p_original[2, :], "-m")
                    ax.plot(pos_ref[0], pos_ref[1], pos_ref[2], "*g")
                    ax.plot(pos[0], pos[1], pos[2], "*k")
                    # ax.plot(pos_start[0], pos_start[1], pos_start[2], "*r")
                    plt.show()
                dt = mJTraj["t"][1]
                offset = np.zeros(3)
                cont = 0

                # t_offset = -T[-1]
        # Desired end-effector transform
        Ted = get_desired_ee_pose(
            t=mJTraj["t"][cont],  # T[-1] + t_offset,
            t_end=mJTraj["t"][-1].copy(),
            pd=mJTraj["p"][:, cont].copy(),
            vd=mJTraj["v"][:, cont].copy(),
            ad=mJTraj["a"][:, cont].copy(),
            jd=mJTraj["j"][:, cont].copy(),
            sd=mJTraj["s"][:, cont].copy(),
            yaw=trajectory["yaw"],
            offset=offset.copy(),
            case=trajectory["case"],
        )
        if not trajectory["slosh-free"]:
            Ted[:3, :3] = R0

        # --------------------------- Task space controller -------------------------- #

        # Calculate the desired end-effector motion
        if controller["task_space"] == "ff":  # feed forward
            # rd_world = np.zeros(3)
            jd = mJTraj["j"][:, cont + 1]
            sd = mJTraj["s"][:, cont + 1]
            rd_world = mJTraj["R"][:, :, cont + 1] @ mJTraj["ang_acc"][:, cont + 1]
            ad = np.hstack([mJTraj["a"][:, cont + 1], rd_world])
            ad_panda = ad.copy()

        elif controller["task_space"] == "sr":  # servo
            v = J @ qd
            # if trajectory["replan"] and e_pos > 0.05:
            #     #     vel = mJTraj["v"][:, cont]
            #     #     ang_vel = mJTraj["R"][:, :, cont] @ mJTraj["ang_vel"][:, cont]
            #     #     vd = np.hstack([vel, ang_vel])
            #     rd_world = mJTraj["R"][:, :, cont] @ mJTraj["ang_acc"][:, cont]
            #     ad = np.hstack([mJTraj["a"][:, cont], rd_world])
            #     ad_panda = ad.copy()
            # else:
            #     #   vd = None

            ad, vdd, e, arrived = acc_p_servo(
                Te=Te,
                Tep=Ted,
                v=v,
                vd=None,
                gain=controller["gain_sr"],
                threshold=controller["stop_tol"],
            )
            ad_panda = ad.copy()

        elif controller["task_space"] == "gc":
            # Geometric control
            at_sf, ar_sf, ex, ev, er, eo = geometric_control_se3(
                p=p,
                v=vel,
                R=R,
                av=ang_vel,
                # p=p_gc,
                # v=v_gc,
                # R=R_gc,
                # av=av_gc,
                pd=(mJTraj["p"][:, cont] + offset).copy(),
                vd=mJTraj["v"][:, cont].copy(),
                ad=mJTraj["a"][:, cont].copy(),
                jd=mJTraj["j"][:, cont].copy(),
                sd=mJTraj["s"][:, cont].copy(),
                yawd=trajectory["yaw"],
                yawdotd=0,
                gains=controller["gains_gc"],
            )

            # at_sf = at_sf - np.array([0, 0, 9.81])
            ad_panda = np.hstack(
                [at_sf, ar_sf @ R.T]
            )  # TODO: Understand why R.T (transposed)

            ad = np.hstack([at_sf, ar_sf])

            # Check if arrived
            arrived = False
            if np.linalg.norm(ex) < controller["stop_tol"]:
                arrived = True

            # SE3 errors
            e = np.hstack([ex, ev, er, eo])

        # Jd += [jd]
        # Sd += [sd]
        Ad += [ad]

        # ############### FILTER
        # ad_ee = ad[:3] + np.array([0, 0, 9.81])
        # e3_ee = panda.fkine(q).A[:3, 2]
        # if trajectory["case"] == "l":
        #     e3_ee = panda.fkine(q).A[:3, 0]

        # ang_error = np.arccos(np.dot(ad_ee / np.linalg.norm(ad_ee), e3_ee))
        # ang_error *= 180 / np.pi
        # if ang_error > 3:
        #     print("Filtering desired acceleration")
        #     ad[:3] = np.dot(ad_ee, e3_ee) * e3_ee - np.array([0, 0, 9.81])
        #     ad_panda[:3] = ad[:3].copy()
        # ###############
        # -------------------------- Joint space controller -------------------------- #

        Jm = panda.jacobm(q, axes="rot")  # manipulability matrix

        # Convert desired end-effector motion to joint space
        if controller["joint_space"] == "inv":
            qdd = J_pinv @ (ad - J_dot @ qd)
            sl = np.zeros(6)
        elif controller["joint_space"] == "socp":
            e3_ee_socp = panda.fkine(q).A[:3, 2]
            if trajectory["case"] == "l":
                e3_ee_socp = panda.fkine(q).A[:3, 0]
            sol = SOCP(
                ad=ad_panda,
                e3_ee=e3_ee_socp,
                J=J,
                H=H,
                J_dot=J_dot,
                Jm=Jm,
                q=q,
                qd=qd,
                qdd=qdd,
                dt=dt,
            )
            qdd = sol[:7]
            sl = sol[7:13]
            esf_socp = sol[-1]
        elif controller["joint_space"] == "qp":
            sol = QP(
                ad=ad_panda,
                J=J,
                H=H,
                J_dot=J_dot,
                Jm=Jm,
                q=q,
                qd=qd,
                qdd=qdd,
                dt=dt,
                controller=controller,
                solver="quadprog",
            )
            if sol is None:
                print("QP failed!")
                Ad = Ad[:-1]
                failed = True
                break
            qdd = sol[:7]
            sl = sol[7:13]
        elif controller["joint_space"] == "qp_term":
            acc = J_dot[:3, :] @ qd + J[:3, :] @ qdd
            acc_ee_norm = np.linalg.norm(acc + np.array([0, 0, 9.81]))
            e3_ee = panda.fkine(q).A[:3, 2]
            if trajectory["case"] == "l":
                e3_ee = panda.fkine(q).A[:3, 0]
            sol = QP_term(
                ad=ad_panda,
                J=J,
                H=H,
                J_dot=J_dot,
                Jm=Jm,
                q=q,
                qd=qd,
                qdd=qdd,
                dt=dt,
                e3_ee=e3_ee,
                acc_ee_norm=acc_ee_norm,
                solver="quadprog",
            )
            if sol is None:
                print("QP failed!")
                Ad = Ad[:-1]
                break
            qdd = sol[:7]
            sl = sol[7:13]
            esf_socp = sol[13]
        elif controller["joint_space"] == "qpv":
            if controller["task_space"] == "gc":
                ad_v = ad_panda - np.array([0, 0, 9.81, 0, 0, 0])
                error = np.sum(np.abs(np.hstack((ex, er))))
            else:
                ad_v = ad_panda.copy()
                error = e.copy()

            v = J @ qd
            vd = v + ad_v * dt
            sol = QP_v(
                vd=vd,
                q=q,
                qd=qd,
                e=error,
                J=J,
                Jm=Jm,
                dt=dt,
                solver="quadprog",
            )
            if sol is None:
                print("QP failed!")
                Ad = Ad[:-1]
                break
            qdd = (sol[:7] - qd) / dt
            qd = sol[:7]
            sl = sol[-6:]
        elif controller["joint_space"] == "qpa":
            if controller["task_space"] == "gc":
                ad_v = ad_panda - np.array([0, 0, 9.81, 0, 0, 0])
                error = np.sum(np.abs(np.hstack((ex, er))))
            else:
                ad_v = ad_panda.copy()
                error = e.copy()

            sol = QP_a(
                ad=ad_v,
                q=q,
                qd=qd,
                e=error,
                J=J,
                Jm=Jm,
                J_dot=J_dot,
                dt=dt,
                solver="quadprog",
            )
            if sol is None:
                print("QP failed!")
                Ad = Ad[:-1]
                break
            qdd = sol[7:14]
            sl = sol[-6:]
        # -------------------------- Forward integrate robot ------------------------- #
        q = q + qd * dt
        if controller["joint_space"] != "qpv":
            qd = qd + qdd * dt

        # ------------------------- Benchmarking measurements ------------------------ #

        # Update jacobian and hessians after iteration for benchmarking
        J = panda.jacob0(q)
        H = panda.hessian0(q)
        J_pinv = np.linalg.pinv(J)
        J_dot = np.tensordot(H, qd, axes=(0, 0))  # H[0] * qd[0] + ... +  H[n] * qd[n]

        # Measure manipulability
        axes = [True, True, True, True, True, True]  # all axes
        J0 = J[axes, :]  # only keep the selected axes
        m += [np.sqrt(np.linalg.det(J0 @ J0.T))]  # manipulability of the selected axes

        # Measure slosh-free error
        g = np.array([0, 0, 9.81])
        # acc_ee = J_dot[:3, :] @ Qd[-1] + J[:3, :] @ qdd + g # With this acceleation esf=0
        acc_ee = J_dot[:3, :] @ qd + J[:3, :] @ qdd + g
        e3_ee = panda.fkine(q).A[:3, 2]
        if trajectory["case"] == "l":
            e3_ee = panda.fkine(q).A[:3, 0]

        # v1 = e3_ee
        # v2 = acc_ee / np.linalg.norm(acc_ee)
        # e_sf += [1 - np.dot(v1.T, v2)]
        v1 = e3_ee
        v2 = acc_ee / np.linalg.norm(acc_ee)
        e_sf += [np.dot(v1.T, v2)]
        if (
            controller["joint_space"] == "socp"
            or controller["joint_space"] == "qp_term"
        ):
            # acc_ee_socp = ad_panda[:3] + sl[:3] + np.array([0, 0, 9.81])
            # esf_socp = np.linalg.norm(acc_ee_socp) - np.dot(acc_ee_socp, e3_ee_socp)
            e_sf_socp += [esf_socp]

        # Log error
        # pd = mJTraj["p"][:, cont] + offset
        # vd = np.hstack(
        #     [mJTraj["v"][:, cont], mJTraj["R"][:, :, cont] @ mJTraj["ang_vel"][:, cont]]
        # )
        # ad = np.hstack(
        #     [mJTraj["a"][:, cont], mJTraj["R"][:, :, cont] @ mJTraj["ang_acc"][:, cont]]
        # )

        # Integrate geometric controller
        p_gc = p_gc + v_gc * dt
        R_gc = R_gc + (R @ hat(av_gc)) * dt
        v_gc = v_gc + (ad[:3] + np.array([0, 0, -9.81])) * dt
        av_gc = av_gc + ad[3:] * dt

        epd = p_gc - panda.fkine(q).A[:3, 3]
        evd = np.hstack((v_gc, av_gc)) - J @ qd
        erd = 1 / 2 * vee(R_gc.T @ R - R.T @ R_gc)
        # ead = ad[:3] - (acc_ee)
        ead = ad - (J @ qdd + J_dot @ qd)
        if controller["task_space"] == "gc":
            ead -= np.array([0, 0, 9.81, 0, 0, 0])  # TODO
        error += [np.hstack([epd, evd, erd, ead])]
        if controller["task_space"] != "ff":
            E += [e]

        # ------------------------------- Update time ------------------------------- #
        # Update by dt seconds
        Q += [q]
        Qd += [qd]
        Qdd += [qdd]
        T += [T[-1] + dt]
        TEd += [Ted]
        Acc_EE += [acc_ee]
        if controller["task_space"] != "inv":
            Sl += [sl]

        print("Time:", round(T[-1], 4), "/", mJTraj["t"][-1], "s")

        # Check if simulation is finished
        cont += 1
        if controller["task_space"] == "ff":
            if cont > len(mJTraj["t"]) - 2:
                cont -= 1
                break
        elif controller["task_space"] == "sr" or controller["task_space"] == "gc":
            if T[-1] > mJTraj["t"][-1] and arrived:
                # pass
                cont -= 1
                print("\nMission accomplished!")
                break
            else:
                cont = min(cont, len(mJTraj["t"]) - 2)

        if (
            time.time() - t_loop_start > trajectory["timeout"]
        ):  # time out after 5 seconds
            print("\nSimulation timeout before getting to the end!")
            # print("\tError, Tol. :", np.linalg.norm(E[-1]), controller["stop_tol"])
            break

    T = np.squeeze(T)
    Q = np.squeeze(Q)
    Qd = np.squeeze(Qd)
    Qdd = np.squeeze(Qdd)
    Qddd = np.vstack((np.zeros(7), np.diff(Qdd, axis=0) / dt))
    error = np.squeeze(error)
    J_cond = np.squeeze(J_cond)
    Jd_cond = np.squeeze(Jd_cond)
    e_sf = np.squeeze(e_sf)
    m = np.squeeze(m)
    Sl = np.squeeze(Sl)
    Ad = np.squeeze(Ad)
    E = np.squeeze(E)
    Acc_EE = np.squeeze(Acc_EE)

    esf_deg = np.arccos(e_sf) * 180 / np.pi

    pos = []
    for q in Q:
        pos += [panda.fkine(q).A[:3, 3]]
    pos = np.array(pos)
    pos_ref = mJTraj["p"].T + offset
    eucl_dist = distance.cdist(pos, pos_ref, "euclidean")
    e_pos = np.min(eucl_dist, axis=1)

    Ep_sol = np.trapz(np.abs(e_pos), dx=dt)
    Esf_sol = np.trapz(np.abs(esf_deg), dx=dt)
    MaxEsf_sol = np.max(np.abs(esf_deg))
    Sl_sol = np.sum(np.trapz(np.abs(Sl), dx=dt, axis=0))
    print("\n\tE_p [m * s]:", Ep_sol)
    print("\tE_sf [deg * s]:", Esf_sol)
    print("\tmax(E_sf) [deg]:", MaxEsf_sol)
    print("\tSl", Sl_sol)

    # --------------------------------- Visualize -------------------------------- #

    if controller["task_space"] == "gc":
        plt.figure()
        plt.suptitle("GC errors")
        plt.subplot(411)
        plt.plot(T[1:], E[:, :3])
        plt.ylabel("ep")
        plt.subplot(412)
        plt.plot(T[1:], E[:, 3:6])
        plt.ylabel("ev")
        plt.subplot(413)
        plt.plot(T[1:], E[:, 6:9])
        plt.ylabel("er")
        plt.subplot(414)
        plt.plot(T[1:], E[:, 9:12])
        plt.ylabel("eo")

    if visualization["singularity"]:
        fig = plt.figure()
        plt.subplot(211)
        plt.plot(T[1:], J_cond)
        plt.ylabel("Jacobian cond.")
        plt.subplot(212)
        plt.plot(T[1:], Jd_cond)
        plt.ylabel("Jd cond.")

    if visualization["desired_acc"]:
        fig = plt.figure()
        plt.subplot(411)
        plt.plot(T[1:], Ad[:, :3])
        plt.ylabel(r"$a_d$")
        try:
            plt.subplot(412)
            plt.plot(T[1:], Jd)
            plt.ylabel(r"$j_d$")
            plt.subplot(413)
            plt.plot(T[1:], Sd)
            plt.ylabel(r"$s_d$")
        except:
            pass
        plt.subplot(414)
        plt.plot(T[1:], Ad[:, 3:])
        plt.ylabel(r"$r_d$")

    if visualization["slosh_freeness"]:
        fig = plt.figure()
        plt.subplot(321)
        plt.plot(T[1:], e_sf, label="True")
        plt.ylabel("e_sf")
        if controller["joint_space"] == "socp":
            plt.plot(
                T[1:],
                np.squeeze(e_sf_socp) / np.linalg.norm(Acc_EE, axis=1),
                "--",
                label="OCP",
            )
        if controller["joint_space"] == "qp_term":
            plt.plot(T[1:], np.squeeze(e_sf_socp), "--", label="OCP")
        plt.legend()
        plt.subplot(322)
        plt.plot(T[1:], esf_deg)
        plt.ylabel("sf [deg.]")
        plt.subplot(312)
        plt.plot(T[1:], m)
        plt.ylabel("m")
        if controller["joint_space"] != "inv":
            plt.subplot(325)
            plt.plot(T[1:], Sl[:, :3])
            # plt.plot(T[1:], Sl[:, 0] - Sl[:, 1], "k--")
            plt.ylabel(r"$Sl_t$")
            plt.subplot(326)
            plt.plot(T[1:], Sl[:, 3:])
            plt.ylabel(r"$Sl_r$")

    # fig = plt.figure()
    # plt.suptitle("QP/Integr. errors")
    # ax = plt.subplot(511)
    # ax.plot(T[:-1], error[:, :3])
    # ax.set_ylabel("ep")
    # ax = plt.subplot(512)
    # ax.plot(T[:-1], error[:, 3:6])
    # ax.set_ylabel("ev")
    # ax = plt.subplot(513)
    # ax.plot(T[:-1], error[:, 6:9])
    # ax.set_ylabel("eR")
    # ax = plt.subplot(514)
    # ax.plot(T[:-1], error[:, 9:12])
    # ax.set_ylabel("e_at")
    # ax = plt.subplot(515)
    # ax.plot(T[:-1], error[:, 12:15])
    # ax.set_ylabel("e_ar")

    if visualization["joint_space"]:
        # Joint space
        fig = plt.figure()
        plt.suptitle("Joint space")
        for k in range(7):
            ax = plt.subplot(4, 7, 1 + k)
            ax.plot(T, Q[:, k])
            ax.plot(
                [T[0], T[-1]],
                [panda_limits["q_min"][k], panda_limits["q_min"][k]],
                "k--",
            )
            ax.plot(
                [T[0], T[-1]],
                [panda_limits["q_max"][k], panda_limits["q_max"][k]],
                "k--",
            )
            ax.set_ylabel(r"$q$" + str(k + 1))

            ax = plt.subplot(4, 7, 8 + k)
            ax.plot(T, Qd[:, k])
            ax.plot(
                [T[0], T[-1]],
                [-panda_limits["qd_max"][k], -panda_limits["qd_max"][k]],
                "k--",
            )
            ax.plot(
                [T[0], T[-1]],
                [panda_limits["qd_max"][k], panda_limits["qd_max"][k]],
                "k--",
            )
            ax.set_ylabel(r"$\dot{q}$" + str(k + 1))

            ax = plt.subplot(4, 7, 15 + k)
            ax.plot(T, Qdd[:, k])
            ax.plot(
                [T[0], T[-1]],
                [-panda_limits["qdd_max"][k], -panda_limits["qdd_max"][k]],
                "k--",
            )
            ax.plot(
                [T[0], T[-1]],
                [panda_limits["qdd_max"][k], panda_limits["qdd_max"][k]],
                "k--",
            )
            ax.set_ylabel(r"$\ddot{q}$" + str(k + 1))

            ax = plt.subplot(4, 7, 22 + k)
            ax.plot(T, Qddd[:, k])
            ax.plot(
                [T[0], T[-1]],
                [-panda_limits["qddd_max"][k], -panda_limits["qddd_max"][k]],
                "k--",
            )
            ax.plot(
                [T[0], T[-1]],
                [panda_limits["qddd_max"][k], panda_limits["qddd_max"][k]],
                "k--",
            )
            ax.set_ylabel(r"$\dddot{q}$" + str(k + 1))

        # Task space visualization
        Rd = np.squeeze(TEd)[:, :3, :3].transpose(
            0, 2, 1
        )  # my visualization transposed w.r.t rtb
        rd = np.squeeze(TEd)[:, :3, 3]

        # Compute end effector pose
        R = np.zeros((len(Q), 3, 3))
        r = np.zeros((len(Q), 3))
        for k, q in enumerate(Q):
            Te = panda.fkine(q).A
            R[k, :, :] = Te[:3, :3].T  # my visualization transposed w.r.t rtb
            r[k, :] = Te[:3, 3]

        plt.show()

    # ---------------------------- Rendered simulation --------------------------- #
    if visualization["simulation"]:  # to generate figures with multiple robots
        # Create swift instance
        env = swift.Swift()
        env.launch(realtime=True)
        env.set_camera_pose([1.3, 0, 0.6], [0, 0, -0.3])
        if trajectory["case"] == "b":
            env.set_camera_pose([1.6, 0.5, 0.85], [0, 0, 0.4])
        elif trajectory["case"] == "h":
            env.set_camera_pose([-0.75, 0.85, 0.75], [0, 0, 0.2])
        # Add panda to the environment
        env.add(panda)
        goal_axes = sg.Axes(length=0.1)
        ee_axes = sg.Axes(0.1)
        env.add(goal_axes)
        env.add(ee_axes)

        # Add trajectory axes
        # n_axes = 100
        # axes_counter = np.linspace(0, 1, n_axes + 1)
        # ind_axes = (axes_counter * (len(Q) - 1)).astype(int)
        # for k in range(n_axes):
        #     goal_axes = sg.Axes(length=0.1)
        #     goal_axes.T = TEd[ind_axes[k]]
        #     # goal_axes.T = panda.fkine(Q[k]).A
        #     env.add(goal_axes)

        # Move the robot
        panda.q = Q[0]
        goal_axes.T = TEd[0]
        ee_axes.T = panda.fkine(Q[0]).A
        env.step(0)

        # exit()
        time.sleep(1)
        # print("Lets do some Slosh-Free moves!")
        # Record video
        # env.start_recording(
        #     file_name=str(trajectory["t_start"]) + "_" + trajectory["case"],
        #     framerate=30.0,
        # )
        for k in range(1, Q.shape[0]):
            panda.q = Q[k]
            goal_axes.T = TEd[k]
            ee_axes.T = panda.fkine(Q[k]).A
            env.step(T[k] - T[k - 1])

        # env.stop_recording()1

    return (
        Ep_sol,
        Esf_sol,
        MaxEsf_sol,
        Sl_sol,
        failed,
        Q,
        Qd,
        Qdd,
        Qddd,
        TEd,
        esf_deg,
        T,
    )


if __name__ == "__main__":
    from config import trajectory, controller, visualization

    # -------------------------------- User input -------------------------------- #
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--t_nav",
        type=float,
        default=4.0,
        help="Navigation time",
    )
    parser.add_argument(
        "--case",
        type=str,
        default="l",
        help="l: lissajous, b: backflip",
    )
    parser.add_argument("--nsf", action="store_true", help="Makes it non-sloshfree")
    parser.add_argument(
        "--no_visualization",
        action="store_true",
        help="Does not show data and animation",
    )

    args = parser.parse_args()

    trajectory["case"] = args.case
    trajectory["slosh-free"] = not args.nsf
    trajectory["t_start"] = args.t_nav
    if args.no_visualization:
        for key in visualization:
            visualization[key] = False

    # ------------------------------ Run simulation ------------------------------ #
    (
        Ep,
        Esf,
        MaxEsf,
        Sl,
        failed,
        Q,
        Qd,
        Qdd,
        Qddd,
        TEd,
        esf_deg,
        t_nav,
    ) = gsfcrm_case_study(
        trajectory=trajectory, controller=controller, visualization=visualization
    )
    print(round(Ep, 4), round(Esf, 4), round(MaxEsf, 4), round(Sl, 4))

    # --------------------------- paper and video data --------------------------- #

    # to generate the paper figure
    if False:
        paper_figure_visualization(trajectory=trajectory, Q=Q, TEd=TEd)

    # to visualzie trajectory scence (used for recording video simulations)
    if False:
        video_scene_visualization(trajectory=trajectory, Q=Q, TEd=TEd)

    # save esf deg data for paper
    if False:
        save_data_paper(trajectory, t_nav, esf_deg)

    # save data fof experiments
    if False:
        save_data_experiments(trajectory, t_nav, Q, Qd, Qdd)
