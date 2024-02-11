import numpy as np

# from gsfc.config import controller
from gsft.utils.panda import panda_limits

import qpsolvers as qp

# import cvxpy as cp


def QP(ad, J, H, J_dot, Jm, q, qd, qdd, dt, controller, solver="clarabel", base="0"):
    # x = [qdd, sl, qd, q]

    # The equality contraints
    Aeq = np.zeros(
        (
            6 + 14,
            7 + 6 + 14,
        )
    )  # [n_eq, n_var]
    beq = np.zeros((6 + 0 + 14,))

    # if controller["task_space"] == "gc":
    #     # Scaling --> keeps slosh-free

    #     if base == "0":
    #         Aeq[:6, :7] = J  # qdd
    #         Aeq[:3, 7] = -ad[:3]  # s*at_sf # scaled slosh-free acceleration
    #         Aeq[3:6, 10:13] = np.eye(3)  # remaining slacks
    #         beq[:3] = -J_dot[:3, :] @ qd - np.array([0, 0, 9.81])
    #         beq[3:6] = ad[3:] - J_dot[3:, :] @ qd
    #     elif base == "e":
    #         Aeq[:6, :7] = J  # qdd
    #         Aeq[2, 7] = -ad[2]  # scaled trans. acceleration
    #         Aeq[5, 12] = 1  # free rot. acceleration
    #         beq[:6] = -J_dot @ qd
    #         beq[:2] += ad[:2]
    #         beq[3:6] += ad[3:]
    #     # # Changing vector --> destroys slosh-free
    #     # Aeq[:6, :7] = J  # qdd
    #     # Aeq[:6, 7 : 7 + 6] = np.eye(6)  # slacks
    #     # beq[:6] = ad - J_dot @ qd - np.array([0, 0, 9.81, 0, 0, 0])

    # else:
    Aeq[:6, :7] = J  # qdd
    if controller["slack"]:
        Aeq[:6, 7 : 7 + 6] = np.eye(6)  # slacks
    beq[:6] = ad - J_dot @ qd
    if controller["task_space"] == "gc":
        beq[:3] -= np.array([0, 0, 9.81])

    Aeq[-14:-7, :7] = -dt * np.eye(7)
    Aeq[-14:-7, -14:-7] = np.eye(7)
    beq[-14:-7] = qd  # qd.reshape((7,)

    Aeq[-7:, -14:-7] = -dt * np.eye(7)
    Aeq[-7:, -7:] = np.eye(7)
    beq[-7:] = q + dt * qd  # qd.reshape((7,) #TODO: ???????????????????????

    # if controller["task_space"] == "gc":
    #     slack_Aeqs = np.zeros((6, Aeq.shape[1]))
    #     slack_beqs = np.zeros(6)

    #     # scaling slack
    #     if controller["slack"]:
    #         slack_Aeqs[0, 7] = 1  # s_ty = s_tx - 1 --> stx - sty = 1
    #         slack_Aeqs[0, 8] = -1  # s_ty = s_tx - 1 --> stx - sty = 1
    #         slack_beqs[0] = 1
    #     else:
    #         slack_Aeqs[0, 7] = 1  # s_tx = 1
    #         slack_beqs[0] = 1
    #         slack_Aeqs[5, 8] = 1  # s_ty = 0

    #     slack_Aeqs[1, 9] = 1  # s_tz = 0
    #     slack_Aeqs[2, 10] = 1  # s_rx = 0
    #     slack_Aeqs[3, 11] = 1  # s_ry = 0
    #     if base == "0":
    #         slack_Aeqs[4, 12] = 1  # s_rz = 0

    #     Aeq = np.vstack((Aeq, slack_Aeqs))
    #     beq = np.hstack((beq, slack_beqs))

    # The inequality constraints --> jerk limits
    Ain = np.zeros((14, Aeq.shape[1]))  # [n_ineq, n_var]
    Ain[:7, :7] = np.eye(7)
    Ain[7:, :7] = -np.eye(7)
    qddlim_max = qdd + dt * (panda_limits["qddd_max"])
    qddlim_min = -(qdd + dt * (-panda_limits["qddd_max"]))
    bin = np.hstack([qddlim_max, qddlim_min])
    # Ain = None
    # bin = None

    # Weighting matrices
    Qw = np.eye(7 + 6 + 0 + 14)  # Quadratic component of objective function
    cw = np.zeros((7 + 6 + 0 + 14,))  # Linear component of objective function

    # Linear component of objective function
    cw[-14:-7] = controller["manipulability"] * -Jm.reshape((7,))

    # if controller["task_space"] == "gc":
    #     # Scaling and yaw --> keeps slosh-free

    #     Qw[7, 7] *= 1e-18  # Translational slack --> s_tx
    #     Qw[8, 8] *= 1e0  # Translational slack --> s_ty = s_tx - 1

    #     Qw[12, 12] *= 1e-18  # Rotational yaw slack free --> s_ty = s_tx - 1

    #     # Qw[9, 9] *= 1e-18  # Unused slack --> s_tz
    #     # Qw[10:13, 10:13] *= 1e0  # Rotational slacks --> s_rx, s_ry, s_rz

    #     # Changing vector --> destroys slosh-free
    #     # Qw[7:13, 7:13] *= 1e-2  # Slack component of Q

    # else:
    Qw[7:13, 7:13] *= 1e3  # Slack component of Q

    Qw[:7, :7] *= 1e-8  # qdd
    Qw[-14:-7, -14:-7] *= 1  # qd
    Qw[-7:, -7:] *= 1e-8  # q

    # The lower and upper bounds on the joint velocity
    qdd_lb = -panda_limits["qdd_max"]
    qdd_ub = panda_limits["qdd_max"]
    sl_lb = -1e10 * np.ones((6,))
    sl_ub = 1e10 * np.ones((6,))
    # sl_lb[0] = 0
    # sl_ub[0] = 1
    qd_lb = -panda_limits["qd_max"]
    qd_ub = panda_limits["qd_max"]
    q_lb = panda_limits["q_min"]  # * 1.5
    q_ub = panda_limits["q_max"]  # * 1.5
    lb = np.hstack((qdd_lb, sl_lb, qd_lb, q_lb))
    ub = np.hstack((qdd_ub, sl_ub, qd_ub, q_ub))

    # Calculate the required joint accelerations
    sol = qp.solve_qp(Qw, cw, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver=solver)

    return sol


def damper_constraints(q, ps, pi, gain, q_lims=None):
    """
    Formulates an inequality contraint which, when optimised for will
    make it impossible for the robot to run into joint limits. Requires
    the joint limits of the robot to be specified.

    ps: The minimum angle (in radians) in which the joint is
        allowed to approach to its limit
    pi: The influence angle (in radians) in which the velocity
        damper becomes active
    gain: The gain for the velocity damper

    returns: Ain, Bin as the inequality contraints for an qp
    """

    Ain = np.zeros((7, 7))
    Bin = np.zeros(7)

    if q_lims is None:
        q_lims = np.vstack((panda_limits["q_min"], panda_limits["q_max"]))

    for i in range(7):
        if q[i] - q_lims[0, i] <= pi:
            Bin[i] = -gain * (((q_lims[0, i] - q[i]) + ps) / (pi - ps))
            Ain[i, i] = -1
        if q_lims[1, i] - q[i] <= pi:
            Bin[i] = gain * ((q_lims[1, i] - q[i]) - ps) / (pi - ps)
            Ain[i, i] = 1

    return Ain, Bin


def QP_v(vd, q, qd, e, J, Jm, dt, solver="clarabel"):
    # Set the gain on the manipulability maximisation
    λm = 0.0

    # Set the gain on the joint velocity norm minimisation
    λq = 0.1

    ### Calculate each component of the quadratic programme
    # Quadratic component of objective function
    Q = np.eye(7 + 6)

    # Joint velocity component of Q
    Q[:7, :7] *= λq

    # Slack component of Q
    Q[7:, 7:] = 1e5 * np.eye(6)  # 1/e
    # Q[7:, 7:] = 1 / e * np.eye(6)  # 1/e

    # The equality contraints
    Aeq = np.c_[J, np.eye(6)]
    beq = vd.reshape((6,))

    # The inequality constraints for joint limit avoidance
    Ain = np.zeros((7 + 6, 7 + 6))
    bin = np.zeros(7 + 6)

    # Form the joint limit velocity damper
    ps = 0.05
    pi = 0.9
    k_qlim = 2
    Ain[:7, :7], bin[:7] = damper_constraints(q=q, ps=ps, pi=pi, gain=k_qlim)

    # Linear component of objective function: the manipulability Jacobian
    c = np.r_[λm * -Jm.reshape((7,)), np.zeros(6)]

    # The lower and upper bounds on the joint velocity and slack variable
    sl_lb = -1e10 * np.ones((6,))  # * 0
    sl_ub = 1e10 * np.ones((6,))  # * 0
    qd_lb = -panda_limits["qd_max"]
    qd_ub = panda_limits["qd_max"]
    # qdd_lb = -panda_limits["qdd_max"]  # * 1e10
    # qdd_ub = +panda_limits["qdd_max"]  # * 1e10
    # qd_lbold = qd_lb
    # qd_lbold2 = qd + qdd_lb * dt
    # qd_lb = np.maximum(qd_lb, qd + qdd_lb * dt)
    # qd_ub = np.minimum(qd_ub, qd + qdd_ub * dt)

    # ind_min = np.argwhere(Ain[:7, :7] == -1)[:, 0]
    # ind_max = np.argwhere(Ain[:7, :7] == 1)[:, 0]
    # qd_lb2 = np.zeros((7))
    # qd_ub2 = np.zeros((7))
    # qd_lb2[ind_min] = -bin[:7][ind_min]
    # qd_ub2[ind_max] = bin[:7][ind_max]

    # print("qd_lb:", np.round(qd_lb, 2))
    # print("qd_ub:", np.round(qd_ub, 2))
    # print("qd_lb2:", np.round(qd_lb2, 2))
    # print("qd_ub2:", np.round(qd_ub2, 2))
    # print("qd_lb2 < qd_ub: ", qd_lb2[ind_min] < qd_ub[ind_min])
    # print("qd_lb  < qd_ub2:", qd_lb[ind_max] < qd_ub2[ind_max])
    # print("\n")

    lb = np.r_[qd_lb, sl_lb]
    ub = np.r_[qd_ub, sl_ub]

    # Solve for the joint velocities qd and apply to the robot
    sol = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver=solver)

    return sol


def QP_a(ad, q, qd, e, J, Jm, J_dot, dt, solver="clarabel"):
    λm = 0.0  # manipulability gain
    λqd = 1  # joint velocity norm gain
    λqdd = 1  # joint acceleration norm gain
    λs = 1e5  # slack gain

    ### Calculate each component of the quadratic programme
    # Quadratic component of objective function
    Q = np.eye(7 + 7 + 6)
    Q[:7, :7] *= λqd  # qd
    Q[7:14, 7:14] *= λqdd  # qdd
    Q[-6:, -6:] = λs * np.eye(6)
    # Q[7:, 7:] = 1 / e * np.eye(6)  # 1/e

    # The equality contraints
    Aeq = np.zeros((6 + 7, 7 + 7 + 6))  # [n_eq, n_var]
    beq = np.zeros((6 + 7,))  # [n_eq,]

    Aeq[:6, 7:14] = J  # qdd
    Aeq[:6, -6:] = np.eye(6)  # slacks
    beq[:6] = ad - J_dot @ qd

    Aeq[6:13, 7:14] = -dt * np.eye(7)
    Aeq[6:13, :7] = np.eye(7)
    beq[6:13] = qd  # qd.reshape((7,)

    # The inequality constraints for joint limit avoidance
    Ain = np.zeros((7 + 7 + 6, 7 + 7 + 6))
    bin = np.zeros(7 + 7 + 6)

    # Form the joint limit velocity damper
    ps = 0.05
    pi = 0.9
    k_qlim = 2
    q_lims = np.vstack((panda_limits["q_min"], panda_limits["q_max"]))
    qd_lims = np.vstack((-panda_limits["qd_max"], panda_limits["qd_max"]))
    Ain[:7, :7], bin[:7] = damper_constraints(
        q=q, ps=ps, pi=pi, gain=k_qlim, q_lims=q_lims
    )
    # Ain[7:14, 7:14], bin[7:14] = damper_constraints(
    #     q=qd, ps=0.05, pi=0.1, gain=0.5, q_lims=qd_lims
    # )

    # Linear component of objective function: the manipulability Jacobian
    c = np.r_[λm * -Jm.reshape((7,)), np.zeros(7), np.zeros(6)]

    # The lower and upper bounds on the joint velocity and slack variable
    sl_lb = -1e10000 * np.ones((6,))  # * 0
    sl_ub = 1e10000 * np.ones((6,))  # * 0
    qd_lb = -panda_limits["qd_max"]  # * 1e1000
    qd_ub = +panda_limits["qd_max"]  # * 1e1000
    qdd_lb = -panda_limits["qdd_max"] * 1e10
    qdd_ub = +panda_limits["qdd_max"] * 1e10
    lb = np.r_[qd_lb, qdd_lb, sl_lb]
    ub = np.r_[qd_ub, qdd_ub, sl_ub]

    # Solve for the joint velocities qd and apply to the robot
    sol = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver=solver)

    return sol
