"""
Source: https://frankaemika.github.io/docs/control_parameters.html
"""

import numpy as np

q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
q_max = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
qd_max = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])
qdd_max = np.array([15.0000, 7.5000, 10.0000, 12.5000, 15.0000, 20.0000, 20.0000])
qddd_max = np.array([7500, 3750, 5000, 6250, 7500, 10000, 10000])
tau_max = np.array([87.0000, 87.0000, 87.0000, 87.0000, 12.0000, 12.0000, 12.0000])
taud_max = np.array([1000, 1000, 1000, 1000, 1000, 1000, 1000])

safety = 1
v_max = 1.7 * safety
a_max = 13 * safety
j_max = 6500 * safety

panda_limits = {
    "q_min": q_min,
    "q_max": q_max,
    "qd_max": qd_max,
    "qdd_max": qdd_max,
    "qddd_max": qddd_max,
    "tau_max": tau_max,
    "v": v_max,
    "a": a_max,
    "j": j_max,
}


def panda_ff_ode_v(x, u, robot):
    # extract states and inputs
    q = x[:7]
    vd = u

    # jacobian pseudo inverse
    J = robot.jacob0(q)
    J_pinv = np.linalg.pinv(J)

    # compute joint velocities
    qd = J_pinv @ vd

    # fit into state derivative
    x_dot = qd

    return x_dot


def panda_ff_ode_a(x, u, robot):
    # extract states and inputs
    q = x[:7]
    qd = x[7:]
    ad = u

    # jacobian, pseudo inverse and time derivative
    J = robot.jacob0(q)
    J_pinv = np.linalg.pinv(J)
    H = robot.hessian0(q)
    J_dot = np.tensordot(H, qd, axes=(0, 0))  # H[0] * qd[0] + ... +  H[n] * qd[n]

    qdd = J_pinv @ (ad - J_dot @ qd)

    x_dot = np.hstack((qd, qdd))
    return x_dot


def RK4(x, u, dt, f, robot):
    k1 = f(x=x, u=u, robot=robot)
    k2 = f(x=x + dt / 2 * k1, u=u, robot=robot)
    k3 = f(x=x + dt / 2 * k2, u=u, robot=robot)
    k4 = f(x=x + dt * k3, u=u, robot=robot)
    x_next = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return x_next


def euler(x, u, dt, f, robot):
    x_next = x + dt * f(x=x, u=u, robot=robot)
    return x_next


def panda_integrate(x, u, dt, method, robot):
    if x.shape[0] == 7:
        f = panda_ff_ode_v
    elif x.shape[0] == 14:
        f = panda_ff_ode_a

    if method == "euler":
        x_next = euler(x=x, u=u, dt=dt, f=f, robot=robot)
    elif method == "RK4":
        x_next = RK4(x=x, u=u, dt=dt, f=f, robot=robot)
    return x_next
