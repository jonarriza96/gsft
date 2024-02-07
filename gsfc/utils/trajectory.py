import numpy as np
import rosbag
import casadi as ca
from scipy.signal import butter, lfilter, freqz

from .rotations import rotation_to_quaternion, quaternion_to_rotation


def save_trajectory(traj, t, filename, params):
    mQ = params["mQ"]
    e3 = params["e3"]
    g = params["g"]
    J = params["J"]
    J_inv = params["J_inv"]

    N = traj.shape[1]

    xQ = traj[0:3, :]
    vQ = traj[3:6, :]
    R = traj[6:15, :].reshape((3, 3, N)).transpose((1, 0, 2))
    Omega = traj[15:18, :]
    f = traj[18, :]
    M = traj[19:22, :]
    t_ = t[:N]

    a_lin = np.zeros((3, N))
    Omega_dot = np.zeros((3, N))
    q = np.zeros((4, N))
    for i in range(N):
        q[:, i] = rotation_to_quaternion(R[:, :, i])
        a_lin[:, i : i + 1] = 1 / mQ * (f[i] * R[:, :, i] @ e3 - mQ * g * e3)
        Omega_dot[:, i] = J_inv @ (M[:, i] - np.cross(Omega[:, i], J @ Omega[:, i]))

    u = np.ones((4, N)) * f / 4
    jerk = np.zeros((3, N))
    snap = np.zeros((3, N))

    # t,p_x,p_y,p_z,q_w,q_x,q_y,q_z,v_x,v_y,v_z,w_x,w_y,w_z,a_lin_x,a_lin_y,a_lin_z,a_rot_x,a_rot_y,a_rot_z,u_1,u_2,u_3,u_4,jerk_x,jerk_y,jerk_z,snap_x,snap_y,snap_z
    data = np.concatenate(
        (t_.reshape((1, -1)), xQ, q, vQ, Omega, a_lin, Omega_dot, u, jerk, snap), axis=0
    ).T

    # add header
    header = "t,p_x,p_y,p_z,q_w,q_x,q_y,q_z,v_x,v_y,v_z,w_x,w_y,w_z,a_lin_x,a_lin_y,a_lin_z,a_rot_x,a_rot_y,a_rot_z,u_1,u_2,u_3,u_4,jerk_x,jerk_y,jerk_z,snap_x,snap_y,snap_z"
    np.savetxt(filename, data, delimiter=",", header=header, comments="")


def rosbag_to_trajectory(filename):
    bag = rosbag.Bag(filename)

    # load odometry data
    t = []
    traj = []

    for topic, msg, t_ in bag.read_messages(
        topics=["/angrybird/agiros_pilot/odometry"]
        # topics=["/mocap/angrybird"]
    ):
        t.append(msg.header.stamp.to_sec())

        p = np.array(
            [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ]
        )
        ori = np.array(
            [
                msg.pose.pose.orientation.w,
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
            ]
        )

        R = quaternion_to_rotation(ori)

        v = np.array(
            [
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z,
            ]
        )
        omg = np.array(
            [
                msg.twist.twist.angular.x,
                msg.twist.twist.angular.y,
                msg.twist.twist.angular.z,
            ]
        )

        traj.append(np.concatenate((p, v, R.T.reshape((9,)), omg)))

    traj = np.squeeze(traj).T
    t = np.squeeze(t)

    acceleration = []
    t_acc = []
    for topic, msg, t_ in bag.read_messages(topics=["/angrybird/agiros_pilot/state"]):
        t_acc.append(msg.header.stamp.to_sec())

        acc = np.array(
            [
                msg.acceleration.linear.x,
                msg.acceleration.linear.y,
                msg.acceleration.linear.z,
                msg.acceleration.angular.x,
                msg.acceleration.angular.y,
                msg.acceleration.angular.z,
            ]
        )

        acceleration.append(acc)

    acceleration = np.squeeze(acceleration).T
    t_acc = np.squeeze(t_acc)

    # load reference data
    t_ref = []
    traj_ref = []

    for topic, msg, t_ in bag.read_messages(
        topics=["/angrybird/agiros_pilot/telemetry"]
    ):
        t_ref.append(msg.header.stamp.to_sec())

        p = np.array(
            [
                msg.reference.pose.position.x,
                msg.reference.pose.position.y,
                msg.reference.pose.position.z,
            ]
        )

        v = np.array(
            [
                msg.reference.velocity.linear.x,
                msg.reference.velocity.linear.y,
                msg.reference.velocity.linear.z,
            ]
        )

        ori = np.array(
            [
                msg.reference.pose.orientation.w,
                msg.reference.pose.orientation.x,
                msg.reference.pose.orientation.y,
                msg.reference.pose.orientation.z,
            ]
        )

        R = quaternion_to_rotation(ori)

        omg = np.array(
            [
                msg.reference.velocity.angular.x,
                msg.reference.velocity.angular.y,
                msg.reference.velocity.angular.z,
            ]
        )

        traj_ref.append(np.concatenate((p, v, R.T.reshape((9,)), omg)))

    traj_ref = np.squeeze(traj_ref).T
    t_ref = np.squeeze(t_ref)

    return (t, traj), (t_ref, traj_ref), (t_acc, acceleration)


def interpolate_function(t, x):
    interp = "linear"
    f_interp_x = ca.interpolant("x", interp, [t], x[0, :])
    f_interp_y = ca.interpolant("y", interp, [t], x[1, :])
    f_interp_z = ca.interpolant("z", interp, [t], x[2, :])

    t_ca = ca.MX.sym("t_ca", 1)
    f_interp = ca.Function(
        "f_interp",
        [t_ca],
        [ca.vertcat(f_interp_x(t_ca), f_interp_y(t_ca), f_interp_z(t_ca))],
    )

    return f_interp


def finite_diff(t, x):
    x_dot = np.zeros(x.shape)
    x_dot[:, 0] = (x[:, 1] - x[:, 0]) / (t[1] - t[0])
    x_dot[:, -1] = (x[:, -1] - x[:, -2]) / (t[-1] - t[-2])
    for i in range(1, x.shape[1] - 1):
        x_dot[:, i] = (x[:, i + 1] - x[:, i - 1]) / (t[i + 1] - t[i - 1])
    return x_dot


def synchronize(t, x, t_ref):
    f = interpolate_function(t, x)
    f_map = f.map(t_ref.size, "openmp")
    x_ref = f_map(t_ref).full()
    return x_ref


def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, btype="low", fs=fs)


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_delay(cutoff, fs, order=5):
    b, a = butter(order, cutoff, btype="low", fs=fs)
    w, h = freqz(b, a, worN=2000)
    return w, h


def filter(t, x, shift=10):
    dt = np.mean(np.diff(t))
    cutoff = 0.515 / shift / dt
    print("Cutoff frequency:", cutoff)
    x_ = np.zeros(x.shape)
    for i in range(x.shape[0]):
        x_[i, :] = butter_lowpass_filter(
            x[i, :], cutoff, 1.0 / np.mean(np.diff(t)), order=5
        )
    x_[:, :-shift] = x_[:, shift:]
    return x_
