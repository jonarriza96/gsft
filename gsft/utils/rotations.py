import numpy as np
from scipy.spatial.transform import Rotation


def closest_to_A_perpendicular_to_B(A, B):
    """Returns closest vector to A which is perpendicular to B"""
    return A - B * np.dot(B, A)


def tangent_to_rotation(e3, e1_des=np.array([1, 0, 0])):
    """
    Converts tangent vector to a possible rotation matrix. Given that infinite
    rotations are eligible, we choose the one whose e1 is closest to e1_des
    Args:
        e3: tangent vector [e3x,e3y,e3z]
        e1_des: desired e1

    Returns:
        R: Rotation matrix [e1,e2,e3], whose first component is e1 and third
           component is closest to e3_des


    """
    e3 = e3 / np.linalg.norm(e3)
    if (e3 == e1_des).all():
        e1 = np.array([0, 0, 1])
    else:
        e1 = closest_to_A_perpendicular_to_B(e1_des, e3)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(e3, e1)
    R = np.vstack([e1, e2, e3]).T
    return R


def hat(x):
    return np.array(
        [
            [0.0, -float(x[2]), float(x[1])],
            [float(x[2]), 0.0, -float(x[0])],
            [-float(x[1]), float(x[0]), 0.0],
        ],
        dtype=np.float64,
    )


def vee(matrix):
    return np.array([matrix[2, 1], matrix[0, 2], matrix[1, 0]])


def att_error(xQ, R, r, Omega, vQ, f, mQ, mC, g, e3, M, J, J_inv):
    xC = xQ + R @ r
    R_dot = R @ hat(Omega)
    xC_dot = vQ + R_dot @ r
    aQ = 1 / mQ * (f * (R @ e3) - mQ * g * e3)
    Omega_dot = J_inv @ (M - hat(Omega) @ J @ Omega)
    aC = aQ + R @ hat(Omega_dot) @ r + R @ hat(Omega) @ hat(Omega) @ r

    fb3_C = mC * (aC + g * e3)
    fb3_Q = mQ * (aQ + g * e3)
    b3_C = fb3_C / np.linalg.norm(fb3_C)
    b3_Q = fb3_Q / np.linalg.norm(fb3_Q)

    att_err = np.arccos(b3_C.T @ b3_Q)
    if att_err > np.pi / 2:
        att_err = np.pi - att_err
    return att_err


def attitude_error_from_accelerations(acc, R, Omega, Omega_dot, params):
    r = params["r"]
    g = params["g"]
    e3 = params["e3"]
    N = acc.shape[1]
    att_err = np.zeros(N)
    for i in range(N):
        aQ = acc[:, i : i + 1]
        aC = (
            aQ
            + R[:, :, i] @ hat(Omega_dot[:, i]) @ r
            + R[:, :, i] @ hat(Omega[:, i]) @ hat(Omega[:, i]) @ r
        )
        fb3_C = aC + g * e3
        fb3_Q = aQ + g * e3
        b3_C = fb3_C / np.linalg.norm(fb3_C)
        b3_Q = fb3_Q / np.linalg.norm(fb3_Q)
        att_err[i] = np.arccos(b3_C.T @ b3_Q)
    return att_err * 180 / np.pi


def attitude_error_from_attitude(acc, R, Omega, Omega_dot, params):
    r = params["r"]
    g = params["g"]
    e3 = params["e3"]
    N = acc.shape[1]
    att_err = np.zeros(N)
    for i in range(N):
        aQ = acc[:, i : i + 1]
        aC = (
            aQ
            + R[:, :, i] @ hat(Omega_dot[:, i]) @ r
            + R[:, :, i] @ hat(Omega[:, i]) @ hat(Omega[:, i]) @ r
        )
        fb3_C = aC + g * e3
        fb3_Q = R[:, 2, i]
        b3_C = fb3_C / np.linalg.norm(fb3_C)
        b3_Q = fb3_Q / np.linalg.norm(fb3_Q)
        att_err[i] = np.arccos(b3_C.T @ b3_Q)
    return att_err * 180 / np.pi


def get_attitude_components(acc, R, Omega, Omega_dot, params):
    r = params["r"]
    g = params["g"]
    e3 = params["e3"]
    N = acc.shape[1]
    att_C = np.zeros((3, N))
    att_C_acc = np.zeros((3, N))
    att_C_omega = np.zeros((3, N))
    att_C_omega_dot = np.zeros((3, N))
    att_Q = np.zeros((3, N))
    for i in range(N):
        aQ = acc[:, i : i + 1]
        aC_acc = aQ + g * e3
        aC_omega = R[:, :, i] @ hat(Omega[:, i]) @ hat(Omega[:, i]) @ r
        aC_omega_dot = R[:, :, i] @ hat(Omega_dot[:, i]) @ r
        fb3_C = aC_acc + aC_omega + aC_omega_dot
        fb3_C_norm = np.linalg.norm(fb3_C)
        b3_C = fb3_C / fb3_C_norm
        fb3_Q = R[:, 2, i]
        b3_Q = fb3_Q / np.linalg.norm(fb3_Q)
        att_C[:, i : i + 1] = b3_C
        att_C_acc[:, i : i + 1] = aC_acc / fb3_C_norm
        att_C_omega[:, i : i + 1] = aC_omega / fb3_C_norm
        att_C_omega_dot[:, i : i + 1] = aC_omega_dot / fb3_C_norm
        att_Q[:, i] = b3_Q
    return att_C, att_Q, att_C_acc, att_C_omega, att_C_omega_dot


def rotation_to_quaternion(R, swap=False):
    """
    Converts rotation matrix to quaterion

    Args:
        R: [e1,e2,e3]

    Returns:
        q: quaternion [qw, qx, qy, qz]
    """
    q = Rotation.from_matrix(R).as_quat()
    if swap is False:
        q = np.concatenate([[q[-1]], q[:-1]])  # [x,y,z,w] --> [w,x,y,z]
    return q


def quaternion_to_rotation(q, swap=False):
    """
    Converts quaternion to rotation matrix

    Args:
        q: quaternion [qw, qx, qy, qz]

    Returns:
        R: [e1,e2,e3]
    """
    if np.linalg.norm(q) < 1e-6:
        return np.eye(3)

    if swap is False:
        q = np.concatenate([q[1:], [q[0]]])  # [w,x,y,z] --> [x,y,z,w]
    R = Rotation.from_quat(q).as_matrix()
    return R


def vec_dot(a, b):
    return np.sum(a * b, axis=0)


def vec_cross(a, b):
    return np.cross(a, b, axis=0)
