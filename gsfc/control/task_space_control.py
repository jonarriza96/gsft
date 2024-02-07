import numpy as np
import math
import roboticstoolbox as rtb

from gsfc.utils.rotations import hat, vee


def vel_p_servo(
    Te: np.ndarray, Tep: np.ndarray, gain: np.ndarray, threshold: float = 0.1
):
    """
    Position-based servoing.

    Returns the end-effector velocity which will cause the robot to approach
    the desired pose.

    :param Te: The current pose of the end-effecor in the base frame.
    :type wTe: ndarray
    :param Tep: The desired pose of the end-effecor in the base frame.
    :type wTep: ndarray
    :param gain: The gain for the controller. A vector corresponding to each
        Cartesian axis.
    :type gain: array-like
    :param threshold: The threshold or tolerance of the final error between
        the robot's pose and desired pose
    :type threshold: float

    :returns v: The velocity of the end-effector which will casue the robot
        to approach Tep
    :rtype v: ndarray(6)
    :returns arrived: True if the robot is within the threshold of the final
        pose
    :rtype arrived: bool
    """

    # Calculate the pose error vector
    e = angle_axis(Te, Tep)

    # Construct our gain diagonal matrix
    k = np.diag(gain)

    # Calculate our desired end0effector velocity
    v = k @ e

    # Check if we have arrived
    arrived = True if np.sum(np.abs(e)) < threshold else False

    return v, arrived


def acc_p_servo(
    Te: np.ndarray,
    Tep: np.ndarray,
    v: np.ndarray,
    gain: np.ndarray,
    threshold: float = 0.1,
    base="0",
    vd=None,
):
    """
    Position-based servoing.

    Returns the end-effector velocity which will cause the robot to approach
    the desired pose.

    :param Te: The current pose of the end-effecor in the base frame.
    :type wTe: ndarray
    :param Tep: The desired pose of the end-effecor in the base frame.
    :type wTep: ndarray
    :param v: The current long and ang. velocity of the end-effecor in the base frame.
    :type wv: ndarray
    :param gain: The gain for the controller. A vector corresponding to each
        Cartesian axis.
    :type gain: array-like
    :param threshold: The threshold or tolerance of the final error between
        the robot's pose and desired pose
    :type threshold: float

    :returns v: The velocity of the end-effector which will casue the robot
        to approach Tep
    :rtype v: ndarray(6)
    :returns arrived: True if the robot is within the threshold of the final
        pose
    :rtype arrived: bool
    """

    # Construct our gain diagonal matrix
    kP = 1 * np.diag(gain)
    kV = 10 * np.diag(gain)

    # ----------------------- Outer loop: Velocity control ----------------------- #
    if vd is None:
        if base == "0":
            # Calculate the pose error vector
            eP = angle_axis(Te, Tep)
            # ep = Tep[:3, 3] - Te[:3, 3]
            # eq = q_error(Te[:3, 0].T, Tep[:3, 0].T)  # np.zeros(3)
            # eP = np.hstack((ep, eq))

            # Check if we have arrived
            e = np.sum(np.abs(eP))
            arrived = True if e < threshold else False

            # Calculate our desired end0effector velocity
            vd = kP @ eP

        elif base == "e":
            vd, arrived = rtb.p_servo(
                Te, Tep, gain=np.diag(kP), threshold=threshold, method="rpy"
            )
            e = 1
    else:
        eP = angle_axis(Te, Tep)
        e = np.sum(np.abs(eP))
        arrived = True if e < threshold else False
    # --------------------- Inner loop: Acceleration control --------------------- #
    # Calculate the velocity twist error
    eV = vd - v

    # Calculate our desired end0effector acceleration
    ad = kV @ eV

    # print("eP", eP)
    # print("eV", eV)
    # print("ad", ad)

    # ##################
    # ad_max = 5
    # # Calculate the magnitude of the ee velocity
    # ad_norm = np.linalg.norm(ad)

    # # if ee vel is greater than the max
    # if ad_norm > ad_max:
    #     print("in")
    #     ad = (ad_max / ad_norm) * ad
    # ##################

    return ad, vd, e, arrived


def angle_axis(T: np.ndarray, Td: np.ndarray) -> np.ndarray:
    """
    Returns the error vector between T and Td in angle-axis form.

    :param T: The current pose
    :param Tep: The desired pose

    :returns e: the error vector between T and Td
    """

    e = np.empty(6)

    # The position error
    e[:3] = Td[:3, -1] - T[:3, -1]

    R = Td[:3, :3] @ T[:3, :3].T

    li = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])

    if np.linalg.norm(li) < 1e-6:
        # If li is a zero vector (or very close to it)

        # diagonal matrix case
        if np.trace(R) > 0:
            # (1,1,1) case
            a = np.zeros((3,))
        else:
            a = np.pi / 2 * (np.diag(R) + 1)
    else:
        # non-diagonal matrix case
        ln = np.linalg.norm(li)
        a = math.atan2(ln, np.trace(R) - 1) * li / ln

    e[3:] = a

    return e


def q_error(q, qd):
    e = np.dot(hat(q * q), qd)
    return e


def geometric_control_se3(p, v, R, av, pd, vd, ad, jd, sd, yawd, yawdotd, gains):
    g = np.array([0, 0, -9.81])
    kx = gains["kx"]
    kv = gains["kv"]
    kr = gains["kr"]
    ko = gains["ko"]

    # translational acceleration
    ex = p - pd
    ev = v - vd
    a_magnitude = np.dot((-kx * ex - kv * ev - g + ad), R[:, 2])
    at_sf = a_magnitude * R[:, 2]

    # commanded rotation matrix from diff. flat
    b2d = np.array([-np.sin(yawd), np.cos(yawd), 0])
    b3d = (-kx * ex - kv * ev - g + ad) / np.linalg.norm(-kx * ex - kv * ev - g + ad)
    b1d = np.cross(b2d, b3d) / np.linalg.norm(np.cross(b2d, b3d))
    b2d = np.cross(b3d, b1d) / np.linalg.norm(np.cross(b3d, b1d))
    Rc = np.vstack([b1d, b2d, b3d]).T

    # commanded angular velocity from diff. flat
    bx = R[:, 0]
    by = R[:, 1]
    bz = R[:, 2]
    hw = (jd - np.dot(bz, jd) * bz) / a_magnitude
    avc = np.array([-np.dot(by, hw), np.dot(bx, hw), yawdotd * bz[2]])

    # commanded angular acceleration from diff. flat
    aac = np.array([0, 0, 0])  # R,jd,sd #TODO

    # rotational acceleration
    er = 1 / 2 * vee(Rc.T @ R - R.T @ Rc)
    eo = av - R.T @ Rc @ avc
    ar_sf = -kr * er - ko * eo  # - hat(av) @ R.T @ Rc @ avc + R.T @ Rc @ aac

    return at_sf, ar_sf, ex, ev, er, eo
