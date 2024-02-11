import numpy as np
import matplotlib.pyplot as plt

from .rotations import hat


def axis_equal(X, Y, Z, ax=None):
    """
    Sets axis bounds to "equal" according to the limits of X,Y,Z.
    If axes are not given, it generates and labels a 3D figure.

    Args:
        X: Vector of points in coord. x
        Y: Vector of points in coord. y
        Z: Vector of points in coord. z
        ax: Axes to be modified

    Returns:
        ax: Axes with "equal" aspect


    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    max_range = (
        np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 2.0
    )
    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y.max() + Y.min()) * 0.5
    mid_z = (Z.max() + Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    return ax


def plot_frames(
    r, e1, e2, e3, interval=0.9, scale=1.0, ax=None, ax_equal=True, planar=False
):
    """
    Plots the moving frame [e1,e2,e3] of the curve r. The amount of frames to
    be plotted can be controlled with "interval".

    Args:
        r: Vector of 3d points (x,y,z) of curve
        e1: Vector of first component of frame
        e2: Vector of second component of frame
        e3: Vector of third component of frame
        interval: Percentage of frames to be plotted, i.e, 1 plots a frame in
                  every point of r, while 0 does not plot any.
        scale: Float to size components of frame
        ax: Axis where plot will be modified

    Returns:
        ax: Modified plot
    """
    # scale = 0.1
    nn = r.shape[0]
    tend = r + e1 * scale
    nend = r + e2 * scale
    bend = r + e3 * scale

    if ax is None:
        ax = plt.figure().add_subplot(111, projection="3d")

    if planar:
        for i in range(0, nn, int(nn * (1 - interval))):  # if nn >1 else 1):
            # ax.plot([r[i, 0], tend[i, 0]], [r[i, 1], tend[i, 1]], "r")
            # ax.plot([r[i, 0], nend[i, 0]], [r[i, 1], nend[i, 1]], "g")

            ax.plot([r[i, 0], tend[i, 0]], [r[i, 2], tend[i, 2]], "r")  # , linewidth=2)
            ax.plot([r[i, 0], bend[i, 0]], [r[i, 2], bend[i, 2]], "g")  # , linewidth=2)
        ax.set_aspect("equal")

    else:
        if ax_equal:
            ax = axis_equal(r[:, 0], r[:, 1], r[:, 2], ax=ax)
        if interval == 1:
            rng = range(nn)
        else:
            rng = range(0, nn, int(nn * (1 - interval)) if nn > 1 else 1)

        for i in rng:
            ax.plot(
                [r[i, 0], tend[i, 0]], [r[i, 1], tend[i, 1]], [r[i, 2], tend[i, 2]], "r"
            )
            ax.plot(
                [r[i, 0], nend[i, 0]], [r[i, 1], nend[i, 1]], [r[i, 2], nend[i, 2]], "g"
            )
            ax.plot(
                [r[i, 0], bend[i, 0]], [r[i, 1], bend[i, 1]], [r[i, 2], bend[i, 2]], "b"
            )

    return ax


def visualize_frames_and_rope(xLd, xQ, Rl, R, ax):
    # plot system frames
    e1 = R[:, 0, :]
    e2 = R[:, 1, :]
    e3 = R[:, 2, :]

    scale = 0.1
    alpha = 1
    for k in range(0, xQ.shape[1], int(xQ.shape[1] / 20)):
        ax.plot(
            [xQ[0, k], xQ[0, k] + scale * e1[0, k]],
            [xQ[1, k], xQ[1, k] + scale * e1[1, k]],
            [xQ[2, k], xQ[2, k] + scale * e1[2, k]],
            "r-",
            alpha=alpha,
        )
        ax.plot(
            [xQ[0, k], xQ[0, k] + scale * e2[0, k]],
            [xQ[1, k], xQ[1, k] + scale * e2[1, k]],
            [xQ[2, k], xQ[2, k] + scale * e2[2, k]],
            "g-",
            alpha=alpha,
        )
        ax.plot(
            [xQ[0, k], xQ[0, k] + scale * e3[0, k]],
            [xQ[1, k], xQ[1, k] + scale * e3[1, k]],
            [xQ[2, k], xQ[2, k] + scale * e3[2, k]],
            "b-",
            alpha=alpha,
        )

    # plot rope and load frames
    e1 = Rl[:, 0, :]
    e2 = Rl[:, 1, :]
    e3 = Rl[:, 2, :]

    scale = 0.1
    alpha = 1
    for k in range(0, xQ.shape[1], int(xQ.shape[1] / 20)):
        ax.plot(
            [xQ[0, k], xLd[0, k]],
            [xQ[1, k], xLd[1, k]],
            [xQ[2, k], xLd[2, k]],
            "-k",
            alpha=0.5,
            linewidth=0.5,
        )

        ax.plot(
            [xLd[0, k], xLd[0, k] + scale * e1[0, k]],
            [xLd[1, k], xLd[1, k] + scale * e1[1, k]],
            [xLd[2, k], xLd[2, k] + scale * e1[2, k]],
            "r-",
            alpha=alpha,
        )
        ax.plot(
            [xLd[0, k], xLd[0, k] + scale * e2[0, k]],
            [xLd[1, k], xLd[1, k] + scale * e2[1, k]],
            [xLd[2, k], xLd[2, k] + scale * e2[2, k]],
            "g-",
            alpha=alpha,
        )
        ax.plot(
            [xLd[0, k], xLd[0, k] + scale * e3[0, k]],
            [xLd[1, k], xLd[1, k] + scale * e3[1, k]],
            [xLd[2, k], xLd[2, k] + scale * e3[2, k]],
            "b-",
            alpha=alpha,
        )

    return ax


def visualize_cup_and_acceleration(xQ, R, Omega, f, M, params, ax):
    mQ = params["mQ"]
    mC = params["mC"]
    g = params["g"]
    J = params["J"]
    J_inv = params["J_inv"]
    e3 = params["e3"]
    r = params["r"]

    cup_points = (
        np.array([[0.1, 0, 0.2], [0.1, 0, 0.0], [-0.1, 0, 0.0], [-0.1, 0, 0.2]]).T * 0.2
    )
    attachment_points = np.concatenate([np.zeros((3, 1)), r], axis=1)

    alpha = 0.5
    scale = -0.01
    for k in range(0, xQ.shape[1], int(xQ.shape[1] / 20)):
        aQ = 1 / mQ * (f[k] * (R[:, :, k] @ e3) - mQ * g * e3)
        Omega_dot = J_inv @ (
            M[:, k : k + 1] - hat(Omega[:, k : k + 1]) @ J @ Omega[:, k : k + 1]
        )
        aC = (
            aQ
            + R[:, :, k] @ hat(Omega_dot) @ r
            + R[:, :, k] @ hat(Omega[:, k : k + 1]) @ hat(Omega[:, k : k + 1]) @ r
        )

        fb3_C = aC + g * e3
        fb3_Q = aQ + g * e3
        p1 = R[:, :, k] @ attachment_points + xQ[:, k : k + 1]
        p2 = R[:, :, k] @ (cup_points + r) + xQ[:, k : k + 1]
        ax.plot(
            p1[0, :],
            p1[1, :],
            p1[2, :],
            "k-",
            alpha=alpha,
        )
        ax.plot(
            p2[0, :],
            p2[1, :],
            p2[2, :],
            "k-",
            alpha=alpha,
        )
        ax.plot(
            [p1[0, 1], (p1[0, 1] + scale * fb3_C[0]).item()],
            [p1[1, 1], (p1[1, 1] + scale * fb3_C[1]).item()],
            [p1[2, 1], (p1[2, 1] + scale * fb3_C[2]).item()],
            "r-",
            alpha=alpha,
            linewidth=3,
        )

    return ax
