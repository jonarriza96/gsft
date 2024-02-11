import numpy as np

import time
import pickle

import roboticstoolbox as rtb
import swift
import spatialgeometry as sg


def paper_figure_visualization(trajectory, Q, TEd):
    # Create swift instance
    env = swift.Swift()
    env.launch(realtime=True)
    env.set_camera_pose([1.3, 0, 0.6], [0, 0, -0.3])

    # Add Axes
    if trajectory["case"] == "b":
        n_axes = 200
    elif trajectory["case"] == "l":
        n_axes = 150
    elif trajectory["case"] == "h":
        n_axes = 300
    axes_counter = np.linspace(0, 1, n_axes + 1)
    # alpha_axes = np.linspace(0.5, 1, n_axes)
    ind_axes = (axes_counter * (len(Q) - 1)).astype(int)
    for i in range(n_axes):
        goal_axes = sg.Arrow(length=0.05, head_radius=0.4, color="cyan")
        # goal_axes = sg.Axes(length=0.05)
        if trajectory["case"] == "l":
            R = TEd[ind_axes[i]][:3, :3]
            e1 = R[:, 0]
            e2 = R[:, 1]
            e3 = R[:, 2]
            R = np.vstack([e3, -e2, e1]).T
            TEd[ind_axes[i]][:3, :3] = R
        goal_axes.T = TEd[ind_axes[i]]
        env.add(goal_axes)

    if trajectory["case"] == "b":
        n_robots = 4
        robot_counter = np.linspace(0, 1, n_robots + 1)
        alpha_robots = np.linspace(0.5, 1, n_robots)
        ind_robots = (robot_counter * (len(Q) - 1)).astype(int)
    elif trajectory["case"] == "l":
        n_robots = 4
        n = 3
        robot_counter = np.linspace(0, 1, n_robots + n)
        alpha_robots = np.linspace(0.5, 1, n_robots)
        ind_robots = (robot_counter * (len(Q) - n)).astype(int)
    elif trajectory["case"] == "h":
        # n_robots = 7
        # robot_counter = np.linspace(0, 1, n_robots)
        # alpha_robots = np.linspace(0.5, 1, n_robots)
        # ind_robots = (robot_counter * (len(Q) - 1)).astype(int)

        # ind_robots = np.array([0, 185, 435, 740, 1110])
        # ind_robots = np.array([90, 435, 740, 1110])
        # ind_robots = np.array([0, 435, 740, 900])
        ind_robots = np.array([0, 200, 740, 900])
        n_robots = len(ind_robots)
        alpha_robots = np.linspace(0.2, 1, n_robots)
        # alpha_robots = np.linspace(1, 0.2, n_robots)

    for i in range(n_robots):
        panda = rtb.models.Panda()
        panda.q = Q[ind_robots[i]]
        # env.add(panda)
        env.add(panda, robot_alpha=alpha_robots[i])

    env.hold()


def video_scene_visualization(trajectory, Q, TEd):
    # Create swift instance
    env = swift.Swift()
    env.launch(realtime=True)
    env.set_camera_pose([1.3, 0, 0.6], [0, 0, -0.3])
    if trajectory["case"] == "b":
        env.set_camera_pose([1.6, 0.5, 0.85], [0, 0, 0.4])

    # Add panda to the environment
    panda = rtb.models.Panda()
    panda.q = Q[0]
    env.add(panda, robot_alpha=0.5)
    # Add Axes
    if trajectory["case"] == "b":
        n_axes = 200
    elif trajectory["case"] == "l":
        n_axes = 150
    elif trajectory["case"] == "h":
        n_axes = 300
    axes_counter = np.linspace(0, 1, n_axes + 1)
    # alpha_axes = np.linspace(0.5, 1, n_axes)
    ind_axes = (axes_counter * (len(Q) - 1)).astype(int)

    # frames
    goal_axes = []
    for i in range(n_axes):
        # goal_axes = sg.Arrow(length=0.05, head_radius=0.4, color="cyan")
        goal_axes += [sg.Axes(length=0.05)]
        # if trajectory["case"] == "l":
        #     R = TEd[ind_axes[i]][:3, :3]
        #     e1 = R[:, 0]
        #     e2 = R[:, 1]
        #     e3 = R[:, 2]
        #     R = np.vstack([e3, -e2, e1]).T
        #     TEd[ind_axes[i]][:3, :3] = R
        goal_axes[-1].T = TEd[ind_axes[i]]
        env.add(goal_axes[-1])

    time.sleep(8)
    for g in goal_axes:
        env.remove(g)

    # axes direction
    for i in range(n_axes):
        goal_axes = sg.Arrow(length=0.05, head_radius=0.4, color="cyan")
        # goal_axes = sg.Axes(length=0.05)
        if trajectory["case"] == "l":
            R = TEd[ind_axes[i]][:3, :3]
            e1 = R[:, 0]
            e2 = R[:, 1]
            e3 = R[:, 2]
            R = np.vstack([e3, -e2, e1]).T
            TEd[ind_axes[i]][:3, :3] = R
        goal_axes.T = TEd[ind_axes[i]]
        env.add(goal_axes)


def save_data_paper(trajectory, t_nav, esf_deg):
    path = "/home/jonarriza96/geometric_slosh_free/gsfc/results/data"
    pickle_path = path + "/" + trajectory["case"] + "_esf.pickle"

    # open existing pickle
    try:
        with open(pickle_path, "rb") as file:
            data = pickle.load(file)
    except:
        data = {}
    # add data to pickle
    if trajectory["slosh-free"]:
        data["t_sf"] = t_nav
        data["e_sf"] = esf_deg
    else:
        data["t_nsf"] = t_nav
        data["e_nsf"] = esf_deg

    # save pickle
    with open(pickle_path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_data_experiments(trajectory, t_nav, Q, Qd, Qdd):
    file_name = str(trajectory["t_start"]) + "_" + trajectory["case"]
    if not trajectory["slosh-free"]:
        file_name += "_nsf"
    pickle_path = (
        "/home/jonarriza96/geometric_slosh_free/gsfc/results/data"
        + "/experiments/"
        + trajectory["case"]
        + "/"
        + file_name
        + ".pickle"
    )

    # 1kHz for experiments
    #################################################
    import scipy.interpolate as interpolate

    # q
    fq1 = interpolate.interp1d(t_nav, Q[:, 0])
    fq2 = interpolate.interp1d(t_nav, Q[:, 1])
    fq3 = interpolate.interp1d(t_nav, Q[:, 2])
    fq4 = interpolate.interp1d(t_nav, Q[:, 3])
    fq5 = interpolate.interp1d(t_nav, Q[:, 4])
    fq6 = interpolate.interp1d(t_nav, Q[:, 5])
    fq7 = interpolate.interp1d(t_nav, Q[:, 6])
    fq = lambda t: np.array([fq1(t), fq2(t), fq3(t), fq4(t), fq5(t), fq6(t), fq7(t)]).T

    # qd
    fqd1 = interpolate.interp1d(t_nav, Qd[:, 0])
    fqd2 = interpolate.interp1d(t_nav, Qd[:, 1])
    fqd3 = interpolate.interp1d(t_nav, Qd[:, 2])
    fqd4 = interpolate.interp1d(t_nav, Qd[:, 3])
    fqd5 = interpolate.interp1d(t_nav, Qd[:, 4])
    fqd6 = interpolate.interp1d(t_nav, Qd[:, 5])
    fqd7 = interpolate.interp1d(t_nav, Qd[:, 6])
    fqd = lambda t: np.array(
        [fqd1(t), fqd2(t), fqd3(t), fqd4(t), fqd5(t), fqd6(t), fqd7(t)]
    ).T

    # qdd
    fqdd1 = interpolate.interp1d(t_nav, Qdd[:, 0])
    fqdd2 = interpolate.interp1d(t_nav, Qdd[:, 1])
    fqdd3 = interpolate.interp1d(t_nav, Qdd[:, 2])
    fqdd4 = interpolate.interp1d(t_nav, Qdd[:, 3])
    fqdd5 = interpolate.interp1d(t_nav, Qdd[:, 4])
    fqdd6 = interpolate.interp1d(t_nav, Qdd[:, 5])
    fqdd7 = interpolate.interp1d(t_nav, Qdd[:, 6])
    fqdd = lambda t: np.array(
        [fqdd1(t), fqdd2(t), fqdd3(t), fqdd4(t), fqdd5(t), fqdd6(t), fqdd7(t)]
    ).T

    # evaluate interpolator at a higher frequency
    t_eval = np.arange(t_nav[0], t_nav[-1], 1e-3)
    Qi = fq(t_eval)
    Qdi = fqd(t_eval)
    Qddi = fqdd(t_eval)

    data = {"t_nav": t_eval, "q": Qi, "qd": Qdi, "qdd": Qddi}
    ##################################################
    # save pickle
    with open(pickle_path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
