import numpy as np

visualization = {
    "task_space": False,
    "joint_space": True,
    "simulation": True,
    "desired_acc": True,
    "slosh_freeness": True,
    "singularity": False,
}

trajectory = {
    "case": "l",  # wp (waypoint), s (straight), b (backflip), lissajous
    "yaw": np.pi / 2,
    "dt": 1e-3,  # Time step when evaluating minimum jerk trajectory
    "t_start": 4,
    "disturbance": False,
    "slosh-free": 1,
    "replan": 0,
    "replan_fb": 1,
    "timeout": 30 * 3,  # 00,
}

sc = 1
controller = {
    "cmd": "wp",  # v (vel), a (acc)
    "task_space": "sr",  # ff (feedforward), sr (servo), gc (geometric control)
    "joint_space": "qp",  # socp and qp_term have got a slosh-free term
    "slack": True,
    "stop_tol": 1e-5,
    "gain_sr": 10
    * np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),  # Gain for servo controller
    "gains_gc": {
        "kx": 18 * sc,
        "kv": 8 * sc,
        "kr": np.array([150, 150, 2]) * sc,
        "ko": 20 * sc,
    },  # Gain for geoemtric controller
    "manipulability": 0,  # Gain for the manipulability
    "sf_th": 1e-3,
}
