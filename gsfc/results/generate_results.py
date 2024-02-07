import numpy as np
import matplotlib.pyplot as plt

import subprocess
import pickle
from sys import exit


def run_command(slosh_free, t_nav, case):
    command = [
        "python",
        "/home/jonarriza96/geometric_slosh_free/gsfc/gsfcrm0.py",
        "--no_visualization",
        "--case",
        case,
        "--t_nav",
        str(t_nav),
    ]

    if not slosh_free:
        command += ["--nsf"]

    print("Running --> ", command)
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    print("Done")

    Ep, Esf, MaxEsf, Sl = result.stdout.split()[-4:]

    return float(Ep), float(Esf), float(MaxEsf), float(Sl)


case_study = "h"

if case_study == "l":
    t_navs = np.array([4, 4.25, 4.5, 4.75, 5, 5.25, 5.5])
elif case_study == "b":
    t_navs = np.array([3.5, 3.75, 4, 4.25, 4.5, 4.75, 5])
elif case_study == "h":
    t_navs = np.array([6.5, 6.75, 7, 7.25, 7.5, 7.75, 8])


Ep = {"sf": [], "nsf": []}
Esf = {"sf": [], "nsf": []}
MaxEsf = {"sf": [], "nsf": []}
Sl = {"sf": [], "nsf": []}
for t_nav in t_navs:
    # slosh-free
    Ep_i, Esf_i, MaxEsf_i, Sl_i = run_command(
        slosh_free=True, t_nav=t_nav, case=case_study
    )
    Ep["sf"] += [Ep_i]
    Esf["sf"] += [Esf_i]
    MaxEsf["sf"] += [MaxEsf_i]
    Sl["sf"] += [Sl_i]

    # non slosh-free
    Ep_i, Esf_i, MaxEsf_i, Sl_i = run_command(
        slosh_free=False, t_nav=t_nav, case=case_study
    )
    Ep["nsf"] += [Ep_i]
    Esf["nsf"] += [Esf_i]
    MaxEsf["nsf"] += [MaxEsf_i]
    Sl["nsf"] += [Sl_i]


for key in ["sf", "nsf"]:
    Ep[key] = np.squeeze(Ep[key])
    Esf[key] = np.squeeze(Esf[key])
    MaxEsf[key] = np.squeeze(MaxEsf[key])
    Sl[key] = np.squeeze(Sl[key])

plt.subplot(411)
plt.plot(t_navs, Ep["nsf"], "ro-")
plt.plot(t_navs, Ep["sf"], "bo-")
plt.ylabel("Ep")
plt.subplot(412)
plt.plot(t_navs, Esf["nsf"], "ro-")
plt.plot(t_navs, Esf["sf"], "bo-")
plt.ylabel("Esf")
plt.subplot(413)
plt.plot(t_navs, MaxEsf["nsf"], "ro-")
plt.plot(t_navs, MaxEsf["sf"], "bo-")
plt.ylabel("MaxEsf")
plt.subplot(414)
plt.plot(t_navs, Sl["nsf"], "ro-")
plt.plot(t_navs, Sl["sf"], "bo-")
plt.ylabel("Sl")
plt.show()

path = "/home/jonarriza96/geometric_slosh_free/gsfc/results/data"
pickle_path = path + "/" + case_study + ".pickle"
data = {"t_navs": t_navs, "Ep": Ep, "Esf": Esf, "MaxEsf": MaxEsf, "Sl": Sl}
with open(pickle_path, "wb") as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


exit()

# --------------------------------- Lissajous ------------------------------ #
t = np.array([3, 3.5, 3.75, 4, 4.5, 5, 5.5, 6, 6.5, 7])

Ep = {
    "sf": np.array(
        [
            0.1547,
            0.07244,
            0.02876,
            0.0172,
            0.015189,
            0.014,
            0.01303,
            0.0122,
            0.0115,
            0.011,
        ]
    ),
    "nsf": np.array(
        [
            0.022,
            0.018999,
            0.023384,
            0.0172,
            0.01527,
            0.014,
            0.01304,
            0.0122,
            0.01155,
            0.011,
        ]
    ),
    # "ff": np.array([np.nan, 0.015137, 0.0175, 0.01997, 0.0223, 0.0245, 0.0268]),
}

Esf = {
    "sf": np.array(
        [
            46.27,
            27.2011,
            11.32225,
            1.9818,
            0.88069,
            0.543,
            0.3539,
            0.2425,
            0.17475,
            0.1369,
        ]
    ),
    "nsf": np.array(
        [
            25.584,
            21.7383,
            20.3505,
            19.1325,
            17.1666,
            15.6504,
            14.3574,
            13.266,
            12.33,
            11.5165,
        ]
    ),
    # "ff": np.array([np.nan, 0.7849, 0.9893, 1.2547, 1.5112, 1.760359, 1.999]),
}

esf_max = {
    "sf": np.array(
        [82.96, 87.2966, 38.352, 6.473, 1.15818, 0.591, 0.32237, 0.19, 0.1181, 0.0755]
    ),
    "nsf": np.array(
        [
            42.2177,
            27.1998,
            22.54198,
            19.0189,
            14.1330,
            10.9696,
            8.79051,
            7.2203,
            6.0459,
            5.1425,
        ]
    ),
    # "ff": np.array([np.nan, 0.98437, 0.486, 0.5324, 0.5715, 0.59973, 0.62]),
}


indx_start = 2
t = t[indx_start:]
for key in ["sf", "nsf"]:
    Ep[key] = Ep[key][indx_start:]
    Esf[key] = Esf[key][indx_start:]
    esf_max[key] = esf_max[key][indx_start:]
# --------------------------------- Visualize -------------------------------- #
fig = plt.figure()
ax = fig.add_subplot(3, 1, 1)
# ax.plot(t, Esf["ff"], "o--k", label="DF + FF + QP")
ax.plot(t, Esf["nsf"], "o-r", label="PD + QP")
ax.plot(t, Esf["sf"], "o-b", label="DF + PD + QP")
ax.legend()
ax.set_ylabel(r"$E_{sf}$ [deg s]")

ax = fig.add_subplot(3, 1, 2)
# ax.plot(t, esf_max["ff"], "o--k", label="DF + FF + QP")
ax.plot(t, esf_max["nsf"], "o-r", label="PD + QP")
ax.plot(t, esf_max["sf"], "o-b", label="DF + PD + QP")
# ax.legend()
ax.set_ylabel(r"$\overline{e_{sf}}$ [deg]")

ax = fig.add_subplot(3, 1, 3)
# ax.plot(t, Ep["ff"], "o--k", label="DF + FF + QP")
ax.plot(t, Ep["nsf"], "o-r", label="PD + QP")
ax.plot(t, Ep["sf"], "o-b", label="DF + PD + QP")
# ax.legend()
ax.set_ylabel(r"$E_p $ [m s]")
ax.set_xlabel("Trajectory end-time [s]")

plt.show()
