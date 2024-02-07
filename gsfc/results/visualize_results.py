import numpy as np
import matplotlib.pyplot as plt

import pickle
from sys import exit

path = "/home/jonarriza96/geometric_slosh_free/gsfc/results/"
case = "h"

fs = 16  # font size
ls = 10  # label size
# ------------------------------- esf_deg plot ------------------------------- #
# pickle_path = path + "data/" + case + "_esf.pickle"
# with open(pickle_path, "rb") as file:
#     data = pickle.load(file)
# t_sf = data["t_sf"]
# e_sf = data["e_sf"]
# t_nsf = data["t_nsf"]
# e_nsf = data["e_nsf"]

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.grid(axis="x")
# ax.tick_params(labelsize=ls)
# ax.plot(t_sf[1:], e_sf, "-b")
# ax.plot(t_nsf[1:], e_nsf, "-r")
# if case == "b":
#     ax.set_ylabel("$e_{sf}$ [deg.]", fontsize=fs)
# ax.set_xlabel("t [s]", fontsize=fs)

# if case == "h":
#     fig.set_size_inches(4.5, 7.8 / 3.5)
# elif case == "l":
#     fig.set_size_inches(4.7, 7.8 / 3.5)
# elif case == "b":
#     fig.set_size_inches(3.4, 7.8 / 3.5)
# plt.tight_layout()

# fig.savefig(path + "figures/" + case + "_esf.pdf", dpi=1800)
# plt.show()

# ----------------------- Navigation time comparison ----------------------- #
pickle_path = path + "data/" + case + ".pickle"
with open(pickle_path, "rb") as file:
    data = pickle.load(file)

t_navs = data["t_navs"]
Ep = data["Ep"]
Esf = data["Esf"]
MaxEsf = data["MaxEsf"]
Sl = data["Sl"]

if case == "b":
    xticklabels = ["3.5", "3.75", "4", "4.25", "4.5", "4.75", "5"]
elif case == "l":
    xticklabels = ["4", "4.25", "4.5", "4.75", "5", "5.25", "5.5"]
elif case == "h":
    xticklabels = ["6.5", "6.75", "7", "7.25", "7.5", "7.75", "8"]

fig, ax = plt.subplots(4, 1, sharex=True)
plt.setp(ax, xticks=t_navs, xticklabels=xticklabels)
for AX in ax:
    AX.grid(axis="x")
ax1, ax2, ax3, ax4 = ax

ax1.invert_xaxis()
ax1.plot(t_navs, Ep["nsf"] * 1000, "ro-")
ax1.plot(t_navs, Ep["sf"] * 1000, "bo-")
ax1.tick_params(labelsize=ls)
if case == "b":
    ax1.set_ylabel("$E_p$\n[mm s]", fontsize=fs)
ax2.plot(t_navs, Esf["nsf"], "ro-")
ax2.plot(t_navs, Esf["sf"], "bo-")
ax2.tick_params(labelsize=ls)
if case == "b":
    ax2.set_ylabel("$E_{sf}$\n[deg. s]", fontsize=fs)

ax3.plot(t_navs, MaxEsf["nsf"], "ro-")
ax3.plot(t_navs, MaxEsf["sf"], "bo-")
if case == "b":
    ax3.set_ylabel("$\max{\,e_{sf}}$\n[deg.]", fontsize=fs)
ax3.tick_params(labelsize=ls)

ax4.plot(t_navs, Sl["sf"], "mo-")  # TODO slacks
if case == "b":
    ax4.set_ylabel("Sl", fontsize=fs)
ax4.set_yscale("log")
ax4.set_xlabel("Trajectory end-time [s]", fontsize=fs)
ax4.tick_params(labelsize=ls)

if case == "h":
    fig.set_size_inches(4.5, 7.8 * 2 / 3)
elif case == "l":
    fig.set_size_inches(4.7, 7.8 * 2 / 3)
elif case == "b":
    fig.set_size_inches(3.4, 7.8 * 2 / 3)
plt.tight_layout()

fig.savefig(path + "figures/" + case + ".pdf", dpi=1800)
plt.show()
