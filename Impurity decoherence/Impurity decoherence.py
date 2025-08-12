"""QWZ Model: Impurity decoherence.

Introduces an impurity into the model and calculates the deocherence as a
function of time. Plots the decoherence magnitude and phase for a variety of
parameters (system length L, energy splitting m, temperature T, chemical
potential μ, impurity location, coupling strength Δ) for linear and logarithmic
time arrays, where the length of the time array may depend on the system
parameters (here the final time of each array depends on the coupling strength
Δ).

@author: Ruaidhrí Campion
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys
sys.path.append("../")
from my_functions import h_QWZ
import itertools as it


# initialising system parameters
lengths = np.linspace(3, 29, 14, dtype=int)
ms = np.array([1., 3.])
Ts, Deltas = (10. ** np.linspace(-4., 1., 11) for _ in range(2))
T_strings, Delta_strings = ([r"10^{" + f"{pow}" + r"}" for pow in [
    "-4", "-3.5", "-3", "-2.5", "-2", "-1.5", "-1", "-0.5", "0", "0.5", "1"
    ]] for _ in range(2))
mus = np.array([0.])
places = ["corner", "edge", "centre"]
imp_index = np.array([np.zeros(len(lengths)), (lengths-1) // 2,
                      ((lengths-1)//2) * (lengths+1)]).T
scales = ["linear", "log"]
t_nos = 201
ts = np.array([[np.linspace(0., last, t_nos), np.geomspace(.1, last, t_nos)]
               for last in 20. / Deltas])
nus = np.empty((len(lengths),
                len(ms),
                len(Ts),
                len(mus),
                len(places),
                len(Deltas),
                len(scales),
                t_nos), dtype=np.complex128)
colours = ["r",
           "darkorange",
           "y",
           "g",
           "b",
           "purple",
           "k",
           "saddlebrown",
           "c",
           "violet",
           "dimgray"]
blinds = ["#d95f02", "#e7298a", "#1b9e77", "#7570b3"]
styles = ["solid", "dashed"]


# computing decoherence for all combinations of parameters
for L, L_ in enumerate(lengths):
    LxLy = L_**2
    LxLy2 = 2*LxLy
    for m, m_ in enumerate(ms):
        h0 = h_QWZ(L_, L_, 0., m_, 1., 1.)
        h0_vals, U = np.linalg.eig(h0)
        h0_vals = np.real(h0_vals)
        eiE_diag = np.exp((1j) * h0_vals)
        Udag = U.T.conj()
        del U
        for p, pl_ in enumerate(places):
            p_ = imp_index[L, p]
            h1 = h0.copy()
            for D, D_ in enumerate(Deltas):
                h1[p_, p_] = h0[p_, p_] + D_
                h1[p_ + LxLy, p_ + LxLy] = h0[p_ + LxLy, p_ + LxLy] + D_
                h1_vals, W = np.linalg.eig(h1)
                e_iD_diag = np.exp((-1j) * np.real(h1_vals))
                M = np.dot(Udag, W)
                del W
                for (T, T_), (mu, mu_) in it.product(enumerate(Ts),
                                                     enumerate(mus)):
                    S_diag = 1. / (1. + np.exp((h0_vals - mu_) / T_))
                    for (s, s_), t in it.product(enumerate(scales),
                                                 range(t_nos)):
                        t_ = ts[s, D, t]
                        nus[L, m, T, mu, p, D, s, t] = np.linalg.det(
                            np.dot(
                                (S_diag * eiE_diag**t_)[:, None] * M,
                                e_iD_diag[:, None]**t_ * M.T.conj()
                                ) + np.diag(1. - S_diag))
                        print(L_, m_, pl_, D_, T_, mu_, s_)
del (h0, Udag, h1)

# defining an array consisting of |ν|, θ, and -log|ν| for plotting convenience
nus_ = np.empty(((3,) + np.shape(nus)))
nus_[0] = abs(nus)
nus_[1] = np.angle(nus)
nus_[2] = -np.log10(nus_[0])


# creating a figure for plotting |ν|, θ
fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True,
                       constrained_layout=True)
ax[0].set_yscale("log")
ax[0].set_ylabel(r"$|\nu(\tau)|$")
ax[1].set_xlabel(r"$\tau$")
ax[1].set_ylim(-1.1 * np.pi, 1.1 * np.pi)
ax[1].set_yticks([-np.pi, 0., np.pi], labels=[r"$-\pi$", "0", r"$\pi$"])
ax[1].set_ylabel(r"$\theta(\tau)$")


# plotting each individual |ν|, θ for all parameter combinations
for D, D_ in enumerate(Deltas):
    D_s = Delta_strings[D]
    ts_ = ts[D, 0]
    ax[1].set_xlim(ts_[0], ts_[-1])
    ax[1].set_xticks(np.linspace(ts_[0], ts_[-1], 5))
    for T, T_ in enumerate(Ts):
        T_s = T_strings[T]
        for L, L_ in enumerate(lengths):
            for m, m_ in enumerate(ms):
                for mu, mu_ in enumerate(mus):
                    for p, pl_ in enumerate(places):
                        nu_ = nus_[:, L, m, T, mu, p, D, 0]
                        ax[0].plot([ts_[0], ts_[-1]],
                                   [min(nu_[0, 1:]), max(nu_[0, 1:])])
                        ax[0].relim()
                        ax[0].autoscale(axis="y")
                        ax[0].set_ylim(ax[0].get_ylim())
                        for lim in ax[0].get_lines():
                            lim.remove()
                        for i in range(2):
                            ax[i].plot(ts_, nu_[i], color=blinds[i])
                        fig.suptitle(
                            r"$L=%s$, $m=%s$, $T=%s$, $\mu=%s$, %s, $\Delta=%s$"
                            % (L_, m_, T_s, mu_, pl_, D_s))
                        fig.savefig(
                            "%sx%s/abs(ν), θ (L = %s, m = %s, T = %s, μ = %s, %s, Δ = %s).pdf"
                            % (L_, L_, L_, m_, T_, mu_, pl_, D_),
                            bbox_inches="tight")
                        for i in range(2):
                            for line in ax[i].get_lines():
                                line.remove()

                        print(D_, T_, L_, m_, mu_, pl_)


# plotting |ν|, θ with varying lengths on each plot
lines, labels = [], []
for L, L_ in enumerate(lengths):
    lines.append(Line2D([0], [0], color=colours[L % len(colours)],
                        linestyle=styles[int(L/(len(colours)))]))
    labels.append(r"$L=%s$" % L_)
legend = fig.legend(lines, labels, loc="center left", bbox_to_anchor=(1, .5))
for D, D_ in enumerate(Deltas):
    D_s = Delta_strings[D]
    ts_ = ts[D, 0]
    ax[1].set_xlim(ts_[0], ts_[-1])
    ax[1].set_xticks(np.linspace(ts_[0], ts_[-1], 5))
    for T, T_ in enumerate(Ts):
        T_s = T_strings[T]
        for m, m_ in enumerate(ms):
            for mu, mu_ in enumerate(mus):
                for p, pl_ in enumerate(places):
                    nu_ = nus_[:, :, m, T, mu, p, D, 0]
                    ax[0].plot([ts_[0], ts_[-1]], [np.amin(nu_[0, :, 1:]),
                                                   np.amax(nu_[0, :, 1:])])
                    ax[0].relim()
                    ax[0].autoscale(axis="y")
                    ax[0].set_ylim(ax[0].get_ylim())
                    for lim in ax[0].get_lines():
                        lim.remove()
                    for i in range(2):
                        for L, L_ in enumerate(lengths):
                            ax[i].plot(ts_, nu_[i, L],
                                       color=colours[L % len(colours)],
                                       linestyle=styles[int(L/(len(colours)))])
                    fig.suptitle(r"$m=%s$, $T=%s$, $\mu=%s$, %s, $\Delta=%s$" %
                                 (m_, T_s, mu_, pl_, D_s))
                    fig.savefig(
                        "L/L abs(ν), θ (m = %s, T = %s, μ = %s, %s, Δ = %s).pdf"
                        % (m_, T_, mu_, pl_, D_), bbox_inches="tight")
                    for i in range(2):
                        for line in ax[i].get_lines():
                            line.remove()

                    print("L", D_, T_, m_, mu_, pl_)
legend.remove()
del legend


# plotting |ν|, θ with varying energy splittings and impurity locations on each
# plot
lines, labels = [], []
for m, m_ in enumerate(ms):
    for p, pl_ in enumerate(places):
        lines.append(Line2D([0], [0], color=colours[p + m*len(places)]))
        labels.append(r"$m=%s$, %s" % (m_, pl_))
legend = fig.legend(lines, labels, loc="center left", bbox_to_anchor=(1, .5))
for D, D_ in enumerate(Deltas):
    D_s = Delta_strings[D]
    ts_ = ts[D, 0]
    ax[1].set_xlim(ts_[0], ts_[-1])
    ax[1].set_xticks(np.linspace(ts_[0], ts_[-1], 5))
    for T, T_ in enumerate(Ts):
        T_s = T_strings[T]
        for L, L_ in enumerate(lengths):
            for mu, mu_ in enumerate(mus):
                nu_ = nus_[:, L, :, T, mu, :, D, 0]
                ax[0].plot([ts_[0], ts_[-1]], [np.amin(nu_[0, :, :, 1:]),
                                               np.amax(nu_[0, :, :, 1:])])
                ax[0].relim()
                ax[0].autoscale(axis="y")
                ax[0].set_ylim(ax[0].get_ylim())
                for lim in ax[0].get_lines():
                    lim.remove()
                for i in range(2):
                    for m in range(len(ms)):
                        for p in range(len(places)):
                            ax[i].plot(ts_, nu_[i, m, p],
                                       color=colours[p + m*len(places)])
                fig.suptitle(r"$L=%s$, $T=%s$, $\mu=%s$, $\Delta=%s$" %
                             (L_, T_s, mu_, D_s))
                fig.savefig(
                    "m, p/m, p abs(ν), θ (L = %s, T = %s, μ = %s, Δ = %s).pdf"
                    % (L_, T_, mu_, D_), bbox_inches="tight")
                for i in range(2):
                    for line in ax[i].get_lines():
                        line.remove()

                print("m, p", D_, T_, L_, mu_)
legend.remove()
del legend


# plotting |ν|, θ with varying temperatures on each plot
lines, labels = [], []
for T, T_s in enumerate(T_strings):
    lines.append(Line2D([0], [0], color=colours[T]))
    labels.append(r"$T=%s$" % T_s)
legend = fig.legend(lines, labels, loc="center left", bbox_to_anchor=(1, .5))
for D, D_ in enumerate(Deltas):
    D_s = Delta_strings[D]
    ts_ = ts[D, 0]
    ax[1].set_xlim(ts_[0], ts_[-1])
    ax[1].set_xticks(np.linspace(ts_[0], ts_[-1], 5))
    for L, L_ in enumerate(lengths):
        for m, m_ in enumerate(ms):
            for mu, mu_ in enumerate(mus):
                for p, pl_ in enumerate(places):
                    nu_ = nus_[:, L, m, :, mu, p, D, 0]
                    ax[0].plot([ts_[0], ts_[-1]], [np.amin(nu_[0, :, 1:]),
                                                   np.amax(nu_[0, :, 1:])])
                    ax[0].relim()
                    ax[0].autoscale(axis="y")
                    ax[0].set_ylim(ax[0].get_ylim())
                    for lim in ax[0].get_lines():
                        lim.remove()
                    for i in range(2):
                        for T in range(len(Ts)):
                            ax[i].plot(ts_, nu_[i, T], color=colours[T])
                    fig.suptitle(r"$L=%s$, $m=%s$, $\mu=%s$, %s, $\Delta=%s$" %
                                 (L_, m_, mu_, pl_, D_s))
                    fig.savefig(
                        "T/T abs(ν), θ (L = %s, m = %s, μ = %s, %s, Δ = %s).pdf"
                        % (L_, m_, mu_, pl_, D_), bbox_inches="tight")
                    for i in range(2):
                        for line in ax[i].get_lines():
                            line.remove()

                    print("T", D_, L_, m_, mu_, pl_)
plt.close(fig)
del fig, ax, legend


# creating a figure for plotting |ν|, -log|ν|
fig, ax = plt.subplots(2, 2, figsize=(12, 6), sharex="col",
                       constrained_layout=True)
ax[0, 0].set_ylabel(r"$|\nu(\tau)|$")
ax[1, 0].set_xlabel(r"$\tau$")
ax[1, 0].set_ylabel(r"$-\log(|\nu(\tau)|)$")
ax[1, 1].set_xscale("log")
for j in range(2):
    ax[1, j].set_xlabel(r"$\tau$")
    for i in range(2):
        ax[i, j].set_yscale("log")


# plotting each individual |ν|, -log|ν| for all parameter combinations
for D, D_ in enumerate(Deltas):
    D_s = Delta_strings[D]
    ts_ = ts[D]
    ax[1, 0].set_xlim(ts_[0, 0], ts_[0, -1])
    ax[1, 0].set_xticks(np.linspace(ts_[0, 0], ts_[0, -1], 5))
    ax[1, 1].set_xlim(ts_[1, 0], ts_[1, -1])
    for T, T_ in enumerate(Ts):
        T_s = T_strings[T]
        for L, L_ in enumerate(lengths):
            for m, m_ in enumerate(ms):
                for mu, mu_ in enumerate(mus):
                    for p, pl_ in enumerate(places):
                        for i in range(2):
                            nu_ = nus_[2*i, L, m, T, mu, p, D]
                            ax[i, 0].plot([ts_[0, 0], ts_[0, -1]],
                                          [min(nu_[0, 1:]), max(nu_[0, 1:])],
                                          [ts_[1, 0], ts_[1, -1]],
                                          [min(nu_[1]), max(nu_[1])])
                            ax[i, 0].relim()
                            ax[i, 0].autoscale(axis="y")
                            range_ = ax[i, 0].get_ylim()
                            for lim in ax[i, 0].get_lines():
                                lim.remove()
                            for j in range(2):
                                ax[i, j].set_ylim(range_)
                                ax[i, j].plot(ts_[j], nu_[j],
                                              color=blinds[2*i])
                        fig.suptitle(
                            r"$L=%s$, $m=%s$, $T=%s$, $\mu=%s$, %s, $\Delta=%s$"
                            % (L_, m_, T_s, mu_, pl_, D_s))
                        fig.savefig(
                            "%sx%s/abs(ν), -log(abs(ν)) (L = %s, m = %s, T = %s, μ = %s, %s, Δ = %s).pdf"
                            % (L_, L_, L_, m_, T_, mu_, pl_, D_),
                            bbox_inches="tight")
                        for i in range(2):
                            for j in range(2):
                                for line in ax[i, j].get_lines():
                                    line.remove()

                        print(D_, T_, L_, m_, mu_, pl_)


# plotting |ν|, -log|ν| with varying lengths on each plot
lines, labels = [], []
for L, L_ in enumerate(lengths):
    lines.append(Line2D([0], [0], color=colours[L % len(colours)],
                        linestyle=styles[int(L/(len(colours)))]))
    labels.append(r"$L=%s$" % L_)
legend = fig.legend(lines, labels, loc="center left", bbox_to_anchor=(1, .5))
for D, D_ in enumerate(Deltas):
    D_s = Delta_strings[D]
    ts_ = ts[D]
    ax[1, 0].set_xlim(ts_[0, 0], ts_[0, -1])
    ax[1, 0].set_xticks(np.linspace(ts_[0, 0], ts_[0, -1], 5))
    ax[1, 1].set_xlim(ts_[1, 0], ts_[1, -1])
    for T, T_ in enumerate(Ts):
        T_s = T_strings[T]
        for m, m_ in enumerate(ms):
            for mu, mu_ in enumerate(mus):
                for p, pl_ in enumerate(places):
                    for i in range(2):
                        nu_ = nus_[2*i, :, m, T, mu, p, D]
                        ax[i, 0].plot([ts_[0, 0], ts_[0, -1]],
                                      [np.amin(nu_[:, 0, 1:]),
                                       np.amax(nu_[:, 0, 1:])],
                                      [ts_[1, 0], ts_[1, -1]],
                                      [np.amin(nu_[:, 1]), np.amax(nu_[:, 1])])
                        ax[i, 0].relim()
                        ax[i, 0].autoscale(axis="y")
                        range_ = ax[i, 0].get_ylim()
                        for lim in ax[i, 0].get_lines():
                            lim.remove()
                        for j in range(2):
                            ax[i, j].set_ylim(range_)
                            for L in range(len(lengths)):
                                ax[i, j].plot(ts_[j], nu_[L, j],
                                              color=colours[L % len(colours)],
                                              linestyle=styles[
                                                  int(L/(len(colours)))])
                    fig.suptitle(r"$m=%s$, $T=%s$, $\mu=%s$, %s, $\Delta=%s$" %
                                 (m_, T_s, mu_, pl_, D_s))
                    fig.savefig(
                        "L/L abs(ν), -log(abs(ν)) (m = %s, T = %s, μ = %s, %s, Δ = %s).pdf"
                        % (m_, T_, mu_, pl_, D_), bbox_inches="tight")
                    for i in range(2):
                        for j in range(2):
                            for line in ax[i, j].get_lines():
                                line.remove()

                print("L", D_, T_, m_, mu_, pl_)
legend.remove()
del legend


# plotting |ν|, -log|ν| with varying energy splittings and impurity locations
# on each plot
lines, labels = [], []
for m, m_ in enumerate(ms):
    for p, pl_ in enumerate(places):
        lines.append(Line2D([0], [0], color=colours[p + m*len(places)]))
        labels.append(r"$m=%s$, %s" % (m_, pl_))
legend = fig.legend(lines, labels, loc="center left", bbox_to_anchor=(1, .5))
for D, D_ in enumerate(Deltas):
    D_s = Delta_strings[D]
    ts_ = ts[D]
    ax[1, 0].set_xlim(ts_[0, 0], ts_[0, -1])
    ax[1, 0].set_xticks(np.linspace(ts_[0, 0], ts_[0, -1], 5))
    ax[1, 1].set_xlim(ts_[1, 0], ts_[1, -1])
    for T, T_ in enumerate(Ts):
        T_s = T_strings[T]
        for L, L_ in enumerate(lengths):
            for mu, mu_ in enumerate(mus):
                for i in range(2):
                    nu_ = nus_[2*i, L, :, T, mu, :, D]
                    ax[i, 0].plot([ts_[0, 0], ts_[0, -1]],
                                  [np.amin(nu_[:, :, 0, 1:]),
                                   np.amax(nu_[:, :, 0, 1:])],
                                  [ts_[1, 0], ts_[1, -1]],
                                  [np.amin(nu_[:, :, 1]),
                                   np.amax(nu_[:, :, 1])])
                    ax[i, 0].relim()
                    ax[i, 0].autoscale(axis="y")
                    range_ = ax[i, 0].get_ylim()
                    for lim in ax[i, 0].get_lines():
                        lim.remove()
                    for j in range(2):
                        ax[i, j].set_ylim(range_)
                        for m in range(len(ms)):
                            for p in range(len(places)):
                                ax[i, j].plot(ts_[j], nu_[m, p, j],
                                              color=colours[p + m*len(places)])
                fig.suptitle(r"$L=%s$, $T=%s$, $\mu=%s$, $\Delta=%s$" %
                             (L_, T_s, mu_, D_s))
                fig.savefig(
                    "m, p/m, p abs(ν), -log(abs(ν)) (L = %s, T = %s, μ = %s, Δ = %s).pdf"
                    % (L_, T_, mu_, D_), bbox_inches="tight")
                for i in range(2):
                    for j in range(2):
                        for line in ax[i, j].get_lines():
                            line.remove()

                print("m, p", D_, T_, L_, mu_)
legend.remove()
del legend


# plotting |ν|, -log|ν| with varying temperatures on each plot
lines, labels = [], []
for T, T_s in enumerate(T_strings):
    lines.append(Line2D([0], [0], color=colours[T]))
    labels.append(r"$T=%s$" % T_s)
legend = fig.legend(lines, labels, loc="center left", bbox_to_anchor=(1, .5))
for D, D_ in enumerate(Deltas):
    D_s = Delta_strings[D]
    ts_ = ts[D]
    ax[1, 0].set_xlim(ts_[0, 0], ts_[0, -1])
    ax[1, 0].set_xticks(np.linspace(ts_[0, 0], ts_[0, -1], 5))
    ax[1, 1].set_xlim(ts_[1, 0], ts_[1, -1])
    for L, L_ in enumerate(lengths):
        for m, m_ in enumerate(ms):
            for mu, mu_ in enumerate(mus):
                for p, pl_ in enumerate(places):
                    for i in range(2):
                        nu_ = nus_[2*i, L, m, :, mu, p, D]
                        ax[i, 0].plot([ts_[0, 0], ts_[0, -1]],
                                      [np.amin(nu_[:, 0, 1:]),
                                       np.amax(nu_[:, 0, 1:])],
                                      [ts_[1, 0], ts_[1, -1]],
                                      [np.amin(nu_[:, 1]), np.amax(nu_[:, 1])])
                        ax[i, 0].relim()
                        ax[i, 0].autoscale(axis="y")
                        range_ = ax[i, 0].get_ylim()
                        for lim in ax[i, 0].get_lines():
                            lim.remove()
                        for j in range(2):
                            ax[i, j].set_ylim(range_)
                            for T in range(len(Ts)):
                                ax[i, j].plot(ts_[j], nu_[T, j],
                                              color=colours[T])
                    fig.suptitle(r"$L=%s$, $m=%s$, $\mu=%s$, %s, $\Delta=%s$" %
                                 (L_, m_, mu, pl_, D_s))
                    fig.savefig(
                        "T/T abs(ν), -log(abs(ν)) (L = %s, m = %s, μ = %s, %s, Δ = %s).pdf"
                        % (L_, m_, mu, pl_, D_), bbox_inches="tight")
                    for i in range(2):
                        for j in range(2):
                            for line in ax[i, j].get_lines():
                                line.remove()

                    print("T", D_, L_, m_, mu_, pl_)
plt.close(fig)
del fig, ax, legend


# creating a figure for plotting |ν|, θ, -log|ν| with varying coupling
# strengths on each plot
fig, ax = plt.subplots(3, 1, figsize=(6, 9), sharex=True,
                       constrained_layout=True)
ax[0].set_yscale("log")
ax[0].set_ylabel(r"$|\nu(\tau)|$")
ax[1].set_ylim(-1.1 * np.pi, 1.1 * np.pi)
ax[1].set_yticks([-np.pi, 0., np.pi], labels=[r"$-\pi$", "0", r"$\pi$"])
ax[1].set_ylabel(r"$\theta(\tau)$")
ax[2].set_xscale("log")
ax[2].set_xlim(np.amin(ts[:, 1]), np.amax(ts[:, 1]))
ax[2].set_xlabel(r"$\tau$")
ax[2].set_yscale("log")
ax[2].set_ylabel(r"$-\log(|\nu(\tau)|)$")


# plotting |ν|, θ, -log|ν| with varying coupling strengths on each plot
lines, labels = [], []
for D, D_s in enumerate(Delta_strings):
    lines.append(Line2D([0], [0], color=colours[D]))
    labels.append(r"$\Delta=%s$" % D_s)
legend = fig.legend(lines, labels, loc="center left", bbox_to_anchor=(1, .5))
for T, T_ in enumerate(Ts):
    T_s = T_strings[T]
    for L, L_ in enumerate(lengths):
        for m, m_ in enumerate(ms):
            for p, pl_ in enumerate(places):
                for mu, mu_ in enumerate(mus):
                    for i in range(3):
                        for D in range(len(Deltas)):
                            ax[i].plot(ts[D, 1], nus_[i, L, m, T, mu, p, D, 1],
                                       color=colours[D])
                    for i in range(2):
                        ax[2*i].relim()
                        ax[2*i].autoscale(axis="y")
                    fig.suptitle(r"$L=%s$, $m=%s$, $T=%s$, $\mu=%s$, %s" %
                                 (L_, m_, T_s, mu_, pl_))
                    fig.savefig(
                        "Δ/Δ abs(ν), θ, -log(abs(ν)) (L = %s, m = %s, T = %s, μ = %s, %s).pdf"
                        % (L_, m_, T_, mu_, pl_), bbox_inches="tight")
                    for i in range(3):
                        for line in ax[i].get_lines():
                            line.remove()

                    print("Δ", T_, L_, m_, mu_, pl_)
plt.close(fig)
del fig, ax, legend


# defining useful variables for plotting grids of graphs
L, mu = -1, 0
L_, mu_ = lengths[L], mus[mu]


# creating a figure for plotting |ν|, θ, -log|ν| separately with varying
# energy splittings, impurity locations, and temperatures
fig, ax = plt.subplots(len(ms)*len(places), len(Ts),
                       figsize=(5*len(Ts), 3*len(ms)*len(places)), sharex=True,
                       constrained_layout=True)
for T, T_s in enumerate(T_strings):
    ax[0, T].set_title(r"$T=%s$" % T_s)
    ax[len(ms)*len(places)-1, T].set_xlabel(r"$\tau$")

# plotting |ν| with varying energy splittings, impurity locations, and
# temperatures
for m, m_ in enumerate(ms):
    for p, pl_ in enumerate(places):
        i = p + m*len(places)
        ax[i, 0].set_ylabel(r"$|\nu(\tau)|$ ($m=%s$, %s)" % (m_, pl_))
        for T in range(len(Ts)):
            ax[i, T].set_yscale("log")
for D, D_ in enumerate(Deltas):
    D_s = Delta_strings[D]
    ts_ = ts[D, 0]
    ax[0, 0].set_xlim(ts_[0], ts_[-1])
    ax[0, 0].set_xticks(np.linspace(ts_[0], ts_[-1], 5))
    for m in range(len(ms)):
        for p in range(len(places)):
            for T in range(len(Ts)):
                i = p + m*len(places)
                nu_ = nus_[0, L, m, T, mu, p, D, 0]
                c = colours[T]
                ax[i, T].plot(ts_[1:], nu_[1:], color=c)
                ax[i, T].relim()
                ax[i, T].autoscale(axis="y")
                ax[i, T].set_ylim(ax[i, T].get_ylim())
                ax[i, T].plot(ts_[:2], nu_[:2], color=c)
    fig.suptitle(r"$L=%s$, $\mu=%s$, $\Delta=%s$" % (L_, mu_, D_s))
    fig.savefig("m, p, T/m, p, T abs(ν) (L = %s, μ = %s, Δ = %s).pdf" %
                (L_, mu_, D_))
    for i in range(len(ms)*len(places)):
        for j in range(len(Ts)):
            for line in ax[i, j].get_lines():
                line.remove()
    print("m, p, T |ν|", D_)

# plotting θ with varying energy splittings, impurity locations, and
# temperatures
for T in range(len(Ts)):
    for m, m_ in enumerate(ms):
        for p, pl_ in enumerate(places):
            i = p + m*len(places)
            ax[i, T].set_yscale("linear")
            ax[i, T].set_ylim(-1.1 * np.pi, 1.1 * np.pi)
            ax[i, T].set_yticks([-np.pi, 0., np.pi],
                                labels=[r"$-\pi$", "0", r"$\pi$"])
            if T == 0:
                ax[i, T].set_ylabel(r"$\theta(\tau)$ ($m=%s$, %s)" % (m_, pl_))
            else:
                ax[i, T].tick_params(labelleft=False)
for D, D_ in enumerate(Deltas):
    D_s = Delta_strings[D]
    ts_ = ts[D, 0]
    ax[0, 0].set_xlim(ts_[0], ts_[-1])
    ax[0, 0].set_xticks(np.linspace(ts_[0], ts_[-1], 5))
    for m in range(len(ms)):
        for p in range(len(places)):
            for T in range(len(Ts)):
                i = p + m*len(places)
                ax[i, T].plot(ts_, nus_[1, L, m, T, mu, p, D, 0],
                              color=colours[T])
    fig.suptitle(r"$L=%s$, $\mu=%s$, $\Delta=%s$" % (L_, mu_, D_s))
    fig.savefig("m, p, T/m, p, T θ (L = %s, μ = %s, Δ = %s).pdf" %
                (L_, mu_, D_))
    for i in range(len(ms)*len(places)):
        for j in range(len(Ts)):
            for line in ax[i, j].get_lines():
                line.remove()

    print("m, p, T θ", D_)

# plotting -log|ν| with varying energy splittings, impurity locations, and
# temperatures
ax[0, 0].set_xscale("log")
for T in range(len(Ts)):
    for m, m_ in enumerate(ms):
        for p, pl_ in enumerate(places):
            i = p + m*len(places)
            ax[i, T].set_yscale("log")
            if T == 0:
                ax[i, T].set_ylabel(r"$-\log(|\nu(\tau)|)$ ($m=%s$, %s)" %
                                    (m_, pl_))
            else:
                ax[i, T].tick_params(labelleft=True)
for D, D_ in enumerate(Deltas):
    D_s = Delta_strings[D]
    ts_ = ts[D, 1]
    ax[0, 0].set_xlim(ts_[0], ts_[-1])
    for m in range(len(ms)):
        for p in range(len(places)):
            for T in range(len(Ts)):
                i = p + m*len(places)
                ax[i, T].plot(ts_, nus_[2, L, m, T, mu, p, D, 1],
                              color=colours[T])
                ax[i, T].relim()
                ax[i, T].autoscale(axis="y")
    fig.suptitle(r"$L=%s$, $\mu=%s$, $\Delta=%s$" % (L_, mu_, D_s))
    fig.savefig("m, p, T/m, p, T -log(abs(ν)) (L = %s, μ = %s, Δ = %s).pdf" %
                (L_, mu_, D_))
    for i in range(len(ms)*len(places)):
        for j in range(len(Ts)):
            for line in ax[i, j].get_lines():
                line.remove()
    print("m, p, T -log|ν|", D_)
plt.close(fig)
del fig, ax


# creating a figure for plotting |ν|, θ, -log|ν| separately with varying
# energy splittings, impurity locations, and coupling strengths
fig, ax = plt.subplots(len(ms)*len(places), len(Deltas),
                       figsize=(5*len(Deltas), 3*len(ms)*len(places)),
                       sharex="col", constrained_layout=True)
for D, D_s in enumerate(Delta_strings):
    ax[0, D].set_title(r"$\Delta=%s$" % D_s)
    ax[len(ms)*len(places)-1, D].set_xlabel(r"$\tau$")

# plotting |ν| with varying energy splittings, impurity locations, and
# coupling strengths
for D in range(len(Deltas)):
    ts_ = ts[D, 0]
    ax[0, D].set_xlim(ts_[0], ts_[-1])
    for i in range(len(ms)*len(places)):
        ax[i, D].set_yscale("log")
for m, m_ in enumerate(ms):
    for p, pl_ in enumerate(places):
        i = p + m*len(places)
        ax[i, 0].set_ylabel(r"$|\nu(\tau)|$ ($m=%s$, %s)" % (m_, pl_))
for T, T_ in enumerate(Ts):
    T_s = T_strings[T]
    for D in range(len(Deltas)):
        ts_ = ts[D, 0]
        c = colours[D]
        for m in range(len(ms)):
            for p in range(len(places)):
                i = p + m*len(places)
                nu_ = nus_[0, L, m, T, mu, p, D, 0]
                ax[i, D].plot(ts_[1:], nu_[1:], color=c)
                ax[i, D].relim()
                ax[i, D].autoscale(axis="y")
                ax[i, D].set_ylim(ax[i, D].get_ylim())
                ax[i, D].plot(ts_[:2], nu_[:2], color=c)
    fig.suptitle(r"$L=%s$, $T=%s$, $\mu=%s$" % (L_, T_s, mu_))
    fig.savefig("m, p, Δ/m, p, Δ abs(ν) (L = %s, T = %s, μ = %s).pdf" %
                (L_, T_, mu_))
    for i in range(len(ms)*len(places)):
        for j in range(len(Deltas)):
            for line in ax[i, j].get_lines():
                line.remove()
    print("m, p, Δ |ν|", T_)

# plotting θ with varying energy splittings, impurity locations, and
# coupling strengths
for D in range(len(Deltas)):
    for m, m_ in enumerate(ms):
        for p, pl_ in enumerate(places):
            i = p + m*len(places)
            ax[i, D].set_yscale("linear")
            ax[i, D].set_ylim(-1.1 * np.pi, 1.1 * np.pi)
            ax[i, D].set_yticks([-np.pi, 0., np.pi],
                                labels=[r"$-\pi$", "0", r"$\pi$"])
            if D == 0:
                ax[i, D].set_ylabel(r"$\theta(\tau)$ ($m=%s$, %s)" % (m_, pl_))
            else:
                ax[i, D].tick_params(labelleft=False)
for T, T_ in enumerate(Ts):
    T_s = T_strings[T]
    for D in range(len(Deltas)):
        ts_ = ts[D, 0]
        for m in range(len(ms)):
            for p in range(len(places)):
                i = p + m*len(places)
                ax[i, D].plot(ts_, nus_[1, L, m, T, mu, p, D, 0],
                              color=colours[D])
    fig.suptitle(r"$L=%s$, $T=%s$, $\mu=%s$" % (L_, T_s, mu_))
    fig.savefig("m, p, Δ/m, p, Δ θ (L = %s, T = %s, μ = %s).pdf" %
                (L_, T_, mu_))
    for i in range(len(ms)*len(places)):
        for j in range(len(Deltas)):
            for line in ax[i, j].get_lines():
                line.remove()
    print("m, p, Δ θ", T_)

# plotting -log|ν| with varying energy splittings, impurity locations, and
# coupling strengths
for D in range(len(Deltas)):
    ts_ = ts[D, 1]
    ax[0, D].set_xlim(ts_[0], ts_[-1])
    ax[0, D].set_xscale("log")
    for m, m_ in enumerate(ms):
        for p, pl_ in enumerate(places):
            i = p + m*len(places)
            ax[i, D].set_yscale("log")
            if D == 0:
                ax[i, D].set_ylabel(r"$-\log(|\nu(\tau)|)$ ($m=%s$, %s)" %
                                    (m_, pl_))
            else:
                ax[i, D].tick_params(labelleft=True)
for T, T_ in enumerate(Ts):
    T_s = T_strings[T]
    for D in range(len(Deltas)):
        ts_ = ts[D, 1]
        for m in range(len(ms)):
            for p in range(len(places)):
                i = p + m*len(places)
                ax[i, D].plot(ts_, nus_[2, L, m, T, mu, p, D, 1],
                              color=colours[D])
                ax[i, D].relim()
                ax[i, D].autoscale(axis="y")
    fig.suptitle(r"$L=%s$, $T=%s$, $\mu=%s$" % (L_, T_s, mu_))
    fig.savefig("m, p, Δ/m, p, Δ -log(abs(ν)) (L = %s, T = %s, μ = %s).pdf" %
                (L_, T_, mu_))
    for i in range(len(ms)*len(places)):
        for j in range(len(Deltas)):
            for line in ax[i, j].get_lines():
                line.remove()
    print("m, p, Δ -log|ν|", T_)
plt.close(fig)
del fig, ax


# creating a figure for plotting |ν|, θ, -log|ν| separately with varying
# temperatures and coupling strengths
fig, ax = plt.subplots(len(Ts), len(Deltas),
                       figsize=(5*len(Deltas), 3*len(Ts)), sharex="col",
                       constrained_layout=True)
for D, D_s in enumerate(Delta_strings):
    ax[0, D].set_title(r"$\Delta=%s$" % D_s)
    ax[len(Ts)-1, D].set_xlabel(r"$\tau$")

# plotting |ν| with varying temperatures and coupling strengths
for D, D_s in enumerate(Delta_strings):
    ts_ = ts[D, 0]
    ax[0, D].set_xlim(ts_[0], ts_[-1])
    for T in range(len(Ts)):
        ax[D, T].set_yscale("log")
for T, T_s in enumerate(T_strings):
    ax[T, 0].set_ylabel(r"$|\nu(\tau)|$ ($T=%s$)" % T_s)
for m, m_ in enumerate(ms):
    for p, pl_ in enumerate(places):
        for D in range(len(Deltas)):
            ts_ = ts[D, 0]
            c = colours[D]
            for T in range(len(Ts)):
                nu_ = nus_[0, L, m, T, mu, p, D, 0]
                ax[T, D].plot(ts_[1:], nu_[1:], color=c)
                ax[T, D].relim()
                ax[T, D].autoscale(axis="y")
                ax[T, D].set_ylim(ax[T, D].get_ylim())
                ax[T, D].plot(ts_[:2], nu_[:2], color=c)
        fig.suptitle(r"$L=%s$, $m=%s$, $\mu=%s$, %s" % (L_, m_, mu_, pl_))
        fig.savefig("T, Δ/T, Δ abs(ν) (L = %s, m = %s, μ = %s, %s).pdf"
                    % (L_, m_, mu_, pl_))
        for i in range(len(Ts)):
            for j in range(len(Deltas)):
                for line in ax[i, j].get_lines():
                    line.remove()
        print("T, Δ |ν|", m_, pl_)

# plotting θ with varying temperatures and coupling strengths
for D in range(len(Deltas)):
    for T, T_s in enumerate(T_strings):
        ax[T, D].set_yscale("linear")
        ax[T, D].set_ylim(-1.1 * np.pi, 1.1 * np.pi)
        ax[T, D].set_yticks([-np.pi, 0., np.pi],
                            labels=[r"$-\pi$", "0", r"$\pi$"])
        if D == 0:
            ax[T, D].set_ylabel(r"$\theta(\tau)$ ($T=%s$)" % T_s)
        else:
            ax[T, D].tick_params(labelleft=False)
for m, m_ in enumerate(ms):
    for p, pl_ in enumerate(places):
        for D in range(len(Deltas)):
            ts_ = ts[D, 0]
            for T in range(len(Ts)):
                ax[T, D].plot(ts_, nus_[1, L, m, T, mu, p, D, 0],
                              color=colours[D])
        fig.suptitle(r"$L=%s$, $m=%s$, $\mu=%s$, %s" % (L_, m_, mu_, pl_))
        fig.savefig("T, Δ/T, Δ θ (L = %s, m = %s, μ = %s, %s).pdf" %
                    (L_, m_, mu_, pl_))
        for i in range(len(Ts)):
            for j in range(len(Deltas)):
                for line in ax[i, j].get_lines():
                    line.remove()
        print("T, Δ θ", m_, pl_)

# plotting -log|ν| with varying temperatures and coupling strengths
for D in range(len(Deltas)):
    ts_ = ts[D, 1]
    ax[0, D].set_xlim(ts_[0], ts_[-1])
    ax[0, D].set_xscale("log")
    for T, T_s in enumerate(T_strings):
        ax[D, T].set_yscale("log")
        if D == 0:
            ax[T, D].set_ylabel(r"$-\log(|\nu(\tau)|)$ ($T=%s$)" % T_s)
        else:
            ax[T, D].tick_params(labelleft=True)
for m, m_ in enumerate(ms):
    for p, pl_ in enumerate(places):
        for D in range(len(Deltas)):
            ts_ = ts[D, 1]
            for T in range(len(Ts)):
                ax[T, D].plot(ts_, nus_[2, L, m, T, mu, p, D, 1],
                              color=colours[D])
                ax[T, D].relim()
                ax[T, D].autoscale(axis="y")
        fig.suptitle(r"$L=%s$, $m=%s$, $\mu=%s$, %s" % (L_, m_, mu_, pl_))
        fig.savefig("T, Δ/T, Δ -log(abs(ν)) (L = %s, m = %s, μ = %s, %s).pdf"
                    % (L_, m_, mu_, pl_))
        for i in range(len(Ts)):
            for j in range(len(Deltas)):
                for line in ax[i, j].get_lines():
                    line.remove()
        print("T, Δ -log|ν|", m_, pl_)
plt.close(fig)
del fig, ax


# Calculating phase frequency median over all combinations of parameters
L, mu = -1, 0
L_, mu_ = lengths[L], mus[mu]
zeroes, freqs = (np.empty((len(ms),
                           len(Ts),
                           len(places),
                           len(Deltas),
                           len(scales)), dtype=object) for _ in range(2))

for m in range(len(ms)):
    for T in range(len(Ts)):
        for p in range(len(places)):
            for D in range(len(Deltas)):
                for s in range(len(scales)):
                    ts_ = ts[D, s]
                    zeroes[m, T, p, D, s] = [0.]
                    nu_ = nus_[1, L, m, T, mu, p, D, s]
                    for t in range(1, t_nos):
                        if nu_[t-1] > 0 and nu_[t] < 0:
                            zeroes[m, T, p, D, s].append(
                                (ts_[t-1] + ts_[t])/2.)
                        elif nu_[t] == 0.:
                            zeroes[m, T, p, D, s].append(ts_[t])
                    z = zeroes[m, T, p, D, s]
                    freqs[m, T, p, D, s] = []
                    for i in range(1, len(z)):
                        freqs[m, T, p, D, s].append(1. / (z[i] - z[i-1]))
freqs_ = np.empty((len(ms),
                   len(Ts),
                   len(places),
                   len(Deltas),
                   4))
freqs_m = np.empty((len(Ts),
                    len(places),
                    len(Deltas),
                    4))
freqs_T = np.empty((len(ms),
                    len(places),
                    len(Deltas),
                    4))
freqs_p = np.empty((len(ms),
                    len(Ts),
                    len(Deltas),
                    4))
freqs_mT = np.empty((len(places),
                     len(Deltas),
                     4))
freqs_mp = np.empty((len(Ts),
                     len(Deltas),
                     4))
freqs_Tp = np.empty((len(ms),
                     len(Deltas),
                     4))
freqs_mTp = np.empty((len(Deltas),
                      4))

for D in range(len(Deltas)):
    for m in range(len(ms)):
        for T in range(len(Ts)):
            for p in range(len(places)):
                freqs__ = np.empty(0)
                for s in range(len(scales)):
                    freqs__ = np.concatenate((freqs__, freqs[m, T, p, D, s]))
                freqs_[m, T, p, D] = [np.median(freqs__), np.std(freqs__),
                                      min(freqs__), max(freqs__)]

    for T in range(len(Ts)):
        for p in range(len(places)):
            freqs__ = np.empty(0)
            for s in range(len(scales)):
                for m in range(len(ms)):
                    freqs__ = np.concatenate((freqs__, freqs[m, T, p, D, s]))
            freqs_m[T, p, D] = [np.median(freqs__), np.std(freqs__),
                                min(freqs__), max(freqs__)]

    for m in range(len(ms)):
        for p in range(len(places)):
            freqs__ = np.empty(0)
            for s in range(len(scales)):
                for T in range(len(Ts)):
                    freqs__ = np.concatenate((freqs__, freqs[m, T, p, D, s]))
            freqs_T[m, p, D] = [np.median(freqs__), np.std(freqs__),
                                min(freqs__), max(freqs__)]

    for m in range(len(ms)):
        for T in range(len(Ts)):
            freqs__ = np.empty(0)
            for s in range(len(scales)):
                for p in range(len(places)):
                    freqs__ = np.concatenate((freqs__, freqs[m, T, p, D, s]))
            freqs_p[m, T, D] = [np.median(freqs__), np.std(freqs__),
                                min(freqs__), max(freqs__)]

    for p in range(len(places)):
        freqs__ = np.empty(0)
        for s in range(len(scales)):
            for m in range(len(ms)):
                for T in range(len(Ts)):
                    freqs__ = np.concatenate((freqs__, freqs[m, T, p, D, s]))
        freqs_mT[p, D] = [np.median(freqs__), np.std(freqs__), min(freqs__),
                          max(freqs__)]

    for T in range(len(Ts)):
        freqs__ = np.empty(0)
        for s in range(len(scales)):
            for m in range(len(ms)):
                for p in range(len(places)):
                    freqs__ = np.concatenate((freqs__, freqs[m, T, p, D, s]))
        freqs_mp[T, D] = [np.median(freqs__), np.std(freqs__), min(freqs__),
                          max(freqs__)]

    for m in range(len(ms)):
        freqs__ = np.empty(0)
        for s in range(len(scales)):
            for T in range(len(Ts)):
                for p in range(len(places)):
                    freqs__ = np.concatenate((freqs__, freqs[m, T, p, D, s]))
        freqs_Tp[m, D] = [np.median(freqs__), np.std(freqs__), min(freqs__),
                          max(freqs__)]

    freqs__ = np.empty(0)
    for s in range(len(scales)):
        for m in range(len(ms)):
            for T in range(len(Ts)):
                for p in range(len(places)):
                    freqs__ = np.concatenate((freqs__, freqs[m, T, p, D, s]))
    freqs_mTp[D] = [np.median(freqs__), np.std(freqs__), min(freqs__),
                    max(freqs__)]

# creating a figure for plotting phase frequency median
fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
ax.set_xscale("log")
ax.set_xlim(Deltas[0], Deltas[-1])
ax.set_xlabel(r"$\Delta$")
ax.set_yscale("log")


# plotting each individual frequency median over time
ax.set_ylabel(r"$f_{\tau}(\Delta)$")
c = blinds[3]
for m, m_ in enumerate(ms):
    for T, T_ in enumerate(Ts):
        T_s = T_strings[T]
        for p, pl_ in enumerate(places):
            freqs__ = freqs_[m, T, p, :, 0]
            ax.axline((Deltas[0], freqs__[0]), (Deltas[0]*2., freqs__[0]*2),
                      linestyle="--", color=c)
            ax.plot(Deltas, freqs__, "-o", color=c)
            ax.relim()
            ax.autoscale(axis="y")
            fig.suptitle(r"$L=%s$, $m=%s$, $T=%s$, $\mu=%s$, %s" %
                         (L_, m_, T_s, mu_, pl_))
            fig.savefig("%sx%s/f_τ (L = %s, m = %s, T = %s, μ = %s, %s).pdf" %
                        (L_, L_, L_, m_, T_, mu_, pl_), bbox_inches="tight")
            for line in ax.get_lines():
                line.remove()

            print(m_, T_, pl_)

# plotting frequency median over time with varying energy splittings and
# impurity locations on each plot
lines, labels = [], []
for m, m_ in enumerate(ms):
    for p, pl_ in enumerate(places):
        lines.append(Line2D([0], [0], color=colours[p + m*len(places)],
                            marker="o"))
        labels.append(r"$m=%s$, %s" % (m_, pl_))
legend = fig.legend(lines, labels, loc="center left", bbox_to_anchor=(1, .5))
for T, T_ in enumerate(Ts):
    T_s = T_strings[T]
    for m in range(len(ms)):
        for p in range(len(places)):
            c = colours[p + m*len(places)]
            freqs__ = freqs_[m, T, p, :, 0]
            ax.axline((Deltas[0], freqs__[0]), (Deltas[0]*2., freqs__[0]*2),
                      linestyle="--", color=c)
            ax.plot(Deltas, freqs__, "-o", color=c)
    ax.relim()
    ax.autoscale(axis="y")
    fig.suptitle(r"$L=%s$, $T=%s$, $\mu=%s$" % (L_, T_s, mu_))
    fig.savefig("m, p/m, p f_τ (L = %s, T = %s, μ = %s).pdf" % (L_, T_, mu_),
                bbox_inches="tight")
    for line in ax.get_lines():
        line.remove()

    print("m, p", T_)
legend.remove()
del legend

# plotting frequency median over time with varying temperatures on each plot
lines, labels = [], []
for T, T_s in enumerate(T_strings):
    lines.append(Line2D([0], [0], color=colours[T], marker="o"))
    labels.append(r"$T=%s$" % T_s)
legend = fig.legend(lines, labels, loc="center left", bbox_to_anchor=(1, .5))
for m, m_ in enumerate(ms):
    for p, pl_ in enumerate(places):
        for T in range(len(Ts)):
            c = colours[T]
            freqs__ = freqs_[m, T, p, :, 0]
            ax.axline((Deltas[0], freqs__[0]), (Deltas[0]*2., freqs__[0]*2),
                      linestyle="--", color=c)
            ax.plot(Deltas, freqs__, "-o", color=c)
        ax.relim()
        ax.autoscale(axis="y")
        fig.suptitle(r"$L=%s$, $m=%s$, $\mu=%s$, %s" % (L_, m_, mu_, pl_))
        fig.savefig("T/T f_τ (L = %s, m = %s, μ = %s, %s).pdf" %
                    (L_, m_, mu_, pl_), bbox_inches="tight")
        for line in ax.get_lines():
            line.remove()

        print("T", m_, pl_)
legend.remove()
del legend


# plotting each individual frequency median over time and energy splitting
ax.set_ylabel(r"$f_{\tau,m}(\Delta)$")
c = blinds[3]
for T, T_ in enumerate(Ts):
    T_s = T_strings[T]
    for p, pl_ in enumerate(places):
        freqs__ = freqs_m[T, p, :, 0]
        ax.axline((Deltas[0], freqs__[0]), (Deltas[0]*2., freqs__[0]*2),
                  linestyle="--", color=c)
        ax.plot(Deltas, freqs__, "-o", color=c)
        ax.relim()
        ax.autoscale(axis="y")
        fig.suptitle(r"$L=%s$, $T=%s$, $\mu=%s$, %s" % (L_, T_s, mu_, pl_))
        fig.savefig("%sx%s/f_(τ, m) (L = %s, T = %s, μ = %s, %s).pdf" %
                    (L_, L_, L_, T_, mu_, pl_), bbox_inches="tight")
        for line in ax.get_lines():
            line.remove()

        print(T_, pl_)

# plotting frequency median over time and energy splitting with varying
# temperatures on each plot
lines, labels = [], []
for T, T_s in enumerate(T_strings):
    lines.append(Line2D([0], [0], color=colours[T], marker="o"))
    labels.append(r"$T=%s$" % T_s)
legend = fig.legend(lines, labels, loc="center left", bbox_to_anchor=(1, .5))
for p, pl_ in enumerate(places):
    for T in range(len(Ts)):
        c = colours[T]
        freqs__ = freqs_m[T, p, :, 0]
        ax.axline((Deltas[0], freqs__[0]), (Deltas[0]*2., freqs__[0]*2),
                  linestyle="--", color=c)
        ax.plot(Deltas, freqs__, "-o", color=c)
    ax.relim()
    ax.autoscale(axis="y")
    fig.suptitle(r"$L=%s$, $\mu=%s$, %s" % (L_, mu_, pl_))
    fig.savefig("T/T f_(τ, m) (L = %s, μ = %s, %s).pdf" % (L_, mu_, pl_),
                bbox_inches="tight")
    for line in ax.get_lines():
        line.remove()

    print("T", pl_)
legend.remove()
del legend

# plotting frequency median over time and energy splitting with varying
# impurity locations on each plot
lines, labels = [], []
for p, pl_ in enumerate(places):
    lines.append(Line2D([0], [0], color=colours[p], marker="o"))
    labels.append(r"%s" % pl_)
legend = fig.legend(lines, labels, loc="center left", bbox_to_anchor=(1, .5))
for T, T_ in enumerate(Ts):
    T_s = T_strings[T]
    for p in range(len(places)):
        c = colours[p]
        freqs__ = freqs_m[T, p, :, 0]
        ax.axline((Deltas[0], freqs__[0]), (Deltas[0]*2., freqs__[0]*2),
                  linestyle="--", color=c)
        ax.plot(Deltas, freqs__, "-o", color=c)
    ax.relim()
    ax.autoscale(axis="y")
    fig.suptitle(r"$L=%s$, $T=%s$, $\mu=%s$" % (L_, T_s, mu_))
    fig.savefig("p/p f_(τ, m) (L = %s, T = %s, μ = %s).pdf" % (L_, T_, mu_),
                bbox_inches="tight")
    for line in ax.get_lines():
        line.remove()

    print("p", T_)
legend.remove()
del legend


# plotting each individual frequency median over time and temperature
ax.set_ylabel(r"$f_{\tau,T}(\Delta)$")
c = blinds[3]
for m, m_ in enumerate(ms):
    for p, pl_ in enumerate(places):
        freqs__ = freqs_T[m, p, :, 0]
        ax.axline((Deltas[0], freqs__[0]), (Deltas[0]*2., freqs__[0]*2),
                  linestyle="--", color=c)
        ax.plot(Deltas, freqs__, "-o", color=c)
        ax.relim()
        ax.autoscale(axis="y")
        fig.suptitle(r"$L=%s$, $m=%s$, $\mu=%s$, %s" % (L_, m_, mu_, pl_))
        fig.savefig("%sx%s/f_(τ, T) (L = %s, m = %s, μ = %s, %s).pdf" %
                    (L_, L_, L_, m_, mu_, pl_), bbox_inches="tight")
        for line in ax.get_lines():
            line.remove()

        print(m_, pl_)

# plotting frequency median over time and temperature with varying energy
# splittings and impurity location on each plot
lines, labels = [], []
for m, m_ in enumerate(ms):
    for p, pl_ in enumerate(places):
        lines.append(Line2D([0], [0], color=colours[p + m*len(places)],
                            marker="o"))
        labels.append(r"$m=%s$, %s" % (m_, pl_))
legend = fig.legend(lines, labels, loc="center left", bbox_to_anchor=(1, .5))
for m in range(len(ms)):
    for p in range(len(places)):
        c = colours[p + m*len(places)]
        freqs__ = freqs_T[m, p, :, 0]
        ax.axline((Deltas[0], freqs__[0]), (Deltas[0]*2., freqs__[0]*2),
                  linestyle="--", color=c)
        ax.plot(Deltas, freqs__, "-o", color=c)
ax.relim()
ax.autoscale(axis="y")
fig.suptitle(r"$L=%s$, $\mu=%s$" % (L_, mu_))
fig.savefig("m, p/m, p f_(τ, T) (L = %s, μ = %s).pdf" % (L_, mu_),
            bbox_inches="tight")
for line in ax.get_lines():
    line.remove()

print("m, p")
legend.remove()
del legend


# plotting each individual frequency median over time and impurity location
ax.set_ylabel(r"$f_{\tau,p}(\Delta)$")
c = blinds[3]
for m, m_ in enumerate(ms):
    for T, T_ in enumerate(Ts):
        T_s = T_strings[T]
        freqs__ = freqs_p[m, T, :, 0]
        ax.axline((Deltas[0], freqs__[0]), (Deltas[0]*2., freqs__[0]*2),
                  linestyle="--", color=c)
        ax.plot(Deltas, freqs__, "-o", color=c)
        ax.relim()
        ax.autoscale(axis="y")
        fig.suptitle(r"$L=%s$, $m=%s$, $T=%s$, $\mu=%s$" % (L_, m_, T_s, mu_))
        fig.savefig("%sx%s/f_(τ, p) (L = %s, m = %s, T = %s, μ = %s).pdf" %
                    (L_, L_, L_, m_, T_, mu_), bbox_inches="tight")
        for line in ax.get_lines():
            line.remove()

        print(m_, T_)

# plotting frequency median over time and impurity location with varying energy
# splittings on each plot
lines, labels = [], []
for m, m_ in enumerate(ms):
    lines.append(Line2D([0], [0], color=colours[m], marker="o"))
    labels.append(r"$m=%s$" % m_)
legend = fig.legend(lines, labels, loc="center left", bbox_to_anchor=(1, .5))
for T, T_ in enumerate(Ts):
    T_s = T_strings[T]
    for m in range(len(ms)):
        c = colours[m]
        freqs__ = freqs_p[m, T, :, 0]
        ax.axline((Deltas[0], freqs__[0]), (Deltas[0]*2., freqs__[0]*2),
                  linestyle="--", color=c)
        ax.plot(Deltas, freqs__, "-o", color=c)
    ax.relim()
    ax.autoscale(axis="y")
    fig.suptitle(r"$L=%s$, $T=%s$, $\mu=%s$" % (L_, T_s, mu_))
    fig.savefig("m/m f_(τ, p) (L = %s, T = %s, μ = %s).pdf" % (L_, T_, mu_),
                bbox_inches="tight")
    for line in ax.get_lines():
        line.remove()

    print("m", T_)
legend.remove()
del legend

# plotting frequency median over time and impurity location with varying
# temperatures on each plot
lines, labels = [], []
for T, T_s in enumerate(T_strings):
    lines.append(Line2D([0], [0], color=colours[T], marker="o"))
    labels.append(r"$T=%s$" % T_s)
legend = fig.legend(lines, labels, loc="center left", bbox_to_anchor=(1, .5))
for m, m_ in enumerate(ms):
    for T in range(len(Ts)):
        c = colours[T]
        freqs__ = freqs_p[m, T, :, 0]
        ax.axline((Deltas[0], freqs__[0]), (Deltas[0]*2., freqs__[0]*2),
                  linestyle="--", color=c)
        ax.plot(Deltas, freqs__, "-o", color=c)
    ax.relim()
    ax.autoscale(axis="y")
    fig.suptitle(r"$L=%s$, $m=%s$, $\mu=%s$" % (L_, m_, mu_))
    fig.savefig("T/T f_(τ, p) (L = %s, m = %s, μ = %s).pdf" % (L_, m_, mu_),
                bbox_inches="tight")
    for line in ax.get_lines():
        line.remove()

    print("T", m_)
legend.remove()
del legend


# plotting each individual frequency median over time, energy splitting, and
# temperature
ax.set_ylabel(r"$f_{\tau,m,T}(\Delta)$")
c = blinds[3]
for p, pl_ in enumerate(places):
    freqs__ = freqs_mT[p, :, 0]
    ax.axline((Deltas[0], freqs__[0]), (Deltas[0]*2., freqs__[0]*2),
              linestyle="--", color=c)
    ax.plot(Deltas, freqs__, "-o", color=c)
    ax.relim()
    ax.autoscale(axis="y")
    fig.suptitle(r"$L=%s$, $\mu=%s$, %s" % (L_, mu_, pl_))
    fig.savefig("%sx%s/f_(τ, m, T) (L = %s, μ = %s, %s).pdf" %
                (L_, L_, L_, mu_, pl_), bbox_inches="tight")
    for line in ax.get_lines():
        line.remove()

    print(pl_)

# plotting frequency median over time, energy splitting, and temperature with
# varying impurity locations on each plot
lines, labels = [], []
for p, pl_ in enumerate(places):
    lines.append(Line2D([0], [0], color=colours[p], marker="o"))
    labels.append(r"%s" % pl_)
legend = fig.legend(lines, labels, loc="center left", bbox_to_anchor=(1, .5))
for p in range(len(places)):
    c = colours[p]
    freqs__ = freqs_mT[p, :, 0]
    ax.axline((Deltas[0], freqs__[0]), (Deltas[0]*2., freqs__[0]*2),
              linestyle="--", color=c)
    ax.plot(Deltas, freqs__, "-o", color=c)
ax.relim()
ax.autoscale(axis="y")
fig.suptitle(r"$L=%s$, $\mu=%s$" % (L_, mu_))
fig.savefig("p/p f_(τ, m, T) (L = %s, μ = %s).pdf" % (L_, mu_),
            bbox_inches="tight")
for line in ax.get_lines():
    line.remove()

print("p")
legend.remove()
del legend


# plotting each individual frequency median over time, energy splitting, and
# impurity location
ax.set_ylabel(r"$f_{\tau,m,p}(\Delta)$")
c = blinds[3]
for T, T_ in enumerate(Ts):
    T_s = T_strings[T]
    freqs__ = freqs_mp[T, :, 0]
    ax.axline((Deltas[0], freqs__[0]), (Deltas[0]*2., freqs__[0]*2),
              linestyle="--", color=c)
    ax.plot(Deltas, freqs__, "-o", color=c)
    ax.relim()
    ax.autoscale(axis="y")
    fig.suptitle(r"$L=%s$, $T=%s$, $\mu=%s$" % (L_, T_s, mu_))
    fig.savefig("%sx%s/f_(τ, m, p) (L = %s, T = %s, μ = %s).pdf" %
                (L_, L_, L_, T_, mu_), bbox_inches="tight")
    for line in ax.get_lines():
        line.remove()

    print(T_)

# plotting frequency median over time, energy splitting, and impurity location
# with varying temperatures on each plot
lines, labels = [], []
for T, T_s in enumerate(T_strings):
    lines.append(Line2D([0], [0], color=colours[T], marker="o"))
    labels.append(r"$T=%s$" % T_s)
legend = fig.legend(lines, labels, loc="center left", bbox_to_anchor=(1, .5))
for T in range(len(Ts)):
    c = colours[T]
    freqs__ = freqs_mp[T, :, 0]
    ax.axline((Deltas[0], freqs__[0]), (Deltas[0]*2., freqs__[0]*2),
              linestyle="--", color=c)
    ax.plot(Deltas, freqs__, "-o", color=c)
ax.relim()
ax.autoscale(axis="y")
fig.suptitle(r"$L=%s$, $\mu=%s$" % (L_, mu_))
fig.savefig("T/T f_(τ, m, p) (L = %s, μ = %s).pdf" % (L_, mu_),
            bbox_inches="tight")
for line in ax.get_lines():
    line.remove()

print("T")
legend.remove()
del legend


# plotting each indiviual frequency median over time, temperature, and impurity
# location
ax.set_ylabel(r"$f_{\tau,T,p}(\Delta)$")
c = blinds[3]
for m, m_ in enumerate(ms):
    freqs__ = freqs_Tp[m, :, 0]
    ax.axline((Deltas[0], freqs__[0]), (Deltas[0]*2., freqs__[0]*2),
              linestyle="--", color=c)
    ax.plot(Deltas, freqs__, "-o", color=c)
    ax.relim()
    ax.autoscale(axis="y")
    fig.suptitle(r"$L=%s$, $m=%s$, $\mu=%s$" % (L_, m_, mu_))
    fig.savefig("%sx%s/f_(τ, T, p) (L = %s, m = %s, μ = %s).pdf" %
                (L_, L_, L_, m_, mu_), bbox_inches="tight")
    for line in ax.get_lines():
        line.remove()

    print(m_)

# plotting frequency median over time, temperature, and impurity location with
# varying energy splittings on each plot
lines, labels = [], []
for m, m_ in enumerate(ms):
    lines.append(Line2D([0], [0], color=colours[m], marker="o"))
    labels.append(r"$m=%s$" % m_)
legend = fig.legend(lines, labels, loc="center left", bbox_to_anchor=(1, .5))
for m in range(len(ms)):
    c = colours[m]
    freqs__ = freqs_Tp[m, :, 0]
    ax.axline((Deltas[0], freqs__[0]), (Deltas[0]*2., freqs__[0]*2),
              linestyle="--", color=c)
    ax.plot(Deltas, freqs__, "-o", color=c)
ax.relim()
ax.autoscale(axis="y")
fig.suptitle(r"$L=%s$, $\mu=%s$" % (L_, mu_))
fig.savefig("m/m f_(τ, T, p) (L = %s, μ = %s).pdf" % (L_, mu_),
            bbox_inches="tight")
for line in ax.get_lines():
    line.remove()

print("m")
legend.remove()
del legend


# plotting each individual frequency median over time, energy splitting,
# temperature, and impurity location
ax.set_ylabel(r"$f_{\tau,m,T,p}(\Delta)$")
c = blinds[3]
freqs__ = freqs_mTp[:, 0]
ax.axline((Deltas[0], freqs__[0]), (Deltas[0]*2., freqs__[0]*2),
          linestyle="--", color=c)
ax.plot(Deltas, freqs__, "-o", color=c)
ax.relim()
ax.autoscale(axis="y")
fig.suptitle(r"$L=%s$, $\mu=%s$" % (L_, mu_))
fig.savefig("%sx%s/f_(τ, m, T, p) (L = %s, μ = %s).pdf" % (L_, L_, L_, mu_),
            bbox_inches="tight")

plt.close(fig)
del fig, ax
