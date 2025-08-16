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


def rescale(ax):
    """Re-scales the y-axis."""
    ax.relim()
    ax.autoscale(axis="y")

def remove_lines(ax):
    """Remove all lines from the figure."""
    for a in ax.reshape(-1):
        for line in a.get_lines():
            line.remove()

def make_legend(fig):
    """Make a legend for subplots."""
    return fig.legend(loc="center left", bbox_to_anchor=(1, .5))


def rescale(ax):
    """Re-scales the y-axis."""
    ax.relim()
    ax.autoscale(axis="y")

def remove_lines(ax):
    """Remove all lines from the figure."""
    for a in ax.reshape(-1):
        for line in a.get_lines():
            line.remove()

def make_legend(fig):
    """Make a legend for subplots."""
    return fig.legend(loc="center left", bbox_to_anchor=(1, .5))


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
np.save("Decoherence data", nus)

# splitting ν into radial and polar parts
nus = np.array([np.abs(nus), np.angle(nus)])


# creating a figure for plotting |ν|, θ
fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True,
                       constrained_layout=True)
ax[0].set_yscale("log")
ax[0].set_ylabel(r"$|\nu(\tau)|$")
ax[1].set_xlabel(r"$\tau$")
ax[1].set_ylim(-1.1 * np.pi, 1.1 * np.pi)
ax[1].set_yticks([-np.pi, 0., np.pi], labels=[r"$-\pi$", "0", r"$\pi$"])
ax[1].set_ylabel(r"$\theta(\tau)$")

for D, (D_, D_s, ts_) in enumerate(zip(Deltas, Delta_strings, ts[:, 0])):
    ax[1].set_xlim(ts_[0], ts_[-1])
    ax[1].set_xticks(np.linspace(ts_[0], ts_[-1], 5))

    # each parameter combination
    for (T, (T_, T_s)), (L, L_), (m, m_), (mu, mu_), (p, pl_) in it.product(
        enumerate(zip(Ts, T_strings)), enumerate(lengths), enumerate(ms),
        enumerate(mus), enumerate(places)):
        for i in range(2):
            ax[i].plot(ts_, nus[i, L, m, T, mu, p, D, 0], color=blinds[i])
        rescale(ax[0])
        fig.suptitle(
            rf"$L={L_}$, $m={m_}$, $T={T_s}$, $\mu={mu_}$, {pl_}, $\Delta={D_s}$"
            )
        fig.savefig(
            f"{L_}x{L_}/abs(ν), θ (L = {L_}, m = {m_}, T = {T_}, μ = {mu_}, {pl_}, Δ = {D_}).pdf",
            bbox_inches="tight")
        remove_lines(ax)
        print(D_, T_, L_, m_, mu_, pl_)

    # varying lengths
    for (T, (T_, T_s)), (m, m_), (mu, mu_), (p, pl_) in it.product(
        enumerate(zip(Ts, T_strings)), enumerate(ms), enumerate(mus),
        enumerate(places)):
        for i, (L, L_) in it.product(range(2), enumerate(lengths)):
            ax[i].plot(ts_, nus[i, L, m, T, mu, p, D, 0],
                       color=colours[L % len(colours)],
                       linestyle=styles[int(L/len(colours))])
        rescale(ax[0])
        fig.suptitle(
            rf"$m={m_}$, $T={T_s}$, $\mu={mu_}$, {pl_}, $\Delta={D_s}$")
        legend = make_legend(fig)
        fig.savefig(
            f"L/L abs(ν), θ (m = {m_}, T = {T_}, μ = {mu_}, {pl_}, Δ = {D_}).pdf",
            bbox_inches="tight")
        remove_lines(ax)
        legend.remove()
        print("L", D_, T_, m_, mu_, pl_)

    # varying energy splittings and impurity locations
    for (T, (T_, T_s)), (L, L_), (mu, mu_) in it.product(
        enumerate(zip(Ts, T_strings)), enumerate(lengths), enumerate(mus)):
        for i, m, p in it.product(range(2), range(len(ms)), range(len(places))):
            ax[i].plot(ts_, nus[i, L, m, T, mu, p, D, 0],
                       color=colours[p + m*len(places)])
        rescale(ax[0])
        fig.suptitle(rf"$L={L_}$, $T={T_s}$, $\mu={mu_}$, $\Delta={D_s}$")
        legend = make_legend(fig)
        fig.savefig(
            f"m, p/m, p abs(ν), θ (L = {L_}, T = {T_}, μ = {mu_}, Δ = {D_}).pdf",
            bbox_inches="tight")
        remove_lines(ax)
        legend.remove()
        print("m, p", D_, T_, L_, mu_)

    # varying temperatures
    for (L, L_), (m, m_), (mu, mu_), (p, pl_) in it.product(
        enumerate(lengths), enumerate(ms), enumerate(mus), enumerate(places)):
        for i, T in it.product(range(2), range(len(Ts))):
            ax[i].plot(ts_, nus[i, L, m, T, mu, p, D, 0], color=colours[T])
        rescale(ax[0])
        fig.suptitle(rf"$L={L_}$, $m={m_}$, $\mu={mu_}$, {pl_}, $\Delta={D_s}$")
        legend = make_legend(fig)
        fig.savefig(
            f"T/T abs(ν), θ (L = {L_}, m = {m_}, μ = {mu_}, {pl_}, Δ = {D_}).pdf",
            bbox_inches="tight")
        remove_lines(ax)
        legend.remove()
        print("T", D_, L_, m_, mu_, pl_)

plt.close(fig)


# creating a figure for plotting |ν|, -log|ν|
nus = np.array([nus[0], -np.log10(nus[0]), nus[1]])
fig, ax = plt.subplots(2, 2, figsize=(12, 6), sharex="col",
                       constrained_layout=True)
ax[0, 0].set_ylabel(r"$|\nu(\tau)|$")
ax[1, 0].set_xlabel(r"$\tau$")
ax[1, 0].set_ylabel(r"$-\log(|\nu(\tau)|)$")
ax[1, 1].set_xscale("log")
for s in range(len(scales)):
    ax[1, s].set_xlabel(r"$\tau$")
    for i in range(2):
        ax[i, s].set_yscale("log")

for D, (D_, D_s, ts_) in enumerate(zip(Deltas, Delta_strings, ts)):
    for i in range(2):
        ax[i, 0].set_xlim(ts_[i, 0], ts_[i, -1])
    ax[1, 0].set_xticks(np.linspace(ts_[0, 0], ts_[0, -1], 5))

    # each parameter combination
    for (T, (T_, T_s)), (L, L_), (m, m_), (mu, mu_), (p, pl_) in it.product(
        enumerate(zip(Ts, T_strings)), enumerate(lengths), enumerate(ms),
        enumerate(mus), enumerate(places)):
        for i, s in it.product(range(2), range(len(scales))):
            ax[i, s].plot(ts_[s], nus[i, L, m, T, mu, p, D, s],
                          color=blinds[2*i])
            rescale(ax[i, s])
        fig.suptitle(
            rf"$L={L_}$, $m={m_}$, $T={T_s}$, $\mu={mu_}$, {pl_}, $\Delta={D_s}$"
            )
        fig.savefig(
            f"{L_}x{L_}/abs(ν), -log(abs(ν)) (L = {L_}, m = {m_}, T = {T_}, μ = {mu_}, {pl_}, Δ = {D_}).pdf",
            bbox_inches="tight")
        remove_lines(ax)
        print(D_, T_, L_, m_, mu_, pl_)

    # varying lengths
    for (T, (T_, T_s)), (m, m_), (mu, mu_), (p, pl_) in it.product(
        enumerate(zip(Ts, T_strings)), enumerate(ms), enumerate(mus),
        enumerate(places)):
        for i, s in it.product(range(2), range(len(scales))):
            for L in range(len(lengths)):
                ax[i, s].plot(ts_[s], nus[i, L, m, T, mu, p, D, s],
                              color=colours[L % len(colours)],
                              linestyle=styles[int(L/(len(colours)))])
                rescale(ax[i, s])
        fig.suptitle(rf"$m={m_}$, $T={T_s}$, $\mu={mu_}$, {pl_}, $\Delta={D_s}$"
                     )
        legend = make_legend(fig)
        fig.savefig(
            f"L/L abs(ν), -log(abs(ν)) (m = {m_}, T = {T_}, μ = {mu_}, {pl_}, Δ = {D_}).pdf",
            bbox_inches="tight")
        remove_lines(ax)
        legend.remove()
        print("L", D_, T_, m_, mu_, pl_)
    
    # varying energy splittings and impurity locations
    for (T, (T_, T_s)), (L, L_), (mu, mu_) in it.product(
        enumerate(zip(Ts, T_strings)), enumerate(lengths), enumerate(mus)):
        for i, s in it.product(range(2), range(len(scales))):
            for m, p in it.product(range(len(ms)), range(len(places))):
                ax[i, s].plot(ts_[s], nus[i, L, m, T, mu, p, D, s],
                              color=colours[p + m*len(places)])
            rescale(ax[i, s])
        fig.suptitle(rf"$L={L_}$, $T={T_s}$, $\mu={mu_}$, $\Delta={D_s}$")
        legend = make_legend(fig)
        fig.savefig(
            f"m, p/m, p abs(ν), -log(abs(ν)) (L = {L_}, T = {T_}, μ = {mu_}, Δ = {D_}).pdf",
            bbox_inches="tight")
        remove_lines(ax)
        legend.remove()
        print("m, p", D_, T_, L_, mu_)
    
    # varying temperatures
    for (L, L_), (m, m_), (mu, mu_), (p, pl_) in it.product(
        enumerate(lengths), enumerate(ms), enumerate(mus), enumerate(places)):
        for i, s in it.product(range(2), range(len(scales))):
            for T in range(len(Ts)):
                ax[i, s].plot(ts_[s], nus[i, L, m, T, mu, p, D, s],
                              color=colours[T])
            rescale(ax[i, s])
        fig.suptitle(rf"$L={L_}$, $m={m_}$, $\mu={mu_}$, {pl_}, $\Delta={D_s}$")
        legend = make_legend(fig)
        fig.savefig(
            f"T/T abs(ν), -log(abs(ν)) (L = {L_}, m = {m_}, μ = {mu_}, {pl_}, Δ = {D_s}).pdf",
            bbox_inches="tight")
        remove_lines(ax)
        legend.remove()
        print("T", D_, L_, m_, mu_, pl_)

plt.close(fig)


# creating a figure for plotting |ν|, θ, -log|ν| with varying coupling strengths
# on each plot
nus[1:] = nus[:0:-1]
fig, ax = plt.subplots(3, 2, figsize=(12, 9), sharex=True, sharey="row",
                       constrained_layout=True)
ax[0, 0].set_yscale("log")
ax[0, 0].set_ylabel(r"$|\nu(\tau)|$")
ax[1, 0].set_ylim(-1.1 * np.pi, 1.1 * np.pi)
ax[1, 0].set_yticks([-np.pi, 0., np.pi], labels=[r"$-\pi$", "0", r"$\pi$"])
ax[1, 0].set_ylabel(r"$\theta(\tau)$")
ax[2, 0].set_yscale("log")
ax[2, 0].set_ylabel(r"$-\log(|\nu(\tau)|)$")
ax[2, 1].set_xscale("log")
for s in range(len(scales)):
    ax[2, s].set_xlim(np.amin(ts[:, s]), np.amax(ts[:, s]))
    ax[2, s].set_xlabel(r"$\tau$")


# plotting |ν|, θ, -log|ν| with varying coupling strengths on each plot
for (T, (T_, T_s)), (L, L_), (m, m_), (mu, mu_), (p, pl_) in it.product(
    enumerate(zip(Ts, T_strings)), enumerate(lengths), enumerate(ms),
    enumerate(mus), enumerate(places)):
    for i, s, D in it.product(range(3), range(len(scales)), range(len(Deltas))):
        ax[i, s].plot(ts[D, s], nus[i, L, m, T, mu, p, D, s], color=colours[D])
    for a in ax[::2].reshape(-1):
        rescale(a)
    fig.suptitle(rf"$L={L_}$, $m={m_}$, $T={T_s}$, $\mu={mu_}$, {pl_}")
    legend = make_legend(fig)
    fig.savefig(
        f"Δ/Δ abs(ν), θ, -log(abs(ν)) (L = {L_}, m = {m_}, T = {T_}, μ = {mu_}, {pl_}).pdf",
        bbox_inches="tight")
    remove_lines(ax)
    legend.remove()
    print("Δ", T_, L_, m_, mu_, pl_)
plt.close(fig)


# defining useful variables for plotting grids of graphs
L, mu = -1, 0
L_, mu_ = lengths[L], mus[mu]


# creating a figure for plotting |ν|, θ, -log|ν| separately with varying energy
# splittings, impurity locations, and temperatures
fig, ax = plt.subplots(len(ms)*len(places), len(Ts),
                       figsize=(5*len(Ts), 3*len(ms)*len(places)), sharex=True,
                       constrained_layout=True)
for T, T_s in enumerate(T_strings):
    ax[0, T].set_title(rf"$T={T_s}$")
    ax[-1, T].set_xlabel(r"$\tau$")

for D, (D_, D_s, ts_) in enumerate(zip(Deltas, Delta_strings, ts)):
    fig.suptitle(rf"$L={L_}$, $\mu={mu_}$, $\Delta={D_s}$")

    # |ν|
    ax[0, 0].set_xscale("linear")
    ax[0, 0].set_xlim(ts_[0, 0], ts_[0, -1])
    ax[0, 0].set_xticks(np.linspace(ts_[0, 0], ts_[0, -1], 5))
    for i, ((m, m_), (p, pl_)) in enumerate(it.product(enumerate(ms),
                                                       enumerate(places))):
        ax[i, 0].set_ylabel(rf"$|\nu(\tau)|$ ($m={m_}$, {pl_})")
        for T in range(len(Ts)):
            ax[i, T].set_yscale("log")
            ax[i, T].plot(ts_[0], nus[0, L, m, T, mu, p, D, 0], color=colours[T])
            rescale(ax[i, T])
    fig.savefig(f"m, p, T/m, p, T abs(ν) (L = {L_}, μ = {mu_}, Δ = {D_}).pdf")
    remove_lines(ax)

    # θ
    for i, ((m, m_), (p, pl_)) in enumerate(it.product(enumerate(ms),
                                                       enumerate(places))):
        ax[i, 0].set_ylabel(rf"$\theta(\tau)$ ($m={m_}$, {pl_})")
        for T in range(len(Ts)):
            ax[i, T].set_yscale("linear")
            ax[i, T].set_ylim(-1.1 * np.pi, 1.1 * np.pi)
            ax[i, T].set_yticks([-np.pi, 0., np.pi], labels=[])
            ax[i, T].plot(ts_[0], nus[1, L, m, T, mu, p, D, 0], color=colours[T])
        ax[i, 0].set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
    fig.savefig(f"m, p, T/m, p, T θ (L = {L_}, μ = {mu_}, Δ = {D_}).pdf")
    remove_lines(ax)

    # -log|ν|
    ax[0, 0].set_xscale("log")
    ax[0, 0].set_xlim(ts_[1, 0], ts_[1, -1])
    for i, ((m, m_), (p, pl_)) in enumerate(it.product(enumerate(ms),
                                                       enumerate(places))):
        ax[i, 0].set_ylabel(rf"$-\log(|\nu(\tau)|)$ ($m={m_}$, {pl_})")
        for T in range(len(Ts)):
            ax[i, T].set_yscale("log")
            ax[i, T].plot(ts_[1], nus[2, L, m, T, mu, p, D, 1], color=colours[T])
            rescale(ax[i, T])
    fig.savefig(
        f"m, p, T/m, p, T -log(abs(ν)) (L = {L_}, μ = {mu_}, Δ = {D_}).pdf")
    remove_lines(ax)

    print("m, p, T", D_)
plt.close(fig)


# creating a figure for plotting |ν|, θ, -log|ν| separately with varying
# energy splittings, impurity locations, and coupling strengths
fig, ax = plt.subplots(len(ms)*len(places), len(Deltas),
                       figsize=(5*len(Deltas), 3*len(ms)*len(places)),
                       sharex="col", constrained_layout=True)
for D, D_s in enumerate(Delta_strings):
    ax[0, D].set_title(rf"$\Delta={D_s}$")
    ax[-1, D].set_xlabel(r"$\tau$")

for T, (T_, T_s) in enumerate(zip(Ts, T_strings)):
    fig.suptitle(rf"$L={L_}$, $T={T_s}$, $\mu={mu_}$" % (L_, T_s, mu_))

    # |ν|
    for D, ts_ in enumerate(ts[:, 0]):
        ax[-1, D].set_xscale("linear")
        ax[-1, D].set_xlim(ts_[0], ts_[-1])
        ax[-1, D].set_xticks(np.linspace(ts_[0], ts_[-1], 5))
    for i, ((m, m_), (p, pl_)) in enumerate(it.product(enumerate(ms),
                                                       enumerate(places))):
        ax[i, 0].set_ylabel(rf"$|\nu(\tau)|$ ($m={m_}$, {pl_})")
        for D, ts_ in enumerate(ts[:, 0]):
            ax[i, D].set_yscale("log")
            ax[i, D].plot(ts_, nus[0, L, m, T, mu, p, D, 0], color=colours[D])
            rescale(ax[i, D])
    fig.savefig(f"m, p, Δ/m, p, Δ abs(ν) (L = {L_}, T = {T_}, μ = {mu_}).pdf")
    remove_lines(ax)

    # θ
    for i, ((m, m_), (p, pl_)) in enumerate(it.product(enumerate(ms),
                                                       enumerate(places))):
        ax[i, 0].set_ylabel(rf"$\theta(\tau)$ ($m={m_}$, {pl_})")
        for D, ts_ in enumerate(ts[:, 0]):
            ax[i, D].set_yscale("linear")
            ax[i, D].set_ylim(-1.1 * np.pi, 1.1 * np.pi)
            ax[i, D].set_yticks([-np.pi, 0., np.pi], labels=[])
            ax[i, D].plot(ts_, nus[1, L, m, T, mu, p, D, 0], color=colours[D])
        ax[i, 0].set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
    fig.savefig(f"m, p, Δ/m, p, Δ θ (L = {L_}, T = {T_}, μ = {mu_}).pdf")
    remove_lines(ax)

    # -log|ν|
    for D, ts_ in enumerate(ts[:, 1]):
        ax[-1, D].set_xscale("log")
        ax[-1, D].set_xlim(ts_[0], ts_[-1])
    for i, ((m, m_), (p, pl_)) in enumerate(it.product(enumerate(ms),
                                                       enumerate(places))):
        ax[i, 0].set_ylabel(rf"$-\log(|\nu(\tau)|)$ ($m={m_}$, {pl_})")
        for D, ts_ in enumerate(ts[:, 1]):
            ax[i, D].set_yscale("log")
            ax[i, D].plot(ts_, nus[2, L, m, T, mu, p, D, 1], color=colours[D])
            rescale(ax[i, D])
    fig.savefig(
        f"m, p, Δ/m, p, Δ -log(abs(ν)) (L = {L_}, T = {T_}, μ = {mu_}).pdf")
    remove_lines(ax)

    print("m, p, Δ", T_)
plt.close(fig)


# creating a figure for plotting |ν|, θ, -log|ν| separately with varying
# temperatures and coupling strengths
fig, ax = plt.subplots(len(Ts), len(Deltas), figsize=(5*len(Deltas), 3*len(Ts)),
                       sharex="col", constrained_layout=True)
for D, D_s in enumerate(Delta_strings):
    ax[0, D].set_title(rf"$\Delta={D_s}$")
    ax[-1, D].set_xlabel(r"$\tau$")

for (m, m_), (p, pl_) in it.product(enumerate(ms), enumerate(places)):
    fig.suptitle(rf"$L={L_}$, $m={m_}$, $\mu={mu_}$, {pl_}")

    # |ν|
    for D, ts_ in enumerate(ts[:, 0]):
        ax[-1, D].set_xscale("linear")
        ax[-1, D].set_xlim(ts_[0], ts_[-1])
        ax[-1, D].set_xticks(np.linspace(ts_[0], ts_[-1], 5))
    for T, T_s in enumerate(T_strings):
        ax[T, 0].set_ylabel(rf"$|\nu(\tau)|$ ($T={T_s}$)")
        for D, ts_ in enumerate(ts[:, 0]):
            ax[T, D].set_yscale("log")
            ax[T, D].plot(ts_, nus[0, L, m, T, mu, p, D, 0], color=colours[D])
            rescale(ax[T, D])
    fig.savefig(f"T, Δ/T, Δ abs(ν) (L = {L_}, m = {m_}, μ = {mu_}, {pl_}).pdf")
    remove_lines(ax)

    # θ
    for T, T_s in enumerate(T_strings):
        ax[T, 0].set_ylabel(rf"$\theta(\tau)$ ($T={T_s}$)")
        for D, ts_ in enumerate(ts[:, 0]):
            ax[T, D].set_yscale("linear")
            ax[T, D].set_ylim(-1.1 * np.pi, 1.1 * np.pi)
            ax[T, D].set_yticks([-np.pi, 0., np.pi], labels=[])
            ax[T, D].plot(ts_, nus[1, L, m, T, mu, p, D, 0], color=colours[D])
        ax[T, 0].set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
    fig.savefig(f"T, Δ/T, Δ θ (L = {L_}, m = {m_}, μ = {mu_}, {pl_}).pdf")
    remove_lines(ax)

    # -log|ν|
    for D, ts_ in enumerate(ts[:, 1]):
        ax[-1, D].set_xscale("log")
        ax[-1, D].set_xlim(ts_[0], ts_[-1])
    for T, T_s in enumerate(T_strings):
        ax[T, 0].set_ylabel(rf"$-\log(|\nu(\tau)|)$ ($T={T_s}$)")
        for D, ts_ in enumerate(ts[:, 1]):
            ax[T, D].set_yscale("log")
            ax[T, D].plot(ts_, nus[2, L, m, T, mu, p, D, 1], color=colours[D])
            rescale(ax[T, D])
    fig.savefig(
        f"T, Δ/T, Δ -log(abs(ν)) (L = {L_}, m = {m_}, μ = {mu_}, {pl_}).pdf")
    remove_lines(ax)

    print("T, Δ", m_, pl_)
plt.close(fig)


# Calculating phase frequency median over all combinations of parameters
L, mu = -1, 0
L_, mu_ = lengths[L], mus[mu]

# TODO find a better way of getting frequencies
zeroes = np.array([[[[[[0] + [z for z in [
    ts[D, s, t]
    if nus[1, L, m, T, mu, p, D, s, t] == 0

    else (ts[D, s, t-1] + ts[D, s, t]) / 2.
    if nus[1, L, m, T, mu, p, D, s, t-1] > 0
    and nus[1, L, m, T, mu, p, D, s, t] < 0

    else None

    for t in range(1, t_nos)] if z is not None]
    for s in range(len(scales))]
    for D in range(len(Deltas))]
    for p in range(len(places))]
    for T in range(len(Ts))]
    for m in range(len(ms))], dtype=object)

freqs = np.array([[[[[[1. / (zeroes_mTpDs[t+1] - zeroes_mTpDs[t])
                       for t in range(len(zeroes_mTpDs)-1)]
                       for zeroes_mTpDs in zeroes_mTpD]
                       for zeroes_mTpD in zeroes_mTp]
                       for zeroes_mTp in zeroes_mT]
                       for zeroes_mT in zeroes_m]
                       for zeroes_m in zeroes], dtype=object)

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
