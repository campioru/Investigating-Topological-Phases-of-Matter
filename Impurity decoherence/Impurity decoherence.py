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
import copy


def h_QWZ(Lx, Ly, ω0, m, tx, ty):
    """QWZ Hamiltonian matrix."""
    hm, hx, hy = (np.zeros((2*Lx*Ly, 2*Lx*Ly), dtype=np.complex_) for _ in
                  range(3))
    for x in range(Lx):
        for y in range(Ly):
            hm[x + y*Lx, x + y*Lx] = ω0 + m
            hm[x + y*Lx + Lx*Ly, x + y*Lx + Lx*Ly] = ω0 - m
    for x in range(Lx-1):
        for y in range(Ly):
            (hx[x+1 + y*Lx, x + y*Lx],
             hx[x + y*Lx, x+1 + y*Lx],
             hx[x+1 + y*Lx, x + y*Lx + Lx*Ly],
             hx[x + y*Lx + Lx*Ly, x+1 + y*Lx]) = (tx/2. for _ in range(4))
            (hx[x+1 + y*Lx + Lx*Ly, x + y*Lx],
             hx[x + y*Lx, x+1 + y*Lx + Lx*Ly],
             hx[x+1 + y*Lx + Lx*Ly, x + y*Lx + Lx*Ly],
             hx[x + y*Lx + Lx*Ly, x+1 + y*Lx + Lx*Ly]) = (-tx/2. for _ in
                                                          range(4))
    for x in range(Lx):
        for y in range(Ly-1):
            (hy[x + (y+1)*Lx, x + y*Lx],
             hy[x + y*Lx, x + (y+1)*Lx]) = (ty/2. for _ in range(2))
            (hy[x + (y+1)*Lx, x + y*Lx + Lx*Ly],
             hy[x + (y+1)*Lx + Lx*Ly, x + y*Lx]) = (ty/2.*1j for _ in range(2))
            (hy[x + y*Lx + Lx*Ly, x + (y+1)*Lx],
             hy[x + y*Lx, x + (y+1)*Lx + Lx*Ly]) = (-ty/2.*1j for _ in
                                                    range(2))
            (hy[x + (y+1)*Lx + Lx*Ly, x + y*Lx + Lx*Ly],
             hy[x + y*Lx + Lx*Ly, x + (y+1)*Lx + Lx*Ly]) = (-ty/2. for _ in
                                                            range(2))
    return hm + hx + hy


# initialising system parameters
lengths = np.linspace(3, 29, 14, dtype=int)
ms = np.array([1., 3.])
Ts, Deltas = (10. ** np.linspace(-4., 1., 11) for _ in range(2))
T_strings, Delta_strings = ([r"10^{-4}",
                             r"10^{-3.5}",
                             r"10^{-3}",
                             r"10^{-2.5}",
                             r"10^{-2}",
                             r"10^{-1.5}",
                             r"10^{-1}",
                             r"10^{-0.5}",
                             r"10^{0}",
                             r"10^{0.5}",
                             r"10^{1}"] for _ in range(2))
mus = np.array([0.])
places = ["corner", "edge", "centre"]
poss = np.empty((len(lengths), len(places)), dtype=int)
for L in range(len(lengths)):
    poss[L] = np.array([0, (lengths[L]-1)/2,
                        ((lengths[L]-1)/2)*lengths[L] + (lengths[L]-1)/2])
scales = ["linear", "log"]
t_nos = 201
ts = np.empty((len(Deltas), len(scales), t_nos))
for D in range(len(Deltas)):
    last = 20./Deltas[D]
    ts[D, 0] = np.linspace(0., last, t_nos)
    ts[D, 1] = 10.**np.linspace(-1., np.log10(last), t_nos)
nus = np.empty((len(lengths),
                len(ms),
                len(Ts),
                len(mus),
                len(places),
                len(Deltas),
                len(scales),
                t_nos), dtype=np.complex_)
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
blinds = ["#d95f02", "#e7298a", "#1b9e77"]
styles = ["solid", "dashed"]


# computing decoherence for all combinations of parameters
for L, L_ in enumerate(lengths):
    LxLy = L_**2
    LxLy2 = 2*LxLy
    SeiEtM, e_iDtMdag = (np.empty((LxLy2, LxLy2), dtype=np.complex_) for _
                         in range(2))
    for m, m_ in enumerate(ms):
        h0 = h_QWZ(L_, L_, 0., m_, 1., 1.)
        h0_vals, U = np.linalg.eig(h0)
        h0_vals = np.real(h0_vals)
        eiE_diag = np.exp((1j) * h0_vals)
        Udag = U.T.conj()
        del U
        for p, pl_ in enumerate(places):
            p_ = poss[L, p]
            h1 = copy.copy(h0)
            for D, D_ in enumerate(Deltas):
                h1[p_, p_] = h0[p_, p_] + D_
                h1[p_ + LxLy, p_ + LxLy] = h0[p_ + LxLy, p_ + LxLy] + D_
                h1_vals, W = np.linalg.eig(h1)
                e_iD_diag = np.exp((-1j) * np.real(h1_vals))
                M = np.dot(Udag, W)
                del W
                Mdag = M.T.conj()
                for T, T_ in enumerate(Ts):
                    for mu, mu_ in enumerate(mus):
                        S_diag = (1. + np.exp((h0_vals - mu_) / T_)) ** (-1.)
                        I_S_diag = 1. - S_diag
                        for s, s_ in enumerate(scales):
                            for t in range(t_nos):
                                t_ = ts[s, D, t]
                                for i in range(LxLy2):
                                    SeiEtM[i, :] = (
                                        S_diag[i] * eiE_diag[i]**t_ * M[i, :])
                                    e_iDtMdag[i, :] = (
                                        e_iD_diag[i]**t_ * Mdag[i, :])
                                final = np.dot(SeiEtM, e_iDtMdag)
                                for i in range(LxLy2):
                                    final[i, i] += I_S_diag[i]
                                nus[L, m, T, mu, p, D, s, t] = np.linalg.det(
                                    final)
                        print(L_, m_, pl_, D_, T_, mu_, s_)
del (LxLy, LxLy2, SeiEtM, e_iDtMdag, h0, h0_vals, Udag, eiE_diag, h1, h1_vals,
     M, e_iD_diag, Mdag, I_S_diag, final)

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
legend = fig.legend(lines, labels, loc="center left",
                    bbox_to_anchor=(1, .5))
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
legend = fig.legend(lines, labels, loc="center left",
                    bbox_to_anchor=(1, .5))
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
legend = fig.legend(lines, labels, loc="center left",
                    bbox_to_anchor=(1, .5))
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
legend = fig.legend(lines, labels, loc="center left",
                    bbox_to_anchor=(1, .5))
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
legend = fig.legend(lines, labels, loc="center left",
                    bbox_to_anchor=(1, .5))
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
m_l, p_l, T_l, D_l = len(ms), len(places), len(Ts), len(Deltas)
mp_l = m_l*p_l
L, mu = -1, 0
L_, mu_ = lengths[L], mus[mu]


# creating a figure for plotting |ν|, θ, -log|ν| separately with varying
# energy splittings, impurity locations, and temperatures
fig, ax = plt.subplots(mp_l, T_l, figsize=(5*T_l, 3*mp_l), sharex=True,
                       constrained_layout=True)
for T, T_s in enumerate(T_strings):
    ax[0, T].set_title(r"$T=%s$" % T_s)
    ax[mp_l-1, T].set_xlabel(r"$\tau$")

# plotting |ν| with varying energy splittings, impurity locations, and
# temperatures
for m, m_ in enumerate(ms):
    for p, pl_ in enumerate(places):
        i = p + m*p_l
        ax[i, 0].set_ylabel(r"$|\nu(\tau)|$ ($m=%s$, %s)" % (m_, pl_))
        for T in range(len(Ts)):
            ax[i, T].set_yscale("log")
for D, D_ in enumerate(Deltas):
    D_s = Delta_strings[D]
    ts_ = ts[D, 0]
    ax[0, 0].set_xlim(ts_[0], ts_[-1])
    ax[0, 0].set_xticks(np.linspace(ts_[0], ts_[-1], 5))
    for m in range(m_l):
        for p in range(p_l):
            for T in range(T_l):
                i = p + m*p_l
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
    for i in range(mp_l):
        for j in range(T_l):
            for line in ax[i, j].get_lines():
                line.remove()
    print("m, p, T |ν|", D_)

# plotting θ with varying energy splittings, impurity locations, and
# temperatures
for T in range(T_l):
    for m, m_ in enumerate(ms):
        for p, pl_ in enumerate(places):
            i = p + m*p_l
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
    for m in range(m_l):
        for p in range(p_l):
            for T in range(T_l):
                i = p + m*p_l
                ax[i, T].plot(ts_, nus_[1, L, m, T, mu, p, D, 0],
                              color=colours[T])
    fig.suptitle(r"$L=%s$, $\mu=%s$, $\Delta=%s$" % (L_, mu_, D_s))
    fig.savefig("m, p, T/m, p, T θ (L = %s, μ = %s, Δ = %s).pdf" %
                (L_, mu_, D_))
    for i in range(mp_l):
        for j in range(T_l):
            for line in ax[i, j].get_lines():
                line.remove()

    print("m, p, T θ", D_)

# plotting -log|ν| with varying energy splittings, impurity locations, and
# temperatures
ax[0, 0].set_xscale("log")
for T in range(T_l):
    for m, m_ in enumerate(ms):
        for p, pl_ in enumerate(places):
            i = p + m*p_l
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
    for m in range(m_l):
        for p in range(p_l):
            for T in range(T_l):
                i = p + m*p_l
                ax[i, T].plot(ts_, nus_[2, L, m, T, mu, p, D, 1],
                              color=colours[T])
                ax[i, T].relim()
                ax[i, T].autoscale(axis="y")
    fig.suptitle(r"$L=%s$, $\mu=%s$, $\Delta=%s$" % (L_, mu_, D_s))
    fig.savefig("m, p, T/m, p, T -log(abs(ν)) (L = %s, μ = %s, Δ = %s).pdf" %
                (L_, mu_, D_))
    for i in range(mp_l):
        for j in range(T_l):
            for line in ax[i, j].get_lines():
                line.remove()
    print("m, p, T -log|ν|", D_)
plt.close(fig)
del fig, ax


# creating a figure for plotting |ν|, θ, -log|ν| separately with varying
# energy splittings, impurity locations, and coupling strengths
fig, ax = plt.subplots(mp_l, D_l, figsize=(5*D_l, 3*mp_l), sharex="col",
                       constrained_layout=True)
for D, D_s in enumerate(Delta_strings):
    ax[0, D].set_title(r"$\Delta=%s$" % D_s)
    ax[mp_l-1, D].set_xlabel(r"$\tau$")

# plotting |ν| with varying energy splittings, impurity locations, and
# coupling strengths
for D in range(D_l):
    ts_ = ts[D, 0]
    ax[0, D].set_xlim(ts_[0], ts_[-1])
    for i in range(mp_l):
        ax[i, D].set_yscale("log")
for m, m_ in enumerate(ms):
    for p, pl_ in enumerate(places):
        i = p + m*p_l
        ax[i, 0].set_ylabel(r"$|\nu(\tau)|$ ($m=%s$, %s)" % (m_, pl_))
for T, T_ in enumerate(Ts):
    T_s = T_strings[T]
    for D in range(D_l):
        ts_ = ts[D, 0]
        c = colours[D]
        for m in range(m_l):
            for p in range(p_l):
                i = p + m*p_l
                nu_ = nus_[0, L, m, T, mu, p, D, 0]
                ax[i, D].plot(ts_[1:], nu_[1:], color=c)
                ax[i, D].relim()
                ax[i, D].autoscale(axis="y")
                ax[i, D].set_ylim(ax[i, D].get_ylim())
                ax[i, D].plot(ts_[:2], nu_[:2], color=c)
    fig.suptitle(r"$L=%s$, $T=%s$, $\mu=%s$" % (L_, T_s, mu_))
    fig.savefig("m, p, Δ/m, p, Δ abs(ν) (L = %s, T = %s, μ = %s).pdf" %
                (L_, T_, mu_))
    for i in range(mp_l):
        for j in range(D_l):
            for line in ax[i, j].get_lines():
                line.remove()
    print("m, p, Δ |ν|", T_)

# plotting θ with varying energy splittings, impurity locations, and
# coupling strengths
for D in range(D_l):
    for m, m_ in enumerate(ms):
        for p, pl_ in enumerate(places):
            i = p + m*p_l
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
    for D in range(D_l):
        ts_ = ts[D, 0]
        for m in range(m_l):
            for p in range(p_l):
                i = p + m*p_l
                ax[i, D].plot(ts_, nus_[1, L, m, T, mu, p, D, 0],
                              color=colours[D])
    fig.suptitle(r"$L=%s$, $T=%s$, $\mu=%s$" % (L_, T_s, mu_))
    fig.savefig("m, p, Δ/m, p, Δ θ (L = %s, T = %s, μ = %s).pdf" %
                (L_, T_, mu_))
    for i in range(mp_l):
        for j in range(D_l):
            for line in ax[i, j].get_lines():
                line.remove()
    print("m, p, Δ θ", T_)

# plotting -log|ν| with varying energy splittings, impurity locations, and
# coupling strengths
for D in range(D_l):
    ts_ = ts[D, 1]
    ax[0, D].set_xlim(ts_[0], ts_[-1])
    ax[0, D].set_xscale("log")
    for m, m_ in enumerate(ms):
        for p, pl_ in enumerate(places):
            i = p + m*p_l
            ax[i, D].set_yscale("log")
            if D == 0:
                ax[i, D].set_ylabel(r"$-\log(|\nu(\tau)|)$ ($m=%s$, %s)" %
                                    (m_, pl_))
            else:
                ax[i, D].tick_params(labelleft=True)
for T, T_ in enumerate(Ts):
    T_s = T_strings[T]
    for D in range(D_l):
        ts_ = ts[D, 1]
        for m in range(m_l):
            for p in range(p_l):
                i = p + m*p_l
                ax[i, D].plot(ts_, nus_[2, L, m, T, mu, p, D, 1],
                              color=colours[D])
                ax[i, D].relim()
                ax[i, D].autoscale(axis="y")
    fig.suptitle(r"$L=%s$, $T=%s$, $\mu=%s$" % (L_, T_s, mu_))
    fig.savefig("m, p, Δ/m, p, Δ -log(abs(ν)) (L = %s, T = %s, μ = %s).pdf" %
                (L_, T_, mu_))
    for i in range(mp_l):
        for j in range(D_l):
            for line in ax[i, j].get_lines():
                line.remove()
    print("m, p, Δ -log|ν|", T_)
plt.close(fig)
del fig, ax


# creating a figure for plotting |ν|, θ, -log|ν| separately with varying
# temperatures and coupling strengths
fig, ax = plt.subplots(T_l, D_l, figsize=(5*D_l, 3*T_l), sharex="col",
                       constrained_layout=True)
for D, D_s in enumerate(Delta_strings):
    ax[0, D].set_title(r"$\Delta=%s$" % D_s)
    ax[T_l-1, D].set_xlabel(r"$\tau$")

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
        for D in range(D_l):
            ts_ = ts[D, 0]
            c = colours[D]
            for T in range(T_l):
                nu_ = nus_[0, L, m, T, mu, p, D, 0]
                ax[T, D].plot(ts_[1:], nu_[1:], color=c)
                ax[T, D].relim()
                ax[T, D].autoscale(axis="y")
                ax[T, D].set_ylim(ax[T, D].get_ylim())
                ax[T, D].plot(ts_[:2], nu_[:2], color=c)
        fig.suptitle(r"$L=%s$, $m=%s$, $\mu=%s$, %s" % (L_, m_, mu_, pl_))
        fig.savefig("T, Δ/T, Δ abs(ν) (L = %s, m = %s, μ = %s, %s).pdf"
                    % (L_, m_, mu_, pl_))
        for i in range(T_l):
            for j in range(D_l):
                for line in ax[i, j].get_lines():
                    line.remove()
        print("T, Δ |ν|", m_, pl_)

# plotting θ with varying temperatures and coupling strengths
for D in range(D_l):
    for T in range(T_l):
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
        for D in range(D_l):
            ts_ = ts[D, 0]
            for T in range(T_l):
                ax[T, D].plot(ts_, nus_[1, L, m, T, mu, p, D, 0],
                              color=colours[D])
        fig.suptitle(r"$L=%s$, $m=%s$, $\mu=%s$, %s" % (L_, m_, mu_, pl_))
        fig.savefig("T, Δ/T, Δ θ (L = %s, m = %s, μ = %s, %s).pdf" %
                    (L_, m_, mu_, pl_))
        for i in range(T_l):
            for j in range(D_l):
                for line in ax[i, j].get_lines():
                    line.remove()
        print("T, Δ θ", m_, pl_)

# plotting -log|ν| with varying temperatures and coupling strengths
for D in range(D_l):
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
        for D in range(D_l):
            ts_ = ts[D, 1]
            for T in range(T_l):
                ax[T, D].plot(ts_, nus_[2, L, m, T, mu, p, D, 1],
                              color=colours[D])
                ax[T, D].relim()
                ax[T, D].autoscale(axis="y")
        fig.suptitle(r"$L=%s$, $m=%s$, $\mu=%s$, %s" % (L_, m_, mu_, pl_))
        fig.savefig("T, Δ/T, Δ -log(abs(ν)) (L = %s, m = %s, μ = %s, %s).pdf"
                    % (L_, m_, mu_, pl_))
        for i in range(T_l):
            for j in range(D_l):
                for line in ax[i, j].get_lines():
                    line.remove()
        print("T, Δ -log|ν|", m_, pl_)
plt.close(fig)
del fig, ax
