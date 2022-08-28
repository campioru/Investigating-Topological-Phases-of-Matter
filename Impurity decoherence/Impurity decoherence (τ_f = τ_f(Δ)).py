"""QWZ Model: Impurity decoherence.

Introduces an impurity into the model and calculates the deocherence as a
function of time. Plots the decoherence magnitude and phase for a variety of
parameters (system length L, energy splitting m, temperature T, chemical
potential μ, impurity location, coupling strength Δ) for linear and logarithmic
time arrays, where the length of the time array is Δ-dependent (shorter times
for stronger coupling strengths).

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


lengths = np.linspace(3, 29, 14)
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


for L, L_ in enumerate(lengths):
    LxLy = L_**2
    LxLy2 = 2*LxLy
    SM, SMeiDt, Mdage_iEt = (np.empty((LxLy2, LxLy2), dtype=np.complex_) for _
                             in range(3))
    for m, m_ in enumerate(ms):
        h0 = h_QWZ(L_, L_, 0., m_, 1., 1.)
        h0_vals, Vdag = np.linalg.eig(h0)
        h0_vals = np.real(h0_vals)
        e_iE_diag = np.exp((-1j) * h0_vals)
        Vdag = Vdag.T.conj()
        for p, pl_ in enumerate(places):
            p_ = poss[L, p]
            h1 = copy.copy(h0)
            for D, D_ in enumerate(Deltas):
                h1[p_, p_] = h0[p_, p_] + D_
                h1[p_ + LxLy, p_ + LxLy] = h0[p_ + LxLy, p_ + LxLy] + D_
                h1_vals, M = np.linalg.eig(h1)
                eiD_diag = np.exp((1j) * np.real(h1_vals))
                M = np.dot(Vdag, M)
                Mdag = M.T.conj()
                for T, T_ in enumerate(Ts):
                    for mu, mu_ in enumerate(mus):
                        S_diag = (1. + np.exp((h0_vals - mu_) / T_)) ** (-1.)
                        for i in range(LxLy2):
                            SM[i] = S_diag[i] * M[i]
                        S_diag = 1. - S_diag
                        for s, s_ in enumerate(scales):
                            for t in range(t_nos):
                                t_ = ts[s, D, t]
                                for i in range(LxLy2):
                                    SMeiDt[:, i] = SM[:, i] * eiD_diag[i]**t_
                                    Mdage_iEt[:, i] = Mdag[:, i] * (
                                        e_iE_diag[i]**t_)
                                final = np.dot(SMeiDt, Mdage_iEt)
                                for i in range(LxLy2):
                                    final[i, i] += S_diag[i]
                                nus[L, m, T, mu, p, D, s, t] = np.linalg.det(
                                    final)
                        print(L_, m_, pl_, D_, T_, mu_, s_)
del (LxLy, LxLy2, SM, SMeiDt, Mdage_iEt, h0, h0_vals, Vdag, e_iE_diag, h1,
     h1_vals, M, eiD_diag, Mdag, S_diag, final)
nus_ = np.empty(((3,) + np.shape(nus)))
nus_[0] = abs(nus)
nus_[1] = np.angle(nus)
nus_[2] = -np.log10(nus_[0])

fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True,
                       constrained_layout=True)

ax[0].tick_params(labelbottom=False)
ax[0].set_yscale("log")
ax[0].set_ylabel(r"$|\nu(\tau)|$")
ax[1].set_xlabel(r"$\tau$")
ax[1].set_ylim(-1.1 * np.pi, 1.1 * np.pi)
ax[1].set_yticks([-np.pi, 0., np.pi], labels=[r"$-\pi$", "0", r"$\pi$"])
ax[1].set_ylabel(r"$\theta(\tau)$")


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
                        for i in range(2):
                            ax[i].plot(ts_, nus_[i, L, m, T, mu, p, D, 0],
                                       color=blinds[i])
                        ax[0].relim()
                        ax[0].autoscale(axis="y")
                        fig.suptitle(
                            r"$L=%s$, $m=%s$, $T=%s$, $\mu=%s$, %s, $\Delta=%s$"
                            % (L_, m_, T_s, mu_, pl_, D_s))
                        fig.savefig(
                            "%sx%s/abs(ν), θ (L = %s, m = %s, T = %s, μ = %s, %s, Δ = %s).pdf"
                            % (L_, L_, L_, m_, T_, mu_, pl_, D_),
                            bbox_inches="tight")
                        for line in ax[0].get_lines():
                            line.remove()
                        for line in ax[1].get_lines():
                            line.remove()

                        print(D_, T_, L_, m_, mu_, pl_)


if len(lengths) <= len(colours):
    lines, labels = [], []
    for L, L_ in enumerate(lengths):
        lines.append(Line2D([0], [0], color=colours[L]))
        labels.append(r"$L=%s$" % L_)
    legend = fig.legend(lines, labels, loc="center left",
                        bbox_to_anchor=(1, .5))
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
                        for L, L_ in enumerate(lengths):
                            for i in range(2):
                                ax[i].plot(ts_, nus_[i, L, m, T, mu, p, D, 0],
                                           color=colours[L])
                        ax[0].relim()
                        ax[0].autoscale(axis="y")
                        fig.suptitle(
                            r"$m=%s$, $T=%s$, $\mu=%s$, %s, $\Delta=%s$" %
                            (m_, T_s, mu_, pl_, D_s))
                        fig.savefig(
                            "L/L abs(ν), θ (m = %s, T = %s, μ = %s, %s, Δ = %s).pdf"
                            % (m_, T_, mu_, pl_, D_), bbox_inches="tight")
                        for line in ax[0].get_lines():
                            line.remove()
                        for line in ax[1].get_lines():
                            line.remove()

                        print("L", D_, T_, m_, mu_, pl_)
    legend.remove()
    del legend
else:
    L_half = int(len(lengths)/2)
    lines, labels = [], []
    lengths_half = lengths[:L_half]
    for L, L_ in enumerate(lengths_half):
        lines.append(Line2D([0], [0], color=colours[L]))
        labels.append(r"$L=%s$" % L_)
    legend = fig.legend(lines, labels, loc="center left",
                        bbox_to_anchor=(1, .5))
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
                        for L, L_ in enumerate(lengths_half):
                            for i in range(2):
                                ax[i].plot(ts_, nus_[i, L, m, T, mu, p, D, 0],
                                           color=colours[L])
                        ax[0].relim()
                        ax[0].autoscale(axis="y")
                        fig.suptitle(
                            r"$m=%s$, $T=%s$, $\mu=%s$, %s, $\Delta=%s$" %
                            (m_, T_s, mu_, pl_, D_s))
                        fig.savefig(
                            "L/L- abs(ν), θ (m = %s, T = %s, μ = %s, %s, Δ = %s).pdf"
                            % (m_, T_, mu_, pl_, D_), bbox_inches="tight")
                        for line in ax[0].get_lines():
                            line.remove()
                        for line in ax[1].get_lines():
                            line.remove()

                        print("L-", D_, T_, m_, mu_, pl_)
    legend.remove()
    del legend

    lines, labels = [], []
    lengths_half = lengths[L_half:]
    for L, L_ in enumerate(lengths_half):
        lines.append(Line2D([0], [0], color=colours[L]))
        labels.append(r"$L=%s$" % L_)
    legend = fig.legend(lines, labels, loc="center left",
                        bbox_to_anchor=(1, .5))
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
                        for L, L_ in enumerate(lengths_half):
                            for i in range(2):
                                ax[i].plot(ts_, nus_[i, L, m, T, mu, p, D, 0],
                                           color=colours[L])
                        ax[0].relim()
                        ax[0].autoscale(axis="y")
                        fig.suptitle(
                            r"$m=%s$, $T=%s$, $\mu=%s$, %s, $\Delta=%s$" %
                            (m_, T_s, mu_, pl_, D_s))
                        fig.savefig(
                            "L/L+ abs(ν), θ (m = %s, T = %s, μ = %s, %s, Δ = %s).pdf"
                            % (m_, T_, mu_, pl_, D_), bbox_inches="tight")
                        for line in ax[0].get_lines():
                            line.remove()
                        for line in ax[1].get_lines():
                            line.remove()

                        print("L+", D_, T_, m_, mu_, pl_)
    legend.remove()
    del legend

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
                for m in range(len(ms)):
                    for p in range(len(places)):
                        for i in range(2):
                            ax[i].plot(ts_, nus_[i, L, m, T, mu, p, D, 0],
                                       color=colours[p + m*len(places)])
                ax[0].relim()
                ax[0].autoscale(axis="y")
                fig.suptitle(
                    r"$L=%s$, $T=%s$, $\mu=%s$, $\Delta=%s$" %
                    (L_, T_s, mu_, D_s))
                fig.savefig(
                    "m, p/m, p abs(ν), θ (L = %s, T = %s, μ = %s, Δ = %s).pdf"
                    % (L_, T_, mu_, D_), bbox_inches="tight")
                for line in ax[0].get_lines():
                    line.remove()
                for line in ax[1].get_lines():
                    line.remove()

                print("m, p", D_, T_, L_, mu_)
legend.remove()
del legend


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
                    for T in range(len(Ts)):
                        for i in range(2):
                            ax[i].plot(ts_, nus_[i, L, m, T, p, D, 0],
                                       color=colours[T])
                    ax[0].relim()
                    ax[0].autoscale(axis="y")
                    fig.suptitle(
                        r"$L=%s$, $m=%s$, $\mu=%s$, %s, $\Delta=%s$" %
                        (L_, m_, mu_, pl_, D_s))
                    fig.savefig(
                        "T/T abs(ν), θ (L = %s, m = %s, μ = %s, %s, Δ = %s).pdf"
                        % (L_, m_, mu_, pl_, D_), bbox_inches="tight")
                    for line in ax[0].get_lines():
                        line.remove()
                    for line in ax[1].get_lines():
                        line.remove()

                    print("T", D_, L_, m_, mu_, pl_)


plt.close(fig)
del fig, ax, legend


fig, ax = plt.subplots(2, 2, figsize=(12, 6), sharex="col",
                       constrained_layout=True)

ax[0, 0].tick_params(labelbottom=False)
ax[0, 0].set_yscale("log")
ax[0, 0].set_ylabel(r"$|\nu(\tau)|$")
ax[1, 0].set_xlabel(r"$\tau$")
ax[1, 0].set_yscale("log")
ax[1, 0].set_ylabel(r"$-\log(|\nu(\tau)|)$")

ax[0, 1].tick_params(labelbottom=False)
ax[0, 1].set_yscale("log")
ax[1, 1].set_xscale("log")
ax[1, 1].set_xlabel(r"$\tau$")
ax[1, 1].set_yscale("log")


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
                            for j in range(2):
                                ax[i, j].plot(ts_[j],
                                              nus_[2*i, L, m, T, mu, p, D, j],
                                              color=blinds[2*i])
                                ax[i, j].relim()
                                ax[i, j].autoscale(axis="y")
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


if len(lengths) <= len(colours):
    lines, labels = [], []
    for L, L_ in enumerate(lengths):
        lines.append(Line2D([0], [0], color=colours[L]))
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
                        for L, L_ in enumerate(lengths):
                            for i in range(2):
                                for j in range(2):
                                    ax[i, j].plot(ts_[j],
                                                  nus_[2*i, L, m, T, mu, p, D,
                                                       j], color=colours[L])
                        for i in range(2):
                            for j in range(2):
                                ax[i, j].relim()
                                ax[i, j].autoscale(axis="y")
                        fig.suptitle(
                            r"$m=%s$, $T=%s$, $\mu=%s$, %s, $\Delta=%s$" %
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
else:
    L_half = int(len(lengths)/2)
    lines, labels = [], []
    lengths_half = lengths[:L_half]
    for L, L_ in enumerate(lengths_half):
        lines.append(Line2D([0], [0], color=colours[L]))
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
                        for L, L_ in enumerate(lengths_half):
                            for i in range(2):
                                for j in range(2):
                                    ax[i, j].plot(ts_[j],
                                                  nus_[2*i, L, m, T, mu, p, D,
                                                       j], color=colours[L])
                        for i in range(2):
                            for j in range(2):
                                ax[i, j].relim()
                                ax[i, j].autoscale(axis="y")
                        fig.suptitle(
                            r"$m=%s$, $T=%s$, $\mu=%s$, %s, $\Delta=%s$" %
                            (m_, T_s, mu_, pl_, D_s))
                        fig.savefig(
                            "L/L- abs(ν), -log(abs(ν)) (m = %s, T = %s, μ = %s, %s, Δ = %s).pdf"
                            % (m_, T_, mu_, pl_, D_), bbox_inches="tight")
                        for i in range(2):
                            for j in range(2):
                                for line in ax[i, j].get_lines():
                                    line.remove()

                    print("L-", D_, T_, m_, mu_, pl_)
    legend.remove()
    del legend

    lines, labels = [], []
    lengths_half = lengths[L_half:]
    for L, L_ in enumerate(lengths_half):
        lines.append(Line2D([0], [0], color=colours[L]))
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
                        for L, L_ in enumerate(lengths_half):
                            for i in range(2):
                                for j in range(2):
                                    ax[i, j].plot(ts_[j],
                                                  nus_[2*i, L, m, T, mu, p, D,
                                                       j], color=colours[L])
                        for i in range(2):
                            for j in range(2):
                                ax[i, j].relim()
                                ax[i, j].autoscale(axis="y")
                        fig.suptitle(
                            r"$m=%s$, $T=%s$, $\mu=%s$, %s, $\Delta=%s$" %
                            (m_, T_s, mu_, pl_, D_s))
                        fig.savefig(
                            "L/L+ abs(ν), -log(abs(ν)) (m = %s, T = %s, μ = %s, %s, Δ = %s).pdf"
                            % (m_, T_, mu_, pl_, D_), bbox_inches="tight")
                        for i in range(2):
                            for j in range(2):
                                for line in ax[i, j].get_lines():
                                    line.remove()

                    print("L+", D_, T_, m_, mu_, pl_)
    legend.remove()
    del legend


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
                for m in range(len(ms)):
                    for p in range(len(places)):
                        for i in range(2):
                            for j in range(2):
                                ax[i, j].plot(ts_[j],
                                              nus_[2*i, L, m, T, mu, p, D, j],
                                              color=colours[p + m*len(places)])
                for i in range(2):
                    for j in range(2):
                        ax[i, j].relim()
                        ax[i, j].autoscale(axis="y")
                fig.suptitle(
                    r"$L=%s$, $T=%s$, $\mu=%s$, $\Delta=%s$" %
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
                    for T in range(len(Ts)):
                        for i in range(2):
                            for j in range(2):
                                ax[i, j].plot(ts_[j],
                                              nus_[2*i, L, m, T, mu, p, D, j],
                                              color=colours[T])
                    for i in range(2):
                        for j in range(2):
                            ax[i, j].relim()
                            ax[i, j].autoscale(axis="y")
                    fig.suptitle(
                        r"$L=%s$, $m=%s$, $\mu=%s$, %s, $\Delta=%s$" %
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


fig, ax = plt.subplots(3, 1, figsize=(6, 9), sharex=True,
                       constrained_layout=True)

ax[0].tick_params(labelbottom=False)
ax[0].set_yscale("log")
ax[0].set_ylabel(r"$|\nu(\tau)|$")
ax[1].tick_params(labelbottom=False)
ax[1].set_ylim(-1.1 * np.pi, 1.1 * np.pi)
ax[1].set_yticks([-np.pi, 0., np.pi], labels=[r"$-\pi$", "0", r"$\pi$"])
ax[1].set_ylabel(r"$\theta(\tau)$")
ax[2].set_xscale("log")
ax[2].set_xlim(np.amin(ts[:, 1]), np.amax(ts[:, 1]))
ax[2].set_xlabel(r"$\tau$")
ax[2].set_yscale("log")
ax[2].set_ylabel(r"$-\log(|\nu(\tau)|)$")


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
                    for D in range(len(Deltas)):
                        ts_ = ts[D, 1]
                        for i in range(3):
                            ax[i].plot(ts_, nus_[i, L, m, T, mu, p, D, 1],
                                       color=colours[D])
                    for i in range(2):
                        ax[i].relim()
                        ax[i].autoscale(axis="y")
                    fig.suptitle(
                        r"$L=%s$, $m=%s$, $T=%s$, $\mu=%s$, %s" %
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
del (labels, lines, D, D_, D_s, i, j, L, L_, lengths_half, line, m, m_, mu,
     mu_, nus_, p, p_, pl_, T, t, T_, t_, T_s, ts_)
