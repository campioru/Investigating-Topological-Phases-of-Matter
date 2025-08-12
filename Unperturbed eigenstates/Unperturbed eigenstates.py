"""QWZ Model: Unperturbed eigenstates.

Calculates the energy eigenvalues and eigenvectors of the unperturbed QWZ model
for a given set of parameters (system size Lx * Ly, energy splitting m).

@author: Ruaidhrí Campion
"""


import numpy as np
import matplotlib.pyplot as plt
import copy
import sys
sys.path.append("../")
from my_functions import h_QWZ


def eigen(A):
    """Eigenvalues and eigenvectors ordered by increasing eigenvalue."""
    un_vals, un_vecs = np.linalg.eig(A)
    vals = np.empty(np.shape(un_vals))
    vecs = np.empty(np.shape(un_vecs), dtype=np.complex_)
    for i in range(len(un_vals)):
        arg = np.argmin(un_vals)
        vals[i] = np.real(un_vals[arg])
        vecs[:, i] = un_vecs[:, arg]
        un_vals = np.delete(un_vals, arg)
        un_vecs = np.delete(un_vecs, arg, 1)
    return vals, vecs


def edge_bulk(Lx, Ly):
    """Seperated indices and positions corresponding to edge and bulk."""
    un_edge_ind, un_bulk_ind = (np.empty(0, dtype=int) for a in range(2))
    un_edge_pos, un_bulk_pos = (np.empty((0, 3), dtype=int) for a in range(2))
    for x in range(Lx):
        for y in range(Ly):
            if x == 0 or y == 0 or x == Lx-1 or y == Ly-1:
                un_edge_ind = np.append(un_edge_ind,
                                        [x + y*Lx, x + y*Lx + Lx*Ly])
                un_edge_pos = np.append(un_edge_pos,
                                        [[x, y, 0], [x, y, 1]], axis=0)
            else:
                un_bulk_ind = np.append(un_bulk_ind,
                                        [x + y*Lx, x + y*Lx + Lx*Ly])
                un_bulk_pos = np.append(un_bulk_pos,
                                        [[x, y, 0], [x, y, 1]], axis=0)
    edge_ind = np.empty(np.shape(un_edge_ind), dtype=int)
    bulk_ind = np.empty(np.shape(un_bulk_ind), dtype=int)
    edge_pos = np.empty(np.shape(un_edge_pos), dtype=int)
    bulk_pos = np.empty(np.shape(un_bulk_pos), dtype=int)
    for i in range(len(edge_ind)):
        arg = np.argmin(un_edge_ind)
        edge_ind[i] = un_edge_ind[arg]
        edge_pos[i] = un_edge_pos[arg]
        un_edge_ind = np.delete(un_edge_ind, arg)
        un_edge_pos = np.delete(un_edge_pos, arg, 0)
    for i in range(len(bulk_ind)):
        arg = np.argmin(un_bulk_ind)
        bulk_ind[i] = un_bulk_ind[arg]
        bulk_pos[i] = un_bulk_pos[arg]
        un_bulk_ind = np.delete(un_bulk_ind, arg)
        un_bulk_pos = np.delete(un_bulk_pos, arg, 0)
    return edge_ind, bulk_ind, edge_pos, bulk_pos.astype


# initialising system parameters
Lx = 20
Ly = 20
LxLy2 = 2*Lx*Ly
ms = np.linspace(0., 5., 11)
h0s = np.empty((len(ms), LxLy2, LxLy2), dtype=np.complex_)
vals = np.empty((len(ms), LxLy2))
vecs = np.empty(np.shape(h0s), dtype=np.complex_)
alphas = np.linspace(1., LxLy2, LxLy2)
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

# calculating h0 and its eigenvalues for each m
for m in range(len(ms)):
    h0s[m] = h_QWZ(Lx, Ly, 0., ms[m], 1., 1.)
    vals[m], vecs[m] = eigen(h0s[m])

# plotting the eigenvalues
fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True,
                       constrained_layout=True)
ax[0].tick_params(labelbottom=False)
ax[0].set_ylabel(r"$\frac{\omega_\alpha-\omega_0}{t}$")
ax[1].set_xlim(0., LxLy2)
ax[1].set_xlabel(r"$\alpha$")
ax[1].set_ylabel(r"$\frac{\omega_\alpha-\omega_0}{t}$")
for m, m_ in enumerate(ms):
    if m_ <= 2.:
        ax[0].scatter(alphas, vals[m], s=1., color=colours[m],
                      label=r"m = %s t" % m_)
    else:
        ax[1].scatter(alphas, vals[m], s=1., color=colours[m],
                      label=r"m = %s t" % m_)
fig.legend(loc="center left", bbox_to_anchor=(1, .5))
# fig.savefig("Eigenvalues.pdf", bbox_inches="tight")
del fig, ax


# repeating the above but for m = t, 3t
ms = np.array([1., 3.])
h0s = np.empty((len(ms), LxLy2, LxLy2), dtype=np.complex_)
vals = np.empty((len(ms), LxLy2))
vecs = np.empty(np.shape(h0s), dtype=np.complex_)

for m in range(len(ms)):
    h0s[m] = h_QWZ(Lx, Ly, 0., ms[m], 1., 1.)
    vals[m], vecs[m] = eigen(h0s[m])


# calculating the bulk and edge probabilities for each eigenstate
probs = (np.abs(vecs)) ** 2.
edge_ind, bulk_ind, edge_pos, bulk_pos = edge_bulk(20, 20)
edge_probs = np.empty((np.shape(probs)[0], len(edge_ind), np.shape(probs)[-1]))
bulk_probs = np.empty((np.shape(probs)[0], len(bulk_ind), np.shape(probs)[-1]))
for m in range(len(ms)):
    for i in range(len(edge_ind)):
        edge_probs[m, i] = probs[m, edge_ind[i]]
    for i in range(len(bulk_ind)):
        bulk_probs[m, i] = probs[m, bulk_ind[i]]
sum_edge_probs = np.empty(
    (np.shape(edge_probs)[0], 1, np.shape(edge_probs)[-1]))
sum_bulk_probs = np.empty(
    (np.shape(bulk_probs)[0], 1, np.shape(bulk_probs)[-1]))
for m in range(len(ms)):
    for i in range(np.shape(edge_probs)[-1]):
        sum_edge_probs[m, 0, i] = np.sum(edge_probs[m, :, i])
    for i in range(np.shape(bulk_probs)[-1]):
        sum_bulk_probs[m, 0, i] = np.sum(bulk_probs[m, :, i])

sum_probs = np.concatenate((sum_edge_probs, sum_bulk_probs), axis=1)

# plotting probabilities and corresponding eigenvalues
fig, ax = plt.subplots(3, 1, figsize=(6, 12), sharex=True,
                       gridspec_kw={'height_ratios': [1.5, 1., 1.5]},
                       constrained_layout=True)
ax[0].set_ylabel(r"Probability ($m=t$)")
ax[1].set_ylabel("Eigenvalue")
ax[2].set_xlim(0., LxLy2)
ax[2].set_xlabel(r"$\alpha$")
ax[2].set_ylabel("Probability ($m=3t$)")

ax[0].scatter(alphas, sum_probs[0, 1], s=1., color="r", label="Bulk")
ax[0].scatter(alphas, sum_probs[0, 0], s=1., color="b", label="Edge")
ax[0].legend(loc="center right")
ax[1].scatter(alphas, vals[1], s=1., color="k", label="$m=3t$")
ax[2].scatter(alphas, sum_probs[1, 1], s=1., color="k", label="Bulk")
ax[2].scatter(alphas, sum_probs[1, 0], s=1., color="g", label="Edge")
ax[2].legend(loc="center right")

a = 0
while sum_probs[0, 1, a] >= sum_probs[0, 0, a]:
    a += 1
a1 = copy.copy(a)
while sum_probs[0, 1, a] <= sum_probs[0, 0, a]:
    a += 1
a2 = copy.copy(a)
ax[1].scatter(np.concatenate((alphas[:a1], alphas[a2:])),
              np.concatenate((vals[0, :a1], vals[0, a2:])), s=1., color="r",
              label=r"$m=t$, $P$(bulk)$>P$(edge)")
ax[1].scatter(alphas[a1:a2], vals[0, a1:a2], s=1., color="b",
              label=r"$m=t$, $P$(bulk)$<P$(edge)")
ax[1].legend(loc="lower right")

# fig.savefig("Eigenvalues & probabilities.pdf")
del fig, ax
