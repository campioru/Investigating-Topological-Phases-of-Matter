"""QWZ Model: Unperturbed eigenstates.

Calculates the energy eigenvalues and eigenvectors of the unperturbed QWZ model
for a given set of parameters (system size Lx * Ly, energy splitting m).

@author: Ruaidhrí Campion
"""


import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from my_functions import h_QWZ
import itertools as it


def eigen(A):
    """Eigenvalues and eigenvectors ordered by increasing eigenvalue."""
    vals, vecs = np.linalg.eig(A)
    perm = vals.argsort()
    return np.real(vals[perm]), vecs[:, perm]


def edge_bulk(Lx, Ly):
    """Seperated indices and positions corresponding to edge and bulk."""
    edge_ind = [x + y*Lx for y, x in it.product(range(Ly), range(Lx)) if
                x in [0, Lx-1] or y in [0, Ly-1]]
    bulk_ind = [x + y*Lx for y, x in it.product(range(1, Ly-1), range(1, Lx-1))]
    for ind in [edge_ind, bulk_ind]:
        ind += [i + Lx*Ly for i in ind]
    return edge_ind, bulk_ind


# initialising system parameters
Lx = 20
Ly = 20
ms = np.linspace(0., 5., 11)

LxLy2 = 2*Lx*Ly
alphas = np.arange(LxLy2)
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
h0s = np.array([h_QWZ(Lx, Ly, 0., m_, 1., 1.) for m_ in ms])
vals = np.empty((len(ms), LxLy2))
vecs = np.empty(np.shape(h0s), dtype=np.complex128)
for m, h0 in enumerate(h0s):
    vals[m], vecs[m] = eigen(h0)

# plotting the eigenvalues
fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True,
                       constrained_layout=True)
ax[0].tick_params(labelbottom=False)
ax[0].set_ylabel(r"$\frac{\omega_\alpha-\omega_0}{t}$")
ax[1].set_xlim(0., LxLy2)
ax[1].set_xlabel(r"$\alpha$")
ax[1].set_ylabel(r"$\frac{\omega_\alpha-\omega_0}{t}$")
for m, m_ in enumerate(ms):
    ax_ = ax[0] if m_ <= 2. else ax[1]
    ax_.scatter(alphas, vals[m], s=1., color=colours[m], label=rf"$m={m_}t$")
fig.legend(loc="center left", bbox_to_anchor=(1, .5))
fig.savefig("Eigenvalues.pdf", bbox_inches="tight")
plt.clf()
del fig, ax


# repeating the above but for m = t, 3t
ms = np.array([1., 3.])
h0s = np.array([h_QWZ(Lx, Ly, 0., m_, 1., 1.) for m_ in ms])
vals = np.empty((len(ms), LxLy2))
vecs = np.empty(np.shape(h0s), dtype=np.complex128)
for m, h0 in enumerate(h0s):
    vals[m], vecs[m] = eigen(h0)


# calculating the bulk and edge probabilities for each eigenstate
probs = (np.abs(vecs)) ** 2.
edge_probs, bulk_probs = (np.sum(probs[:, ind], axis=1) for ind in
                          edge_bulk(Lx, Ly))

# plotting probabilities and corresponding eigenvalues
fig, ax = plt.subplots(3, 1, figsize=(6, 12), sharex=True,
                       gridspec_kw={'height_ratios': [1.5, 1., 1.5]},
                       constrained_layout=True)
ax[0].set_ylabel(r"Probability ($m=t$)")
ax[1].set_ylabel("Eigenvalue")
ax[2].set_xlim(0., LxLy2)
ax[2].set_xlabel(r"$\alpha$")
ax[2].set_ylabel(r"Probability ($m=3t$)")

ax[0].scatter(alphas, bulk_probs[0], s=1., color="r", label="Bulk")
ax[0].scatter(alphas, edge_probs[0], s=1., color="b", label="Edge")
ax[0].legend(loc="center right")

ax[2].scatter(alphas, bulk_probs[1], s=1., color="k", label="Bulk")
ax[2].scatter(alphas, edge_probs[1], s=1., color="g", label="Edge")
ax[2].legend(loc="center right")

bulk_greater = np.where(edge_probs[0] <= bulk_probs[0])[0]
edge_greater = np.where(edge_probs[0] > bulk_probs[0])[0]
ax[1].scatter(bulk_greater, vals[0, bulk_greater], s=1., color="r",
              label=r"$m=t$, $P(\mathrm{bulk})\geqslant P(\mathrm{edge})$")
ax[1].scatter(edge_greater, vals[0, edge_greater], s=1., color="b",
              label=r"$m=t$, $P(\mathrm{bulk})<P(\mathrm{edge})$")
ax[1].scatter(alphas, vals[1], s=1., color="k", label=r"$m=3t$")
ax[1].legend(loc="lower right")

fig.savefig("Eigenvalues & probabilities.pdf")
plt.clf()
del fig, ax
