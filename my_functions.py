import numpy as np
import itertools as it

def h_QWZ(Lx, Ly, ω0, m, tx, ty):
    """QWZ Hamiltonian matrix representation."""
    hm, hx, hy = (np.zeros((2*Lx*Ly, 2*Lx*Ly), dtype=np.complex128) for _ in
                  range(3))
    for x, y in it.product(range(Lx), range(Ly)):
        hm[x + y*Lx, x + y*Lx] = ω0 + m
        hm[x + y*Lx + Lx*Ly, x + y*Lx + Lx*Ly] = ω0 - m
    for x, y in it.product(range(Lx-1), range(Ly)):
        (hx[x+1 + y*Lx, x + y*Lx], hx[x + y*Lx, x+1 + y*Lx],
         hx[x+1 + y*Lx, x + y*Lx + Lx*Ly], hx[x + y*Lx + Lx*Ly, x+1 + y*Lx]
         ) = (tx/2. for _ in range(4))
        (hx[x+1 + y*Lx + Lx*Ly, x + y*Lx], hx[x + y*Lx, x+1 + y*Lx + Lx*Ly],
         hx[x+1 + y*Lx + Lx*Ly, x + y*Lx + Lx*Ly],
         hx[x + y*Lx + Lx*Ly, x+1 + y*Lx + Lx*Ly]) = (-tx/2. for _ in range(4))
    for x, y in it.product(range(Lx), range(Ly-1)):
        hy[x + (y+1)*Lx, x + y*Lx], hy[x + y*Lx, x + (y+1)*Lx] = (ty/2. for _ in
                                                                  range(2))
        (hy[x + (y+1)*Lx, x + y*Lx + Lx*Ly], hy[x + (y+1)*Lx + Lx*Ly, x + y*Lx]
         ) = (ty/2.*1j for _ in range(2))
        (hy[x + y*Lx + Lx*Ly, x + (y+1)*Lx], hy[x + y*Lx, x + (y+1)*Lx + Lx*Ly]
         ) = (-ty/2.*1j for _ in range(2))
        (hy[x + (y+1)*Lx + Lx*Ly, x + y*Lx + Lx*Ly],
         hy[x + y*Lx + Lx*Ly, x + (y+1)*Lx + Lx*Ly]) = (-ty/2. for _ in
                                                        range(2))
    return hm + hx + hy