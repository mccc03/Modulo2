"""Lanczos algortithm with exact diagonalizatoin.
Also: generate Hamiltonian of the transverse field Ising model.
H = -J sum_i sigma^x_i sigma^x_{i+1} - g sum_i sigma^z i; periodic boundary cond.
"""

import numpy as np
import scipy
from scipy import sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt


Id = sparse.csr_matrix(np.eye(2))
Sx = sparse.csr_matrix([[0., 1.], [1., 0.]])
Sz = sparse.csr_matrix([[1., 0.], [0., -1.]])
Splus = sparse.csr_matrix([[0., 1.], [0., 0.]])
Sminus = sparse.csr_matrix([[0., 0.], [1., 0.]])


def singesite_to_full(op, i, L):
    op_list = [Id]*L  # = [Id, Id, Id ...] with L entries
    op_list[i] = op
    full = op_list[0]
    for op_i in op_list[1:]:
        full = sparse.kron(full, op_i, format="csr")
    return full


def gen_sx_list(L):
    return [singesite_to_full(Sx, i, L) for i in range(L)]


def gen_sz_list(L):
    return [singesite_to_full(Sz, i, L) for i in range(L)]


def gen_hamiltonian(sx_list, sz_list, g, J=1.):
    L = len(sx_list)
    H = sparse.csr_matrix((2**L, 2**L))
    for j in range(L):
        H = H - J *( sx_list[j] * sx_list[(j+1)%L])
        H = H - g * sz_list[j]
    return H

def magnetization(sx_list,ground):
    L = len(sx_list)
    #M = sparse.csr_matrix((2**L, 2**L))
    M=np.zeros((2**L,2**L))
    for j in range(L):
        M = M + sx_list[j]
    M=np.inner(ground.conj(), M*M).real
    M=np.inner(M,ground)
    M=(1/L)*math.sqrt(M)
    return M


def lanczos(psi0, H, N=200, stabilize=False):
    """Perform a Lanczos iteration building the tridiagonal matrix T and ONB of the Krylov space."""
    if psi0.ndim != 1:
        raise ValueError("psi0 should be a vector, "
                         "i.e., a numpy array with a single dimension of len 2**L")
    if H.shape[1] != psi0.shape[0]:
        raise ValueError("Shape of H doesn't match len of psi0.")
    psi0 = psi0/np.linalg.norm(psi0)
    vecs = [psi0]
    T = np.zeros((N, N))
    psi = H @ psi0  # @ means matrix multiplication
    # and works both for numpy arrays and scipy.sparse.csr_matrix
    alpha = T[0, 0] = np.inner(psi0.conj(), psi).real
    psi = psi - alpha* vecs[-1]
    for i in range(1, N):
        beta = np.linalg.norm(psi)
        if beta  < 1.e-13:
            print("Lanczos terminated early after i={i:d} steps:"
                  "full Krylov space built".format(i=i))
            T = T[:i, :i]
            break
        psi /= beta
        # note: mathematically, psi should be orthogonal to all other states in `vecs`
        if stabilize:
            for vec in vecs:
                psi -= vec * np.inner(vec.conj(), psi)
            psi /= np.linalg.norm(psi)
        vecs.append(psi)
        psi = H @ psi - beta * vecs[-2]
        alpha = np.inner(vecs[-1].conj(), psi).real
        psi = psi - alpha * vecs[-1]
        T[i, i] = alpha
        T[i-1, i] = T[i, i-1] = beta
    return T, vecs

'''
def colorplot(xs, ys, data, **kwargs):
    """Create a colorplot with matplotlib.pyplot.imshow.
    Parameters
    ----------
    xs : 1D array, shape (n,)
        x-values of the points for which we have data; evenly spaced
    ys : 1D array, shape (m,)
        y-values of the points for which we have data; evenly spaced
    data : 2D array, shape (m, n)
        ``data[i, j]`` corresponds to the points ``(xs[i], ys[j])``
    **kwargs :
        additional keyword arguments, given to `imshow`.
    """
    data = np.asarray(data)
    if data.shape != (len(xs), len(ys)):
        raise ValueError("Shape of data doesn't match len of xs and ys!")
    dx = (xs[-1] - xs[0])/(len(xs)-1)
    assert abs(dx - (xs[1]-xs[0])) < 1.e-10
    dy = (ys[-1] - ys[0])/(len(ys)-1)
    assert abs(dy - (ys[1]-ys[0])) < 1.e-10
    extent = (xs[0] - 0.5 * dx, xs[-1] + 0.5 * dx,  # left, right
              ys[0] - 0.5 * dy, ys[-1] + 0.5 * dy)  # bottom, top
    kwargs.setdefault('aspect', 'auto')
    kwargs.setdefault('interpolation', 'nearest')
    kwargs.setdefault('extent', extent)
    # convention of imshow: matrix like data[row, col] with (0, 0) top left.
    # but we want data[col, row] with (0, 0) bottom left -> transpose and invert y axis
    plt.imshow(data.T[::-1, :], **kwargs)



L=10
g=0.8
sz_list=gen_sz_list(L)
sx_list=gen_sx_list(L)
H= gen_hamiltonian(sx_list, sz_list, g, J=1.)
psi_start = np.random.random(H.shape[0])
T,vecs=lanczos(psi_start, H, N=200, stabilize=True)

# c) find the ground state
E, v = np.linalg.eigh(T)
v0 = v[:, 0]
psi0 = np.dot(np.transpose(vecs), v0)  # Note: we need np.dot or @, * doesn't work
# alternative:
#psi0 = vecs[0] * v0[0]
#for i in range(1, len(vecs)):
#    psi0 = psi0 +  vecs[i] * v0[i]

print("norm = ", np.linalg.norm(psi0))
E0 = np.inner(psi0, H @ psi0) / np.linalg.norm(psi0)
print("E0 = {E0:.15f}, relative error {err:.3e}".format(
        E0=E0, err=abs(E0-E[0])/abs(E[0])))
var = np.linalg.norm(H @ psi0)**2 - E0**2
print("variance = {var:.3e}".format(var=var))
'''

'''

Ls = [6, 8, 10, 12]
gs = np.linspace(0., 2., 21)

plt.figure()
for L in Ls:
    sx_list = gen_sx_list(L)
    sz_list = gen_sz_list(L)
    mags = []
    for g in gs:
        H = gen_hamiltonian(sx_list, sz_list, g, J=1.)
        psi_start = np.random.random(H.shape[0])
        T,vecs=lanczos(psi_start, H, N=200, stabilize=True)
        E, v = np.linalg.eigh(T)
        v0 = v[:, 0]  # first column of v is the ground state
        psi0 = np.dot(np.transpose(vecs), v0)
        mag=magnetization(sz_list,psi0)
        mags.append(mag)

    mags = np.array(mags)
    print(mags)
    plt.plot(gs, mags, label="L={L:d}".format(L=L))
plt.xlabel("g")
plt.ylabel("C")
plt.legend()
'''

Ls = [10]
gs = np.linspace(0., 2., 20 )

plt.figure()
for L in Ls:
    sx_list = gen_sx_list(L)
    sz_list = gen_sz_list(L)
    mags=[]
    #sxsx = sx_list[0]*sx_list[L//2]
    #corrs = []
    '''
    #mags=[]
    #M=sparse.csr_matrix((2**L, 2**L))
    j=0
    while j<L:
        M += sx_list[j]
        j+1
    print(M)
    '''
    for g in gs:
        H= gen_hamiltonian(sx_list, sz_list, g, J=1.)
        psi_start = np.random.random(H.shape[0])
        T,vecs=lanczos(psi_start, H, N=200, stabilize=True)
        # c) find the ground state
        print(T)
        E, v = np.linalg.eigh(T)
        v0 = v[:, 0]
        psi0 = np.dot(np.transpose(vecs), v0)
        #mag=magnetization(M,psi0)
        #corr = np.inner(psi0, sxsx*psi0)

        M=0
        for i in range(L):
            for j in range(L):
                M=M+(1/L**2)*np.inner(psi0, sx_list[i]*sx_list[j]*psi0)
        M=np.sqrt(M)
        #mag = math.sqrt(np.inner(psi0, M*M*psi0))
        #mag=mag/L
        #corrs.append(corr)
        mags.append(M)
    #corrs = np.array(corrs)
    mags=np.array(mags)
    print(gs)
    #print(corrs)
    print(mags)
    #plt.plot(gs, corrs, label="L={L:d}".format(L=L))
    #plt.plot(gs, mags, label="L={L:d}".format(L=L))

#plt.xlabel("g")
#plt.ylabel("C")
#plt.legend()
#plt.show()
