import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def Lanczos( A, v, m ):
    n = len(v)
    if m>n: m = n;
    # from here https://en.wikipedia.org/wiki/Lanczos_algorithm
    V = np.zeros( (m,n) )
    T = np.zeros( (m,m) )
    vo   = np.zeros(n)
    beta = 0
    for j in range(1, m-1 ):
        w    = np.dot( A, v )
        alfa = np.dot( w, v )
        w    = w - alfa * v - beta * vo
        beta = np.sqrt( np.dot( w, w ) )
        vo   = v
        v    = w / beta
        T[j,j  ] = alfa
        T[j,j+1] = beta
        T[j+1,j] = beta
        V[j,:]   = v
    w    = np.dot( A,  v )
    alfa = np.dot( w, v )
    w    = w - alfa * v - beta * vo
    T[m-1,m-1] = np.dot( w, v )
    V[m-1]     = w / np.sqrt( np.dot( w, w ) )
    return T, V


# ---- generate matrix A
n = 60; m=30
sqrtA = np.random.rand( n,n ) - 0.5
A = np.dot( sqrtA, np.transpose(sqrtA) )

# ---- full solve for eigenvalues for reference
esA, vsA = np.linalg.eig( A )

# ---- approximate solution by Lanczos
v0   = np.random.rand( n ); v0 /= np.sqrt( np.dot( v0, v0 ) )
T, V = Lanczos( A, v0, m=m )
esT, vsT = np.linalg.eig( T )
VV = np.dot( V, np.transpose( V ) ) # check orthogonality

#print "A : "; print A
#print (VV)
print ('exact eigenvalues are:')
print (np.sort(esA))
print (np.sort(esT))

plt.plot( esA, np.ones(n)*0.2,  '+' )
plt.plot( esT, np.ones(m)*0.1,  '+' )
plt.ylim(0,1)
plt.show()
