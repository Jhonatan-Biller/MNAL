import numpy as np

def subs_dir(A):

    m, n = A.shape
    x    = np.zeros(m)
    x[0] = A[0,-1] / A[0,0]
    
    for i in range(1,m):
        soma = 0
        
        for j in range(0,i+1):
            soma = soma + A[i,j] * x[j]
        
        x[i] = (A[i,-1] - soma) / A[i,i]

    return x


def subs_reg(A):

    m, n = A.shape

    x = np.zeros(m)
    
    x[m-1] = A[m-1,-1] / A[m-1,-2]

    for i in range(m-2,-1,-1):
    
        soma = 0
        
        for j in range(i+1,n-1):
            
            soma = soma + A[i,j] * x[j]
        
        x[i] = (A[i,-1] - soma) / A[i,i]

    return x

def palu(A):
    
    n = len(A)
    P = np.eye(n)
    L = np.eye(n)
    U = A.copy()

    for k in range(n - 1):
        
        idc = np.argmax(np.abs(U[k:, k])) + k
        
        if U[idc, k] != 0:
            
            U[[k, idc], :] = U[[idc, k], :]
            P[[k, idc], :] = P[[idc, k], :]
            if k > 0:
                L[[k, idc], :k] = L[[idc, k], :k]

        for i in range(k + 1, n):
            if U[k, k] != 0:
                L[i, k] = U[i, k] / U[k, k]
                U[i, :] = U[i, :] - L[i, k] * U[k, :]

    return P, L, U

