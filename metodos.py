import numpy as np

def piv_parc(A, k):

    i = np.argmax(np.abs(A[k:, k]))
    
    A[[k, i+k], :] = A[[i+k, k], :]

    return A

def piv_tot(A, k, col):

    C = A[k:,k:-1]

    i, j = np.unravel_index(np.argmax(np.abs(C)), C.shape)
    
    A[[k, i+k], :] = A[[i+k, k], :]
    A[:, [k, j+k]] = A[:, [j+k, k]]
    
    col[k], col[j+k] = col[j+k], col[k]

    return A, col 

def reord_x(col, x):
    return x[col]

import numpy as np

def escalonamento(A):

    m, n = A.shape

    for i in range(0,m):

      s = np.max(np.abs(A[i, i:]))
        
      if s != 0:
          
        A[i,:] = A[i,:] / s

    A = piv_parc(A,0)
    
    for i in range(0,m):

        c = np.abs(A[i,i])
        
        if c != 0:
                        
            A[i,:] = A[i,:] / c
    
    return A

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

def subs_dir(A):

    m, n = A.shape

    x = np.zeros(m)

    x[0] = A[0,-1] / A[0,0]
    
    for i in range(1,m):
        
        soma = 0
        
        for j in range(0,i+1):
            
            soma = soma + A[i,j] * x[j]
        
        x[i] = (A[i,-1] - soma) / A[i,i]

    return x

def gauss(A,piv):
    
    m   = A.shape[0]
    M   = np.zeros_like(A, dtype=float)
    col = list(range(0,m))

    for k in range(0, m - 1):
        
        if   piv == 'tot' :
            A, col = piv_tot(A, k,col) 
            
        elif piv == 'parc':
            A      = piv_parc(A, k) 

        else:
            return []
        
        if A[k, k] != 0: 
            for i in range(k + 1, m):
                M[i, k] = A[i, k] / A[k, k]  
                A[i, :] = A[i, :] - M[i, k] * A[k, :]  

    if piv =='tot':
        return M, A, col
    return M, A

def lu(A):

    n = len(A)
    L = np.eye(n,n)
    
    for k in range(0,n-1):

        if A[k,k] == 0:
                return [],[]
        
        for i in range(k+1,n):
                    
            L[i,k] = A[i,k] / A[k,k]
            A[i,:] = A[i,:] - L[i,k] * A[k,:]
    
    return L, A 

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

def ldv(A):
    
    L, U = lu(A)
    
    D = np.zeros_like(U)
    
    np.fill_diagonal(D, np.diag(U))
    
    for k in range(0,len(D)):
        
        U[k,:] /= D[k,k]

    return L, D, U

def ldl_cholesky(A):

    L, U = lu(A)

    D = np.zeros_like(A)
    
    d = np.sqrt(np.diag( U ))

    np.fill_diagonal(D, d)
    
    G = L @ D
    
    return G

def gaxpy_cholesky(A):

    n = len(A)
    
    for j in range(0,n):
        
        if j > 0:

            A[j:n,j] = A[j:n,j] - A[j:n,0:j] @ A[j,0:j].T
        
        A[j:n,j] = A[j:n,j] / np.sqrt(A[j,j])
        
    return np.tril(A)

def cholesky(A):

    n = len(A)

    for k in range(0,n):   
        s = 0
        
        for i in range(0,k):
            s = s + np.square(A[k,i])
            
        s = A[k,k] - s

        if s <= 0:
            return []
            
        A[k,k] = np.sqrt(s)
        
        for j in range(k+1,n):
            s = 0
            
            for i in range(0,k):    
                s = s + A[j,i]*A[k,i]
                
            A[j,k] = (A[j,k]-s)/A[k,k]

    return np.tril(A)
    
def inv(A):

    P, L, U = palu(A)
    n       = len(A)
    Ainv    = np.zeros_like(A)
    
    for i in range(0,n):

        ei    = np.zeros(n)
        ei[i] = 1

        y = subs_dir(np.column_stack((L, P @ ei)))  # Ly = P@ei
    
        Ainv[:, i] = subs_reg(np.column_stack((U, y))) # Ux = y

    return Ainv    

def qr_gramschmidt_classico(A):
    m, n = A.shape

    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    R[0, 0] = np.linalg.norm(A[:, 0])
    Q[:, 0] = A[:, 0] / R[0, 0]

    for k in range(1, n):
        
        R[:k, k] = Q[:, :k].T @ A[:, k]
        z        = A[:, k] - Q[:, :k] @ R[:k, k]
        R[k, k]  = np.linalg.norm(z)
        Q[:, k]  = z / R[k, k]

    return Q, R

#B = np.array([[1,0,2],[0,1,1],[1,2,0]],dtype=float)