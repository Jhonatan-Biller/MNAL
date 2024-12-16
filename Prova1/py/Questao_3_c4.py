import numpy as np 

def hilbert(m,n):

    H = np.zeros((m,n), dtype=float)

    for i in range(0,m):

        for j in range(0,n):

            H[i,j] = 1/(i+j+1)
            
    return H

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

def gaxpy_cholesky(A):

    n = len(A)
    
    for j in range(0,n):
        
        if j > 0:

            A[j:n,j] = A[j:n,j] - A[j:n,0:j] @ A[j,0:j].T
        
        A[j:n,j] = A[j:n,j] / np.sqrt(A[j,j])
        
    return np.tril(A)

# H(6)
H = hilbert(6,6)

# Vetor b
b = np.array([1, 1, 1, 1, 1, 1],dtype=float)

# Verificando se H é simétrica
if np.array_equal(H, H.T): print('H é simétrica.')

# Obtendo G
G = gaxpy_cholesky(H)

# Gy = b
yg = subs_dir(np.column_stack((G,b)))

# G^tx = y
xg = subs_reg(np.column_stack((G.T,yg)))

# Exibindo resultados
print('A solução por Cholesky é:\n')

for i in range(0,6):
    print('x{} = {}'.format(i+1,xg[i]))
    
# Vejamos que é solução | essa é a *GAXPY cholesky*
print('\n||H * x - b||_2 = ', np.linalg.norm(H @ xg - b))