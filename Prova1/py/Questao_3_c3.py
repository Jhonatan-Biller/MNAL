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

# H(6)
H = hilbert(6,6)

# Vetor b
b = np.array([1, 1, 1, 1, 1, 1])

# Obtendo a fatoração LU de H(6):
L, U = lu(H)

# Ly = b 
yl = subs_dir(np.column_stack((L,b)))

# Ux = y
xl = subs_reg(np.column_stack((U,yl)))

# Exibindo resultados
print('Solução por Fatoração LU:\n')

for i in range(0,6):
    print('x{} = {}'.format(i+1,xl[i]))
    
#Vejamos que é solução
print('\n||H * x - b||_2 = ', np.linalg.norm(H @ xl - b))