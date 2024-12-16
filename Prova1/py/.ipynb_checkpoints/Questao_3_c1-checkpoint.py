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
    
def gauss_sem_pivotacao(A):
    
    m   = A.shape[0]
    M   = np.zeros_like(A, dtype=float)

    for k in range(0, m - 1):
        
        for i in range(k + 1, m):
            M[i, k] = A[i, k] / A[k, k]  
            A[i, :] = A[i, :] - M[i, k] * A[k, :]  
                
    return A

# H(6)
H = hilbert(6,6)

# Vetor b
b = np.array([1, 1, 1, 1, 1, 1])

# Construindo a matriz [H|b]
Hb = np.column_stack((H,b))

# Obtém triangular superior de [H|b]
Ht = gauss_sem_pivotacao(Hb.copy())

# Obtém uma aproximação de x tal que Hx=b:
x_gauss_sem_piv = subs_reg(Ht.copy())

# Exibindo resultados
print('A solução encontrada pela eliminação de Gauss sem pivoteamento é:\n')

for i in range(0,6):
    print('x{} = {}'.format(i+1,x_gauss_sem_piv[i]))
    
# Vejamos que é solução
print('\n||H * x - b||_2 = ', np.linalg.norm(H @ x_gauss_sem_piv - b))