import numpy as np

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

# --- Definições iniciais

# Definições do domínio
a = 0; b = 10; h = 1/1000

# Definições da Equação Diferencial
c = 0.0001; d = 10000; u0 = 0; un = np.cos(b)

# Definindo o termo fonte:
def f(x):
    return x**2

# --- Inicio do algoritmo

# Criando vetor com os pontos de malha
vx = np.arange(a,b,h)
#vx = np.append(vx,b)

# Calculando constantes
A =  1/np.square(h) - c/(2*h)
B = -2/np.square(h) + d
C =  1/np.square(h) + c/(2*h)

# Criando matriz de diferenças D
m = len(vx); n = m + 1
D = np.zeros((m,n))

# Construindo a primeira linha da matriz
D[0, 0] = B
D[0, 1] = C
D[0,-1] = f(a) - u0 * A

# Construindo linhas intermediárias
for i in range(1,m-1):

    D[i,i-1:i+2] = np.array([A,B,C])
    D[i,-1]      = f(vx[i])

# Construindo última linha
D[m-1,-3] = A
D[m-1,-2] = B
D[m-1,-1] = f(b) - un * C

# Obtendo a fatoração LU
L, U = lu(D[:,:-1])

# Resolvendo Ly = b
yl = subs_dir(np.column_stack((L,D[:,-1])))

# Resolvendo Ux = y
xl = subs_reg(np.column_stack((U,yl)))

# Exibindo o indice de vx tal que vx = 5
print('No indice {} temos que v{}=5.'.format(np.where(vx==5)[0]))

# Exibindo a solução em x=5 
print('A solução por fatoração LU em x=5 é',xl[5000])