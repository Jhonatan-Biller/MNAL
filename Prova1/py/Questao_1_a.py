import numpy as np 

def piv_parc(A, k):

    i = np.argmax(np.abs(A[k:, k]))
    
    A[[k, i+k], :] = A[[i+k, k], :]

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

def gauss(A):
    
    m   = A.shape[0]
    M   = np.zeros_like(A, dtype=float)
    

    for k in range(0, m - 1):
             
        A      = piv_parc(A, k) 
   
        if A[k, k] != 0: 
            for i in range(k + 1, m):
                M[i, k] = A[i, k] / A[k, k]  
                A[i, :] = A[i, :] - M[i, k] * A[k, :]  

    return A
    
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


# Obtendo triangular inferior
D_tri_inf = gauss(D) 

# Obtendo solução do sistema (ui)
x_gauss = subs_reg(D_tri_inf)

# Obtendo o indice de vx tal que vx = 5
print('No indice {} temos que v{}=5.'.format(np.where(vx==5)[0]))

# Obtendo a solução em x = 5
print('A solução por eliminação de Gauss com pivotação parcial em x = 5 é',x_gauss[5000])