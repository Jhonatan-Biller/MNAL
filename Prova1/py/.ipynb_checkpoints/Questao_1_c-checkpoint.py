import numpy as np 

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

# Fazendo Ad = D[:,:-1]
Ad = D[:,:-1]

# Verificando se Ad é simétrica
if np.array_equal(Ad, Ad.T):
    print('A matriz é simétrica.')
else:
    print('A matriz não é simétrica.')