import numpy as np 

def hilbert(m,n):

    H = np.zeros((m,n), dtype=float)

    for i in range(0,m):

        for j in range(0,n):

            H[i,j] = 1/(i+j+1)
            
    return H

# H(6)
H = hilbert(6,6)

# Obtendo H^{-1}
H_inv = np.linalg.inv(H.copy())

# Calculando número de condicionamento usando || . ||_2
k = np.linalg.norm(H) * np.linalg.norm(H_inv)

#Exibindo resultados
print('O número de condicionamento k_2(H) é:',k)
print('O número de condicionamento arredondado é k_2(H)={:.2e}'.format(k))