import numpy as np 

def hilbert(m,n):

    H = np.zeros((m,n), dtype=float)

    for i in range(0,m):

        for j in range(0,n):

            H[i,j] = 1/(i+j+1)
            
    return H

H = hilbert(6,6)

print('Matriz de Hilbert H(6):\n',H)