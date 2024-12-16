import numpy as np
import sympy as sp

# Define as variáveis simbólicas
x, y, z = sp.symbols('x y z')

# Define as funções do sistema
f1 = x**2 + y**2 + z**3 - 9
f2 = x * y * z - 1
f3 = x + y - z**2

# Define o vetor de funções
F = sp.Matrix([f1, f2, f3])

# Calcula a Jacobiana
J = F.jacobian([x, y, z])

#Chute inicial
xi = np.array([2.0, 1.0, 1.0])

# Tolerância
tol = 10e-8

# Máximo de iterações
max_iter = 100

for i in range(max_iter):
    # Avalia F(x) no ponto atual xi
    Fx = np.array([f.subs({x: xi[0], y: xi[1], z: xi[2]}).evalf() for f in F], dtype=float)
    
    # Avalia J(x) no ponto atual xi
    Jx = np.array(J.subs({x: xi[0], y: xi[1], z: xi[2]}).evalf(), dtype=float)
        
    # Resolve o sistema linear J(x) * Δx = -F(x)
    delta_x = np.linalg.solve(Jx, -Fx)
    
    xi = xi + delta_x
        
    # Critério de parada
    if np.linalg.norm(delta_x) < tol:
        print(f"Convergiu em {i+1} iterações.")
        break
else:
    print("Não convergiu dentro do número máximo de iterações.")

print("Solução aproximada:", xi)

# Vejamos que [2.22424483 0.28388497 1.58370761] é uma boa aproximação para a solução do sistema
print('F(x):',np.array([xi[0]**2+xi[1]**2+xi[2]**3-9,xi[0]*xi[1]*xi[2]-1,xi[0]+xi[1]-xi[2]**2]))