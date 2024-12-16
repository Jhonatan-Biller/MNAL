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

# Exibe  Jacobiana
print('Jacobiana:\n',J)