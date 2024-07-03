import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def u0(x):
    return np.sin(np.pi * x**2) - x**2/ np.pi 

def integrand(y, u):
    return y * u(y)

def gauss_quadrature(func, a, b, nodes, weights):
    # Transformar nodos al intervalo [a, b]
    transformed_nodes = (b-a) * 0.5 * nodes + (b+a) * 0.5
    transformed_weights = (b-a) * 0.5 * weights 
    integral = np.sum(transformed_weights * func(transformed_nodes))
    return integral

# Parámetros
N = int(1e3)
t = np.linspace(0, 1, N)
iterations = 10
nodes, weights = np.polynomial.legendre.leggauss(20)

# Inicialización de la función
u_n = u0(t)

iteration = 0
while iteration < iterations: 
    u_new = []
    u_n_inter = interp1d(t, u_n, kind='linear', fill_value='extrapolate')
    for tp in t:
        # Cálculo de la integral usando cuadratura de Gauss
        integral = tp**2 * gauss_quadrature(lambda tp: integrand(tp, u_n_inter), 0, tp, nodes, weights)
        u_new.append(u0(tp) + integral)
    
    u_n = np.array(u_new)
    iteration += 1

# Visualización del resultado
plt.plot(t, u_n, "x", label="Solución", markersize=3)
plt.plot(t, u0(t), label="Solución exacta")
plt.xlabel("x")
plt.ylabel("U(x)")
plt.title("Solución de la ecuación de Picard")
plt.legend()
plt.show()