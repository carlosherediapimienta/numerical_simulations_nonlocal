import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def y0(t):
    return np.ones_like(t)

def integrand(t, y):
    return y(t)

def gauss_quadrature(func, a, b, nodes, weights):
    # Transformar nodos al intervalo [a, b]
    transformed_nodes = (b-a) * 0.5 * nodes + (b+a) * 0.5
    transformed_weights = (b-a) * 0.5 * weights 
    integral = np.sum(transformed_weights * func(transformed_nodes))
    return integral

# Parámetros
N = int(1e3)
t = np.linspace(0, 1, N)
iterations = int(1e2)
nodes, weights = np.polynomial.legendre.leggauss(20)

# Inicialización de la función
y_n = y0(t)

iteration = 0
while iteration < iterations: 
    y_new = []
    y_n_inter = interp1d(t, y_n, kind='linear', fill_value='extrapolate')
    for tp in t:
        # Cálculo de la integral usando cuadratura de Gauss
        integral = gauss_quadrature(lambda tp: integrand(tp, y_n_inter), 0, tp, nodes, weights)
        y_new.append(y0(tp) + integral)
    
    y_n = np.array(y_new)
    iteration += 1

# Visualización del resultado
plt.plot(t, y_n, "x", label="Solución", markersize=3)
plt.plot(t, np.exp(t), label="Solución exacta")
plt.xlabel("x")
plt.ylabel("U(x)")
plt.title("Solución de la ecuación de Picard")
plt.legend()
plt.show()