import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def u_real(x):
    return np.exp(-x) * np.cos(2 * x)

def u0(x):
    return np.exp(-2 * x)  # Suposición inicial

def first_integrand(t, u):
    return u(t)

def second_integrand(xp, u, nodes, weights):  
    return 2 * u(xp) + 5 * gauss_quadrature(lambda tp: first_integrand(tp, u), 0, xp, nodes, weights)

def gauss_quadrature(func, a, b, nodes, weights):
    # Transformar nodos al intervalo [a, b]
    transformed_nodes = (b - a) * 0.5 * nodes + (b + a) * 0.5
    transformed_weights = (b - a) * 0.5 * weights 
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
        integral = 1 - gauss_quadrature(lambda xp: second_integrand(xp, u_n_inter, nodes, weights), 0, tp, nodes, weights)
        u_new.append(integral)
    
    u_n = np.array(u_new)
    iteration += 1

# Visualización del resultado
plt.plot(t, u_n, "x", label="Solución", markersize=3)
plt.plot(t, u_real(t), label="Solución exacta")
plt.xlabel("x")
plt.ylabel("U(x)")
plt.title("Solución de la ecuación de Picard")
plt.legend()
plt.grid(True)
plt.show()
