import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def u_real(x):
    return 0.5 * np.exp(-x) * np.sin(2 * x) # Solución exacta
    # return np.exp(-x) * np.cos(2 * x)

def u0(x):
    #return 0.5*(1 - np.exp(-2 * x)) # Suposición inicial
    return x

def first_integrand(t, u):
    return u(t)

def second_integrand(xp, u, nodes, weights):  
    return gauss_quadrature(lambda tp: first_integrand(tp, u), 0, xp, nodes, weights)

def gauss_quadrature(func, a, b, nodes, weights):
    # Transformar nodos al intervalo [a, b]
    transformed_nodes = (b - a) * 0.5 * nodes + (b + a) * 0.5
    transformed_weights = (b - a) * 0.5 * weights 
    integral = np.sum(transformed_weights * func(transformed_nodes))
    return integral

# Parámetros
N = int(1e3)
t = np.linspace(0, 3, N)
iterations = 50
nodes, weights = np.polynomial.legendre.leggauss(20)

# Inicialización de la función
u_n = u0(t)

iteration = 0
while iteration < iterations: 
    u_new = []
    u_n_inter = interp1d(t, u_n, kind='linear', fill_value='extrapolate')
    for tp in t:
        # Cálculo de la integral usando cuadratura de Gauss
        integral_rhs = - 5 * gauss_quadrature(lambda xp: second_integrand(xp, u_n_inter, nodes, weights), 0, tp, nodes, weights)
        integral_lhs = - 2 * gauss_quadrature(lambda xp: first_integrand(xp, u_n_inter), 0, tp, nodes, weights)
        integral = integral_rhs + integral_lhs
        u_new.append(u0(tp) + integral)
    
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
