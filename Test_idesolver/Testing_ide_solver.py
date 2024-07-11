from idesolver import IDESolver
import matplotlib.pyplot as plt
import numpy as np

N= int(1e3)
x_i = 0
x_f = 3

solver = IDESolver(
    x = np.linspace(x_i, x_f, N),
    y_0 = 0,
    c = lambda x, y: 1 - 2 * y,
    d = lambda x: -5,
    k = lambda x, s: 1,
    f = lambda y: y,
    lower_bound = lambda x: 0,
    upper_bound = lambda x: x,
)

solver.solve()

solver.x  
solver.y 


fig = plt.figure(figsize = (10, 6))
ax = fig.add_subplot(111)

exact = 0.5 * np.exp(- solver.x) * np.sin(2 * solver.x)

ax.plot(solver.x, solver.y, label = 'IDESolver Solution', linestyle = '-', linewidth = 3)
ax.plot(solver.x, exact, label = 'Analytic Solution', linestyle = ':', linewidth = 3)

ax.legend(loc = 'best')
ax.grid(True)

ax.set_title(f'Solution for Global Error Tolerance = {solver.global_error_tolerance}')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y(x)$')

plt.show()