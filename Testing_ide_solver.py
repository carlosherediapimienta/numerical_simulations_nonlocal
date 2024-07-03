import numpy as np
from idesolver import IDESolver
import matplotlib.pyplot as plt



solver = IDESolver(
    x = np.linspace(0, 1, 100),
    y_0 = 1,
    c = lambda x, y: -2 * y,
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

exact = np.exp(-solver.x) * np.cos(2 * solver.x)

ax.plot(solver.x, solver.y, label = 'IDESolver Solution', linestyle = '-', linewidth = 3)
ax.plot(solver.x, exact, label = 'Analytic Solution', linestyle = ':', linewidth = 3)

ax.legend(loc = 'best')
ax.grid(True)

ax.set_title(f'Solution for Global Error Tolerance = {solver.global_error_tolerance}')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y(x)$')

plt.show()