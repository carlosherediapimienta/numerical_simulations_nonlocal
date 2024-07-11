import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import interp1d

from typing import Callable, Optional

class NonlocalSolver:
    def __init__(self, f: Callable, t_span: list, y0: np.array, max_iterations: int, verbose: bool = True):
        self.f = f
        self.t_span = t_span
        self.y0 = y0
        self.t = np.linspace(t_span[0], t_span[1], int(1e3))
        self.method_ode = 'RK45'
        self.atol_ode = 1e-8
        self.rtol_ode = 1e-8
        self.atol_integral = 1e-8
        self.rtol_integral = 1e-8
        self.global_error_tolerance = 1e-6
        self.k = lambda x, s: x / (1+s)
        self.F= lambda y: y
        self.gamma = lambda t: 1/(np.log(2)**2)
        self.smoothing_factor = 0.5
        self.max_iteration = max_iterations
        self.verbose = verbose


    def __initial_solution__(self) -> np.array:
        return self.__solve_ode__(self.f)
    
    def __solve_ode__(self, rhs_ode: Callable) -> np.array:
        solution = solve_ivp(fun=rhs_ode, t_span=self.t_span, y0=self.y0, t_eval=self.t, 
                             method=self.method_ode, atol=self.atol_ode, rtol=self.rtol_ode)         
        return solution.y
    
    def __rhs_with_integral_part__(self, y: np.array) -> np.array:
        y_interpolated = interp1d(self.t, y, kind='cubic', fill_value="extrapolate", assume_sorted=True)

        def integral(t):
            def integrand(tp):
                return self.k(t,tp) * self.F(y_interpolated(tp))
            
            result = []
            for i in range(self.y0.size):
                r, *_ = quad(lambda tp: integrand(tp)[i], 0, 1, epsabs=self.atol_integral, epsrel=self.rtol_integral)
                result.append(r)

            return np.array(result)
        
        def rhs(t, y):
            return self.f(t, y_interpolated(t)) + (self.gamma(t) * integral(t))
        
        return self.__solve_ode__(rhs)
    
    def __global_error__(self, y_new: np.array, y_guess: np.array) -> float:
        diff = y_new - y_guess
        return np.sqrt(np.vdot(diff, diff))
    
    def __next_y__(self, y_current: np.array, y_guess: np.array) -> np.array:
        return (self.smoothing_factor * y_current) +((1 - self.smoothing_factor) * y_guess)            
        
    def solve(self):
        y_0 = self.__initial_solution__()
        y_guess = self.__rhs_with_integral_part__(y_0)
        current_error = self.__global_error__(y_0, y_guess)

        self.iteration = 0
        if self.verbose:
            print(f"Iteration {self.iteration} advanced. Current error: {current_error}.")

        y_current = y_0
        while current_error > self.global_error_tolerance:

            y_new = self.__next_y__(y_current, y_guess)
            y_guess = self.__rhs_with_integral_part__(y_new)
            current_error = self.__global_error__(y_new, y_guess)

            y_current = y_new
            self.iteration += 1
            
            if self.verbose:    
                print(f"Iteration {self.iteration} advanced. Current error: {current_error}.")

            if self.max_iteration is not None and self.iteration >= self.max_iteration:
                print(f"Maximum number of iterations reached. Current error: {current_error}.") 
                break

        self.y = y_guess
        self.global_error = current_error

        if self.y0.size == 1:
            self.y = self.y[0]

        return self.y            
        
t = [0, 5]
f = lambda x, y: y - 0.5 * x + 1/(1+x) - np.log(1+x) 
solver = NonlocalSolver(f=f, t_span=t, y0=np.array([0]), max_iterations=100)
solver.solve()

############################################################################################################
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (10, 6))
ax = fig.add_subplot(111)

#exact = 0.5 * np.exp(- solver.t) * np.sin(2 * solver.t)
exact = np.log(1+solver.t)

ax.plot(solver.t, solver.y, label = 'IDESolver Solution', linestyle = '-', linewidth = 3)
ax.plot(solver.t, exact, label = 'Analytic Solution', linestyle = ':', linewidth = 3)

ax.legend(loc = 'best')
ax.grid(True)

ax.set_title(f'Solution for Global Error Tolerance = {solver.global_error_tolerance}')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y(x)$')

plt.show()
    
