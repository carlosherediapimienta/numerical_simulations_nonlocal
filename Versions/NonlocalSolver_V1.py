import numpy as np
from scipy.integrate import solve_ivp, fixed_quad
from scipy.interpolate import interp1d
from concurrent.futures import ThreadPoolExecutor

from typing import Callable, Optional

class NonlocalSolver:
    def __init__(self, f: Callable, dL: Callable, t_span: list, y0: np.array, betas: list, alpha: float,
                  lambda_:float = 0, verbose: bool = True):
        
        np.random.seed(33)

        self.f = f
        self.t_span = t_span
        self.y0 = y0 
        self.t = np.linspace(t_span[0], t_span[1], t_span[1]*100)
        self.method_ode = 'RK45'
        self.atol_ode = 1e-8
        self.rtol_ode = 1e-8
        self.global_error_tolerance = 1e-5
        self.betas = betas
        self.alpha = alpha
        self.lambda_ = lambda_


        self.k = lambda i, s: self.alpha * np.exp(-((1-self.betas[i-1]) / self.alpha) * s, dtype='longdouble')            
        self.dL= dL

        self.gamma = lambda t: (1 - self.betas[0]) / (self.alpha * (1 - self.betas[0] ** t)) * np.sqrt((1 - self.betas[1] ** t) / (1 - self.betas[1]), dtype='longdouble')
        self.epsilon = 1e-8
        self.hat_epsilon = lambda t: self.alpha * np.sqrt((1 - self.betas[1] ** t) / (1 - self.betas[1]), dtype='longdouble') * self.epsilon

        self.smoothing_factor = 0.5
        self.smoothing_factor_max = 0.9999
        self.increments = np.linspace(self.smoothing_factor, self.smoothing_factor_max, num=100)
        self.max_value_index = False


        self.max_iteration = int(1e10)
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
            def integrand(i, tp):
                k_value = self.k(i, tp)
                df_value = self.dL(y_interpolated(t - tp))
                if i == 1:
                    return k_value * (df_value + 0.5 * self.lambda_ * y_interpolated(t - tp))
                else:
                    return k_value * (df_value + 0.5 * self.lambda_ * y_interpolated(t - tp)) ** 2

            result = []
            for i in range(self.y0.size):
                value_numerator, _ = fixed_quad(lambda tp: integrand(1, tp)[i], 0, t, n=100)
                value_denominator, _ = fixed_quad(lambda tp: integrand(2, tp)[i], 0, t, n=100)

                value_denominator = np.sqrt(value_denominator, dtype='longdouble') + self.hat_epsilon(t)
                result.append(value_numerator / value_denominator)
            return np.array(result)

        def rhs(t, y):
            return self.f(t, y_interpolated(t)) - self.gamma(t) * integral(t)

        return self.__solve_ode__(rhs)
    
    def __global_error__(self, y_new: np.array, y_guess: np.array) -> float:
        diff = y_new - y_guess
        return np.sqrt(np.vdot(diff, diff), dtype='longdouble')
    
    def __next_y__(self, y_current: np.array, y_guess: np.array) -> np.array:
        return (self.smoothing_factor * y_current) + ((1.0 - self.smoothing_factor) * y_guess)            
        
    def solve(self):
        self.iteration = 0

        y_current = self.__initial_solution__()
        y_guess = self.__rhs_with_integral_part__(y_current)
        current_error = self.__global_error__(y_current, y_guess)

        if self.verbose:
            print(f"Iteration {self.iteration} advanced. Current error: {current_error}.")

        last_error = current_error
        while current_error > self.global_error_tolerance:
            
            y_new = self.__next_y__(y_current, y_guess)
            y_guess = self.__rhs_with_integral_part__(y_new)
            current_error = self.__global_error__(y_new, y_guess)

            y_current = y_new
            self.iteration += 1

            if current_error > last_error:
                    if self.max_value_index:
                        print(f'Maximum value reached. The algorithm will stop without reaching the desired tolerance. The error is {current_error}.')
                        break

                    try:
                        next_factor = self.increments[np.searchsorted(self.increments, self.smoothing_factor, side='right')]
                    except IndexError:
                        next_factor = self.smoothing_factor_max
                        print(f'Smoothing factor is at maximum value.')
                        self.max_value_index = True

                    self.smoothing_factor = min(self.smoothing_factor_max, next_factor)
            last_error = current_error

            if self.verbose and self.iteration % 10 == 0:
                print(f"Iteration {self.iteration} advanced. Current error: {current_error}.")

            if self.iteration >= self.max_iteration:
                print(f"Maximum number of iterations reached. Current error: {current_error}.") 
                break
            
        print(f'Last iteration: {self.iteration}. Final error: {current_error}')

        self.y = y_guess
        self.global_error = current_error

        if self.y0.size == 1:
            self.y = self.y[0]

        return self.y            