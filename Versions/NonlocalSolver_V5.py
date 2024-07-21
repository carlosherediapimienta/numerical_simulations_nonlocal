import numpy as np
from scipy.integrate import fixed_quad
from scipy.interpolate import interp1d
from concurrent.futures import ThreadPoolExecutor
from numba import njit

from typing import Callable

class NonlocalSolver:
    def __init__(self, f: Callable, dL: Callable, t_span: list, y0: np.array, betas: list,
                 alpha: float, lambda_:float = 0, verbose: bool = True):
        
        np.random.seed(33)

        self.f = f
        self.t_span = t_span
        self.y0 = y0 
        self.alpha = alpha
        self.t = np.arange(t_span[0], t_span[1], alpha)
        self.betas = betas
        self.lambda_ = lambda_

        self.k = lambda i, s: self.alpha * np.exp(- ((1 - self.betas[i-1]) / (self.alpha)) * s)   
        self.dL = dL
        
        step_t_betas =lambda t: t + 1
        self.gamma = lambda t: (1 - self.betas[0]) / (self.alpha * (1 - self.betas[0] ** step_t_betas(t))) * np.sqrt((1 - self.betas[1] ** step_t_betas(t)) / (1 - self.betas[1]))
        self.epsilon = 1e-8

        self.hat_epsilon = lambda t: self.alpha * np.sqrt((1 - self.betas[1] ** step_t_betas(t)) / (1 - self.betas[1])) * self.epsilon

        self.smoothing_factor = 0.5
        self.smoothing_factor_max = 0.9999
        self.increments = np.linspace(self.smoothing_factor, self.smoothing_factor_max, num=int(1e2))
        self.max_value_index = False

        self.max_iteration = int(1e10)
        self.global_error_tolerance = 1e-4
        self.verbose = verbose

    def __initial_solution__(self) -> np.array:
        return self.__solve_ode__(self.f)
    
    def __solve_ode__(self, rhs_ode: Callable) -> np.array:
        t_values = self.t
        y_values = np.zeros(t_values.shape)
        y_values[0] = self.y0
        for i in range(1, len(t_values)):
            y_values[i] = y_values[i - 1] + self.alpha * rhs_ode(t_values[i - 1], y_values[i - 1])
        return y_values
    
    def __rhs_with_integral_part__(self, y: np.array) -> np.array:
        y_interpolated = interp1d(self.t, y, kind='cubic', fill_value="extrapolate", assume_sorted=True)

        def integral(t):
            def integrand(i, tp):
                k_value = self.k(i, tp)
                df_value = self.dL(y_interpolated(t - tp))
                common_term = df_value + 0.5 * self.lambda_ * y_interpolated(t - tp)
                return k_value * common_term if i == 1 else k_value * (common_term ** 2)
            
            numerador_func = lambda tp: integrand(1, tp)
            denominador_func = lambda tp: integrand(2, tp)

            with ThreadPoolExecutor() as executor:
                future_numerator = executor.submit(fixed_quad, numerador_func, 0, t, n=int(1e2))
                future_denominator = executor.submit(fixed_quad, denominador_func, 0, t, n=int(1e2))

                value_numerator, _ = future_numerator.result()
                value_denominator, _ = future_denominator.result()

            value_denominator = np.sqrt(value_denominator) + self.hat_epsilon(t)
            return np.array(value_numerator / value_denominator)

        def rhs(t, y):
            return self.f(t, y_interpolated(t)) - self.gamma(t) * integral(t)

        return self.__solve_ode__(rhs)
    
    @staticmethod
    @njit(parallel=True)
    def __global_error__(y_new: np.array, y_guess: np.array) -> float:
        diff = y_new - y_guess
        return np.sqrt(np.sum(diff ** 2))
    
    @staticmethod
    @njit(parallel=True)
    def __next_y__(smoothing_factor: float, y_current: np.array, y_guess: np.array) -> np.array:
        return (smoothing_factor * y_current) + ((1.0 - smoothing_factor) * y_guess)            
        
    def solve(self):
        self.iteration = 0

        y_current = self.__initial_solution__()
        y_guess = self.__rhs_with_integral_part__(y_current)
        current_error = self.__global_error__(y_current, y_guess)

        if self.verbose:
            print(f"Iteration {self.iteration} advanced. Current error: {current_error}.")

        last_error = current_error
        while current_error > self.global_error_tolerance:
            
            y_new = self.__next_y__(self.smoothing_factor, y_current, y_guess)
            y_guess = self.__rhs_with_integral_part__(y_new)
            current_error = self.__global_error__(y_new, y_guess)

            y_current = y_new
            self.iteration += 1

            if current_error > last_error:
                    if self.max_value_index:
                        print(f'Maximum value of the smoothing factor reached. The algorithm will stop without reaching the desired tolerance. The error is {current_error}.')
                        break

                    try:
                        next_factor = self.increments[np.searchsorted(self.increments, self.smoothing_factor, side='right')]
                    except IndexError:
                        next_factor = self.smoothing_factor_max
                        print(f'Smoothing factor is at maximum value.')
                        self.max_value_index = True

                    self.smoothing_factor = min(self.smoothing_factor_max, next_factor)
            last_error = current_error

            if self.verbose and self.iteration % 1 == 0:
                print(f"Iteration {self.iteration} advanced. Current error: {current_error}.")

            if self.iteration >= self.max_iteration:
                print(f"Maximum number of iterations reached. Current error: {current_error}.") 
                break
            
        print(f'Last iteration: {self.iteration}. Final error: {current_error}')

        self.y = y_guess
        self.global_error = current_error

        return self.t, self.y           