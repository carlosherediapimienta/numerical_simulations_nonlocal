from scipy.integrate import fixed_quad
from scipy.interpolate import interp1d
from concurrent.futures import ThreadPoolExecutor
from numba import njit
from typing import Callable
import numpy as np

class NonlocalSolverMomentum:
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
        self.k = lambda i, s: (1-betas[i-1]) * np.exp( - ((1-betas[i-1]) / alpha) * s)
        self.dL = dL
        self.alpha_t = lambda t: np.sqrt((1-betas[1]**(t/alpha))/alpha) / (1-betas[0]**(t/alpha)) 
        self.epsilon_t = lambda t: np.sqrt(alpha*(1-betas[1]**(t/alpha))) * 1e-8
    

        self.smoothing_factor = 0.5
        self.smoothing_factor_max = 0.9999
        self.increments = np.linspace(self.smoothing_factor, self.smoothing_factor_max, num=int(1e3))
        self.max_value_index = False

        self.max_iteration = int(1e10)
        self.global_error_tolerance = 1e-4
        self.verbose = verbose


    def __initial_solution__(self) -> np.array:
        return self.__solve_ode__(self.f)
    
    def __solve_ode__(self, rhs_ode: Callable) -> np.array:
        t_values = self.t
        y_values = np.zeros(t_values.shape)
        y_values[0] = self.y0.item() if isinstance(self.y0, np.ndarray) else self.y0
        for i in range(1, len(t_values)):
            y_values[i] = y_values[i - 1] + self.alpha * rhs_ode(t_values[i - 1], y_values[i - 1])
        return y_values
    
    def __rhs_with_integral_part__(self, y: np.array) -> np.array:
        y_interpolated = interp1d(self.t, y, kind='cubic', fill_value="extrapolate", assume_sorted=True)

        self.m = []
        self.v = []

        def integral(t):
            def integrand(i, tp):
                k_value = self.k(i, t - tp)
                df_value = self.dL(y_interpolated(tp))
                common_term = df_value + 0.5 * self.lambda_ * y_interpolated(tp)
                return k_value * common_term if i == 1 else k_value * (common_term ** 2)
            
            numerador_func = lambda tp: integrand(1, tp)
            denominador_func = lambda tp: integrand(2, tp)

            with ThreadPoolExecutor() as executor:
                future_numerator = executor.submit(fixed_quad, numerador_func, 0, t, n=int(1e3))
                future_denominator = executor.submit(fixed_quad, denominador_func, 0, t, n=int(1e3))

                value_numerator, _ = future_numerator.result()
                value_denominator, _ = future_denominator.result()

            v_value = value_denominator
            v_sqrt_value = np.sqrt(v_value) 
            m_value = value_numerator

            self.m.append((t, m_value))
            self.v.append((t, v_value))   

            return m_value / (v_sqrt_value + self.epsilon_t(t))

        def rhs(t, y):
            return self.f(t, y_interpolated(t)) - self.alpha_t(t) * integral(t)
                
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


class NonlocalSolverMomentumRMSProp:
    def __init__(self, f: Callable, dL: Callable, t_span: list, y0: np.array, beta: float,
                 alpha: float, lambda_:float = 0, verbose: bool = True):
        
        np.random.seed(33)

        self.f = f
        self.t_span = t_span
        self.y0 = y0 
        self.alpha = alpha
        self.t = np.arange(t_span[0], t_span[1], alpha)
        self.beta = beta
        self.lambda_ = lambda_
        self.k = lambda s: (1-self.beta) / self.alpha * np.exp( - ((1-beta) / alpha) * s)
        self.dL = dL
        self.epsilon = 1e-8
    

        self.smoothing_factor = 0.5
        self.smoothing_factor_max = 0.9999
        self.increments = np.linspace(self.smoothing_factor, self.smoothing_factor_max, num=int(1e3))
        self.max_value_index = False

        self.max_iteration = int(1e10)
        self.global_error_tolerance = 1e-4
        self.verbose = verbose


    def __initial_solution__(self) -> np.array:
        return self.__solve_ode__(self.f)
    
    def __solve_ode__(self, rhs_ode: Callable) -> np.array:
        t_values = self.t
        y_values = np.zeros(t_values.shape)
        y_values[0] = self.y0.item() if isinstance(self.y0, np.ndarray) else self.y0
        for i in range(1, len(t_values)):
            y_values[i] = y_values[i - 1] + self.alpha * rhs_ode(t_values[i - 1], y_values[i - 1])
        return y_values
    
    def __rhs_with_integral_part__(self, y: np.array) -> np.array:
        y_interpolated = interp1d(self.t, y, kind='cubic', fill_value="extrapolate", assume_sorted=True)

        self.v = []

        def integral(t):
            def integrand(tp):
                k_value = self.k(tp)
                df_value = self.dL(y_interpolated(t-tp))
                common_term = df_value + 0.5 * self.lambda_ * y_interpolated(t-tp)
                return k_value * (common_term ** 2)
            
            v_value, _ = fixed_quad(lambda tp: integrand(tp), 0, t+self.alpha, n=int(1e3))
            self.v.append((t, v_value))

            G_sqrt_value = np.sqrt(v_value)  
            return np.array(self.dL(y_interpolated(t)) / (G_sqrt_value + self.epsilon))

        def rhs(t, y):
            return self.f(t, y_interpolated(t)) - integral(t)
                
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

class NonlocalSolverAdaGrad:
    def __init__(self, f: Callable, dL: Callable, t_span: list, y0: np.array, lr_decay: float=0,
                 alpha: float=0.01, lambda_:float = 0, verbose: bool = True):
        
        np.random.seed(33)

        self.f = f
        self.t_span = t_span
        self.y0 = y0 
        self.alpha = alpha
        self.t = np.arange(t_span[0], t_span[1], alpha)
        self.lambda_ = lambda_
        self.k = lambda s: 1 / alpha
        self.dL = dL
        self.alpha_t = lambda s: 1 / (1 + s * (lr_decay / alpha))
    

        self.smoothing_factor = 0.5
        self.smoothing_factor_max = 0.9999
        self.increments = np.linspace(self.smoothing_factor, self.smoothing_factor_max, num=int(1e3))
        self.max_value_index = False

        self.max_iteration = int(1e10)
        self.global_error_tolerance = 1e-4
        self.verbose = verbose


    def __initial_solution__(self) -> np.array:
        return self.__solve_ode__(self.f)
    
    def __solve_ode__(self, rhs_ode: Callable) -> np.array:
        t_values = self.t
        y_values = np.zeros(t_values.shape)
        y_values[0] = self.y0.item() if isinstance(self.y0, np.ndarray) else self.y0
        for i in range(1, len(t_values)):
            y_values[i] = y_values[i - 1] + self.alpha * rhs_ode(t_values[i - 1], y_values[i - 1])
        return y_values
    
    def __rhs_with_integral_part__(self, y: np.array) -> np.array:
        y_interpolated = interp1d(self.t, y, kind='cubic', fill_value="extrapolate", assume_sorted=True)

        self.v = []

        def integral(t):
            def integrand(tp):
                k_value = self.k(t-tp)
                df_value = self.dL(y_interpolated(tp))
                common_term = df_value + 0.5 * self.lambda_ * y_interpolated(tp)
                return k_value * (common_term ** 2)
            
            v_value, _ = fixed_quad(lambda tp: integrand(tp), 0, t + self.alpha, n=int(1e3))
            self.v.append((t, v_value))

            G_sqrt_value = np.sqrt(v_value)  
            return np.array(self.dL(y_interpolated(t)) / G_sqrt_value)

        def rhs(t, y):
            return self.f(t, y_interpolated(t)) - self.alpha_t(t) * integral(t)
                
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
    
