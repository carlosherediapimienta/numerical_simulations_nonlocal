from scipy.integrate import fixed_quad
from scipy.interpolate import interp1d
from concurrent.futures import ThreadPoolExecutor
from numba import njit
from typing import Callable
import numpy as np

class NonlocalSolverMomentumAdam:
    def __init__(self, f: Callable, dL: Callable, t_span: list, y0: np.array, betas: list,
                 alpha: float, lambda_:float = 0, verbose: bool = True):
        """
        Initializes the solver with the given parameters.

        :param f: Callable representing the system dynamics function.
        :param dL: Callable representing the gradient of the loss function.
        :param t_span: List with the time span [t0, tf].
        :param y0: Initial condition for y.
        :param betas: List of damping parameters.
        :param alpha: Time step for integration.
        :param lambda_: Lambda parameter for the regularization term (optional).
        :param verbose: Boolean controlling the verbosity of messages (optional).
        """
        
        np.random.seed(33)

        self.f = f
        self.t_span = t_span
        self.y0 = y0 
        self.alpha = alpha
        self.t = np.arange(t_span[0], t_span[1], alpha)
        self.betas = betas
        self.lambda_ = lambda_
        self.k = lambda i, s: (1-betas[i-1]) / alpha * np.exp( - ((1-betas[i-1]) / alpha) * s)  # Kernel function for integrand computation.
        self.dL = dL
        self.alpha_t = lambda t: np.sqrt(1-betas[1]**(t/alpha)) / (1-betas[0]**(t/alpha))  # Time-scaling factor.
        self.epsilon_t = lambda t: np.sqrt(1-betas[1]**(t/alpha)) * 1e-8  # Regularization term to avoid division by zero.

        # Smoothing parameters for the solution algorithm.
        self.smoothing_factor = 0.5
        self.smoothing_factor_max = 0.9999
        self.increments = np.linspace(self.smoothing_factor, self.smoothing_factor_max, num=int(1e3))
        self.max_value_index = False

        # Control parameters for the solution.
        self.max_iteration = int(1e10)
        self.global_error_tolerance = 1e-4
        self.verbose = verbose


    def __initial_solution__(self) -> np.array:
        """
        Computes the initial solution using the original differential equation.

        :return: Initial approximate solution for y.
        """
        return self.__solve_ode__(self.f)
    
    def __solve_ode__(self, rhs_ode: Callable) -> np.array:
        """
        Solves the ordinary differential equation (ODE) for a given right-hand side.

        :param rhs_ode: Callable representing the right-hand side of the ODE.
        :return: Solution of the ODE.
        """
        t_values = self.t
        y_values = np.zeros(t_values.shape)
        y_values[0] = self.y0.item() if isinstance(self.y0, np.ndarray) else self.y0
        for i in range(1, len(t_values)):
            y_values[i] = y_values[i - 1] + self.alpha * rhs_ode(t_values[i - 1], y_values[i - 1])
        return y_values
    
    def __rhs_with_integral_part__(self, y: np.array) -> np.array:
        """
        Computes the right-hand side of the ODE including the integral part.

        :param y: Array with y values for interpolation.
        :return: Solution with the integral term applied.
        """
        y_interpolated = interp1d(self.t, y, kind='cubic', fill_value="extrapolate", assume_sorted=True)

        self.m = []  # Stores m(t) values.
        self.v = []  # Stores v(t) values.
        self.m.append((0, 0))
        self.v.append((0, 0))

        def integral(t):
            """
            Computes the integral term for a given time t.

            :param t: Current time.
            :return: Value of the integral at t.
            """
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
            """
            Computes the adjusted right-hand side of the ODE.

            :param t: Current time.
            :param y: Current value of y.
            :return: Value of the derivative at t.
            """
            return self.f(t, y_interpolated(t)) - self.alpha_t(t) * integral(t)
                
        return self.__solve_ode__(rhs)
    
    @staticmethod
    @njit(parallel=True)
    def __global_error__(y_new: np.array, y_guess: np.array) -> float:
        """
        Computes the global error between two approximations.

        :param y_new: New approximation.
        :param y_guess: Previous approximation.
        :return: Computed global error.
        """
        diff = y_new - y_guess
        return np.sqrt(np.sum(diff ** 2))
    
    @staticmethod
    @njit(parallel=True)
    def __next_y__(smoothing_factor: float, y_current: np.array, y_guess: np.array) -> np.array:
        """
        Computes the next value of y using a smoothing factor.

        :param smoothing_factor: Smoothing factor.
        :param y_current: Current value of y.
        :param y_guess: New estimate of y.
        :return: New value of y.
        """
        return (smoothing_factor * y_current) + ((1.0 - smoothing_factor) * y_guess)            
        
    def solve(self):
        """
        Solves the nonlocal differential equation with momentum term and manages convergence.

        :return: Tuple with time values and the corresponding solutions.
        """
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
        """
        Initializes the NonlocalSolverMomentumRMSProp class.

        Parameters:
        - f (Callable): Function that defines the right-hand side of the ODE.
        - dL (Callable): Derivative of the loss function (gradient).
        - t_span (list): Time span [t0, tf].
        - y0 (np.array): Initial value of the solution.
        - beta (float): Momentum parameter for RMSProp.
        - alpha (float): Step size in the time integration.
        - lambda_ (float): Regularization parameter.
        - verbose (bool): Indicates whether to print messages during execution.
        """
        
        # Set the random seed for reproducibility
        np.random.seed(33)

        # Assign the input parameters to the instance variables
        self.f = f
        self.t_span = t_span
        self.y0 = y0 
        self.alpha = alpha
        self.t = np.arange(t_span[0], t_span[1], alpha)
        self.beta = beta
        self.lambda_ = lambda_
        self.k = lambda s: (1 - self.beta) / self.alpha * np.exp(- ((1 - beta) / alpha) * s)  # Kernel function
        self.dL = dL
        self.epsilon = 1e-8  # Small value to prevent division by zero

        # Initialize smoothing factors for the RMSProp method
        self.smoothing_factor = 0.5
        self.smoothing_factor_max = 0.9999
        self.increments = np.linspace(self.smoothing_factor, self.smoothing_factor_max, num=int(1e3))
        self.max_value_index = False

        # Set maximum iterations and error tolerance
        self.max_iteration = int(1e10)
        self.global_error_tolerance = 1e-4
        self.verbose = verbose


    def __initial_solution__(self) -> np.array:
        """
        Computes the initial solution by solving the ODE using the provided function f.

        Returns:
        - np.array: The initial solution array.
        """
        return self.__solve_ode__(self.f)
    
    def __solve_ode__(self, rhs_ode: Callable) -> np.array:
        """
        Solves the ordinary differential equation (ODE) using Euler's method.

        Parameters:
        - rhs_ode (Callable): The right-hand side of the ODE.

        Returns:
        - np.array: The solution of the ODE at each time step.
        """
        t_values = self.t
        y_values = np.zeros(t_values.shape)
        y_values[0] = self.y0.item() if isinstance(self.y0, np.ndarray) else self.y0
        for i in range(1, len(t_values)):
            y_values[i] = y_values[i - 1] + self.alpha * rhs_ode(t_values[i - 1], y_values[i - 1])
        return y_values
    
    def __rhs_with_integral_part__(self, y: np.array) -> np.array:
        """
        Computes the right-hand side of the ODE including the integral part.

        Parameters:
        - y (np.array): The current solution array.

        Returns:
        - np.array: The right-hand side of the ODE adjusted with the integral term.
        """
        # Interpolate the solution for smoother integration
        y_interpolated = interp1d(self.t, y, kind='cubic', fill_value="extrapolate", assume_sorted=True)

        self.v = []
        self.v.append((0, 0))

        def integral(t):
            """
            Computes the integral part of the ODE.

            Parameters:
            - t: Current time.

            Returns:
            - np.array: The integral term evaluated at time t.
            """
            def integrand(tp):
                k_value = self.k(tp)
                df_value = self.dL(y_interpolated(t - tp))
                common_term = df_value + 0.5 * self.lambda_ * y_interpolated(t - tp)
                return k_value * (common_term ** 2)
            
            v_value, _ = fixed_quad(lambda tp: integrand(tp), 0, t + self.alpha, n=int(1e3))
            self.v.append((t, v_value))

            G_sqrt_value = np.sqrt(v_value)  
            return np.array(self.dL(y_interpolated(t)) / (G_sqrt_value + self.epsilon))

        def rhs(t, y):
            """
            Computes the right-hand side of the ODE at a given time.

            Parameters:
            - t: Current time.
            - y: Current value of the solution.

            Returns:
            - np.array: The right-hand side of the ODE.
            """
            return self.f(t, y_interpolated(t)) - integral(t)
                
        return self.__solve_ode__(rhs)
    
    @staticmethod
    @njit(parallel=True)
    def __global_error__(y_new: np.array, y_guess: np.array) -> float:
        """
        Computes the global error between the new solution and the guessed solution.

        Parameters:
        - y_new (np.array): The new solution array.
        - y_guess (np.array): The guessed solution array.

        Returns:
        - float: The global error.
        """
        diff = y_new - y_guess
        return np.sqrt(np.sum(diff ** 2))
    
    @staticmethod
    @njit(parallel=True)
    def __next_y__(smoothing_factor: float, y_current: np.array, y_guess: np.array) -> np.array:
        """
        Updates the solution based on the smoothing factor.

        Parameters:
        - smoothing_factor (float): The current smoothing factor.
        - y_current (np.array): The current solution array.
        - y_guess (np.array): The guessed solution array.

        Returns:
        - np.array: The updated solution array.
        """
        return (smoothing_factor * y_current) + ((1.0 - smoothing_factor) * y_guess)            
        
    def solve(self):
        """
        Solves the ODE using the nonlocal Momentum RMSProp method.

        Returns:
        - tuple: Time array and the final solution array.
        """
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
        """
        Initializes the NonlocalSolverAdaGrad class.

        Parameters:
        - f (Callable): Function that defines the right-hand side of the ODE.
        - dL (Callable): Derivative of the loss function (gradient).
        - t_span (list): Time span [t0, tf].
        - y0 (np.array): Initial value of the solution.
        - lr_decay (float): Learning rate decay factor.
        - alpha (float): Step size in the time integration.
        - lambda_ (float): Regularization parameter.
        - verbose (bool): Indicates whether to print messages during execution.
        """
        
        # Set the random seed for reproducibility.
        np.random.seed(33)

        # Assign the input parameters to the instance variables.
        self.f = f
        self.t_span = t_span
        self.y0 = y0 
        self.alpha = alpha
        self.t = np.arange(t_span[0], t_span[1], alpha)
        self.lambda_ = lambda_
        self.k = lambda s: 1 / alpha  # Kernel function.
        self.dL = dL
        self.alpha_t = lambda s: 1 / (1 + s * (lr_decay / alpha))  # Learning rate adjustment.

        # Initialize smoothing factors for the AdaGrad method.
        self.smoothing_factor = 0.5
        self.smoothing_factor_max = 0.9999
        self.increments = np.linspace(self.smoothing_factor, self.smoothing_factor_max, num=int(1e3))
        self.max_value_index = False

        # Set maximum iterations and error tolerance.
        self.max_iteration = int(1e10)
        self.global_error_tolerance = 1e-4
        self.verbose = verbose


    def __initial_solution__(self) -> np.array:
        """
        Computes the initial solution by solving the ODE using the provided function f.

        Returns:
        - np.array: The initial solution array.
        """
        return self.__solve_ode__(self.f)
    
    def __solve_ode__(self, rhs_ode: Callable) -> np.array:
        """
        Solves the ordinary differential equation (ODE) using Euler's method.

        Parameters:
        - rhs_ode (Callable): The right-hand side of the ODE.

        Returns:
        - np.array: The solution of the ODE at each time step.
        """
        t_values = self.t
        y_values = np.zeros(t_values.shape)
        y_values[0] = self.y0.item() if isinstance(self.y0, np.ndarray) else self.y0
        for i in range(1, len(t_values)):
            y_values[i] = y_values[i - 1] + self.alpha * rhs_ode(t_values[i - 1], y_values[i - 1])
        return y_values
    
    def __rhs_with_integral_part__(self, y: np.array) -> np.array:
        """
        Computes the right-hand side of the ODE including the integral part.

        Parameters:
        - y (np.array): The current solution array.

        Returns:
        - np.array: The right-hand side of the ODE adjusted with the integral term.
        """
        # Interpolate the solution for smoother integration.
        y_interpolated = interp1d(self.t, y, kind='cubic', fill_value="extrapolate", assume_sorted=True)

        self.v = []
        self.v.append((0, 0))

        def integral(t):
            """
            Computes the integral part of the ODE.

            Parameters:
            - t: Current time.

            Returns:
            - np.array: The integral term evaluated at time t.
            """
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
            """
            Computes the right-hand side of the ODE at a given time.

            Parameters:
            - t: Current time.
            - y: Current value of the solution.

            Returns:
            - np.array: The right-hand side of the ODE.
            """
            return self.f(t, y_interpolated(t)) - self.alpha_t(t) * integral(t)
                
        return self.__solve_ode__(rhs)
    
    @staticmethod
    @njit(parallel=True)
    def __global_error__(y_new: np.array, y_guess: np.array) -> float:
        """
        Computes the global error between the new solution and the guessed solution.

        Parameters:
        - y_new (np.array): The new solution array.
        - y_guess (np.array): The guessed solution array.

        Returns:
        - float: The global error.
        """
        diff = y_new - y_guess
        return np.sqrt(np.sum(diff ** 2))
    
    @staticmethod
    @njit(parallel=True)
    def __next_y__(smoothing_factor: float, y_current: np.array, y_guess: np.array) -> np.array:
        """
        Updates the solution based on the smoothing factor.

        Parameters:
        - smoothing_factor (float): The current smoothing factor.
        - y_current (np.array): The current solution array.
        - y_guess (np.array): The guessed solution array.

        Returns:
        - np.array: The updated solution array.
        """
        return (smoothing_factor * y_current) + ((1.0 - smoothing_factor) * y_guess)            
        
    def solve(self):
        """
        Solves the ODE using the nonlocal AdaGrad method.

        Returns:
        - tuple: Time array and the final solution array.
        """
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
    
