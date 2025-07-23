from scipy.integrate import fixed_quad
from scipy.interpolate import interp1d
from typing import Callable
import numpy as np
from numba import njit

try:
    import jax
    import jax.numpy as jnp
    from interpax import interp1d 

    device = jax.devices()[0]
    jax.config.update("jax_enable_x64", True)
    DTYPE = jnp.float64
    
    print(f"[NonlocalSolver] Usando JAX en: {device.device_kind} ({device.platform})")
except Exception:
    raise ImportError("[NonlocalSolver] JAX no está instalado. Instálalo para usar este optimizador.")


# ------------------------------------------------------------
#  NonlocalSolverAdam  
# ------------------------------------------------------------

class NonlocalSolverMomentumAdam:
    def __init__(self, f: Callable, dL: Callable, t_span: list, y0, betas: list,
                 alpha: float, lambda_:float = 0.0, verbose: bool = True):
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

        self.y0     = jnp.asarray(y0,    dtype=DTYPE)
        self.alpha  = jnp.asarray(alpha, dtype=DTYPE)
        self.lambda_= jnp.asarray(lambda_, dtype=DTYPE)
        
        self.t_span = t_span
        self.t      = jnp.arange(t_span[0], t_span[1], self.alpha, dtype=DTYPE)
        self.betas  = [DTYPE(b) for b in betas]

        self.f  = jax.jit(f)  if not hasattr(f,  "lower") else f
        self.dL = jax.jit(dL) if not hasattr(dL, "lower") else dL

        self.alpha_t = lambda t: jnp.where(t <= 1e-12, 1.0, jnp.sqrt(1 - betas[1] ** (t / alpha)) / (1 - betas[0] ** (t / alpha)))
        self.epsilon_t = lambda t: jnp.sqrt(1-betas[1]**(t/alpha)) * DTYPE(1e-8)  # Regularization term to avoid division by zero.

        # Smoothing parameters for the solution algorithm.
        self.smoothing_factor     = jnp.asarray(0.5,    dtype=DTYPE)
        self.smoothing_factor_max = jnp.asarray(0.9999, dtype=DTYPE)
        self.increments = jnp.linspace(
            self.smoothing_factor,
            self.smoothing_factor_max,
            num=int(1e3),
            dtype=DTYPE,
        )
        self.max_value_index = False

        # Control parameters for the solution.
        self.max_iteration = int(1e10)
        self.global_error_tolerance = 1e-4
        self.verbose = verbose


    def __initial_solution__(self) -> jnp.ndarray:
        """
        Computes the initial solution using the original differential equation.

        :return: Initial approximate solution for y.
        """
        return self.__solve_ode__(lambda t, y, idx=None: self.f(t, y))
    
    def __solve_ode__(self, rhs_ode: Callable) -> jnp.ndarray:
        """
        Solves the ordinary differential equation (ODE) for a given right-hand side.

        :param rhs_ode: Callable representing the right-hand side of the ODE.
        :return: Solution of the ODE.
        """

        t_values = self.t
        idxs   = jnp.arange(len(t_values), dtype=jnp.int32)
        y0_scalar = jnp.asarray(self.y0, dtype=DTYPE).ravel()[0]
        args = jnp.stack((t_values[:-1], idxs[:-1]), axis=1)

        def step(y_prev, args):
            t, idx = args
            idx = jnp.asarray(idx).astype(jnp.int32)
            y_next = y_prev + self.alpha * rhs_ode(t, y_prev, idx)
            return y_next, y_next

        _, y_hist = jax.lax.scan(step, y0_scalar, args)
        return jnp.concatenate([jnp.array([y0_scalar], dtype=DTYPE), y_hist])
    
    def __rhs_with_integral_part__(self, y: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the right-hand side of the ODE including the integral part.

        :param y: Array with y values for interpolation.
        :return: Solution with the integral term applied.
        """

        @jax.jit
        def y_interp(t):
            return interp1d(t, self.t, y, method="cubic")

        @jax.jit
        def ema(g_vals, beta):
            def step(prev, g):
                new = beta * prev + (DTYPE(1.0) - beta) * g
                return new, new
            _, out = jax.lax.scan(step, DTYPE(0.0), g_vals)
            return out
        
        m_vals = jax.vmap(lambda t: self.dL(y_interp(t)) + DTYPE(0.5) * self.lambda_ * y_interp(t))(self.t)
        v_vals = jax.vmap(lambda t: (self.dL(y_interp(t)) + DTYPE(0.5) * self.lambda_ * y_interp(t))**2)(self.t)

        m_conv = ema(m_vals, self.betas[0])  # beta1 para m
        v_conv = ema(v_vals, self.betas[1])  # beta2 para v
        v_sqrt = jnp.sqrt(v_conv)
        epsilons = jax.vmap(self.epsilon_t)(self.t)
        alphas = jax.vmap(self.alpha_t)(self.t)

        self.m_conv = m_conv
        self.v_sqrt = v_sqrt

        @jax.jit
        def rhs(t, y, idx):
            denom = v_sqrt[idx] + epsilons[idx]
            return self.f(t, y_interp(t)) - alphas[idx] * (m_conv[idx] / denom)
 
        return self.__solve_ode__(rhs)
        
    @staticmethod
    @jax.jit
    def __global_error__(y_new: jnp.ndarray, y_guess: jnp.ndarray) -> float:
        """
        Computes the global error between two approximations.

        :param y_new: New approximation.
        :param y_guess: Previous approximation.
        :return: Computed global error.
        """
        return jnp.linalg.norm(y_new - y_guess)
    
    @staticmethod
    @jax.jit
    def __next_y__(smoothing_factor: float, y_current: jnp.ndarray, y_guess: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the next value of y using a smoothing factor.

        :param smoothing_factor: Smoothing factor.
        :param y_current: Current value of y.
        :param y_guess: New estimate of y.
        :return: New value of y.
        """
        return (smoothing_factor * y_current) + ((DTYPE(1.0) - smoothing_factor) * y_guess)            
        
    def solve(self):
        """
        Solves the nonlocal differential equation with momentum term and manages convergence.

        :return: Tuple with time values and the corresponding solutions.
        """
        self.iteration = 0
        self.m_conv_history, self.v_sqrt_history = [], []

        y_current = self.__initial_solution__()
        y_guess = self.__rhs_with_integral_part__(y_current)

        self.m_conv_history.append(list(zip(self.t, self.m_conv)))
        self.v_sqrt_history.append(list(zip(self.t, self.v_sqrt)))

        current_error = self.__global_error__(y_current, y_guess)

        if self.verbose:
            print(f"Iteration {self.iteration} advanced. Current error: {current_error}.")

        if jnp.isnan(current_error) or jnp.isinf(current_error):
            print("Error diverged (NaN or Inf) in initial guess. Stopping.")
            self.y = y_guess
            self.global_error = current_error
            print(f'Last iteration: {self.iteration}. Final error: {current_error}')
            return self.t, self.y

        last_error = current_error
        while current_error > self.global_error_tolerance:
            
            y_new = self.__next_y__(self.smoothing_factor, y_current, y_guess)
            y_guess = self.__rhs_with_integral_part__(y_new)

            self.m_conv_history.append(list(zip(self.t, self.m_conv)))
            self.v_sqrt_history.append(list(zip(self.t, self.v_sqrt)))
            
            current_error = self.__global_error__(y_new, y_guess)

            y_current = y_new
            self.iteration += 1

            if jnp.isnan(current_error) or jnp.isinf(current_error):
                print("Error diverged (NaN or Inf). Stopping.")
                break

            if current_error > last_error:
                if self.max_value_index:
                    print(f'Maximum value of the smoothing factor reached. The algorithm will stop without reaching the desired tolerance. The error is {current_error}.')
                    break

                try:
                    next_factor = self.increments[jnp.searchsorted(self.increments, self.smoothing_factor, side='right')]
                except IndexError:
                    next_factor = self.smoothing_factor_max
                    print(f'Smoothing factor is at maximum value.')
                    self.max_value_index = True

                self.smoothing_factor = min(self.smoothing_factor_max, next_factor)
            last_error = current_error

            if self.verbose and self.iteration % 20 == 0:
                print(f"Iteration {self.iteration} advanced. Current error: {current_error}.")

            if self.iteration >= self.max_iteration:
                print(f"Maximum number of iterations reached. Current error: {current_error}.") 
                break
            
        print(f'Last iteration: {self.iteration}. Final error: {current_error}')

        self.y = y_guess
        self.global_error = current_error

        return self.t, self.y
    
# ------------------------------------------------------------
#  NonlocalSolverRMSProp
# ------------------------------------------------------------   

class NonlocalSolverMomentumRMSProp:
    """
    Versión JAX del solver no-local con término tipo RMSProp.
    Su interfaz y flujo interno replican a NonlocalSolverMomentumAdam
    para que ambas clases sean intercambiables.
    """
    # -------------  inicialización  ----------------------------------
    def __init__(self, f: Callable, dL: Callable, t_span: list, y0,
                 beta: float, alpha: float, lambda_: float = 0.0,
                 verbose: bool = True):

        # hiper-parámetros ------------------------------------------------
        self.y0      = jnp.asarray(y0,    dtype=DTYPE)
        self.alpha   = jnp.asarray(alpha, dtype=DTYPE)
        self.lambda_ = jnp.asarray(lambda_, dtype=DTYPE)
        self.beta    = DTYPE(beta)              # un solo parámetro

        # malla temporal --------------------------------------------------
        self.t_span = t_span
        self.t      = jnp.arange(t_span[0], t_span[1], self.alpha, dtype=DTYPE)

        # funciones dinamica / gradiente (JIT si no vienen ya compiladas) --
        self.f  = jax.jit(f)  if not hasattr(f,  "lower") else f
        self.dL = jax.jit(dL) if not hasattr(dL, "lower") else dL

        #   ε constante (RMSProp clásico)
        self.epsilon = DTYPE(1e-8)

        # ----------  parámetros de suavizado para la relajación ----------
        self.smoothing_factor     = jnp.asarray(0.5,    dtype=DTYPE)
        self.smoothing_factor_max = jnp.asarray(0.9999, dtype=DTYPE)
        self.increments = jnp.linspace(
            self.smoothing_factor, self.smoothing_factor_max,
            num=int(1e3), dtype=DTYPE)
        self.max_value_index = False

        # control de iteraciones / tolerancias ----------------------------
        self.max_iteration           = int(1e10)
        self.global_error_tolerance  = 1e-4
        self.verbose = verbose

    # ====================================================================
    #   utilidades privadas
    # ====================================================================

    def __initial_solution__(self) -> jnp.ndarray:
        """Primera integración (sin parte integral)."""
        return self.__solve_ode__(lambda t, y, idx=None: self.f(t, y))

    # --------------------------------------------------------------------
    def __solve_ode__(self, rhs_ode: Callable) -> jnp.ndarray:
        """Euler explícito + lax.scan (idéntico a la versión Adam)."""
        t_values = self.t
        idxs     = jnp.arange(len(t_values), dtype=jnp.int32)
        y0_scalar = self.y0.ravel()[0]                     # escalar
        args = jnp.stack((t_values[:-1], idxs[:-1]), axis=1)

        def step(y_prev, args):
            t, idx = args
            idx = idx.astype(jnp.int32)
            y_next = y_prev + self.alpha * rhs_ode(t, y_prev, idx)
            return y_next, y_next

        _, y_hist = jax.lax.scan(step, y0_scalar, args)
        return jnp.concatenate([jnp.array([y0_scalar], dtype=DTYPE), y_hist])

    # --------------------------------------------------------------------
    def __rhs_with_integral_part__(self, y: jnp.ndarray) -> jnp.ndarray:
        """
        Calcula RHS(t) = f(t, y(t)) - g(t) / √(v(t)+ε)    con
        v(t)   = EMA_β( g(t)^2 ),    g(t) = dL(y) + λ/2 * y
        """

        # --- interpolador C^2 (cúbico) sobre la malla autosuficiente -----
        @jax.jit
        def y_interp(t):
            return interp1d(t, self.t, y, method="cubic")

        # --- operador EMA (una sola beta) --------------------------------
        @jax.jit
        def ema(g_vals, beta):
            def step(prev, g):
                new = beta * prev + (DTYPE(1.0) - beta) * g
                return new, new
            _, out = jax.lax.scan(step, DTYPE(0.0), g_vals)
            return out

        # --- g(t) ---------------------------------------------------------
        g_vals = jax.vmap(
            lambda t: self.dL(y_interp(t)) + DTYPE(0.5)*self.lambda_*y_interp(t)
        )(self.t)

        # --- RMSProp: v(t) = EMA_β[g²]  -----------------------------------
        v_vals  = g_vals**2
        v_conv  = ema(v_vals, self.beta)
        v_sqrt  = jnp.sqrt(v_conv)

        # guardar para inspección / trazas
        self.v_sqrt = v_sqrt

        # --- RHS final ----------------------------------------------------
        @jax.jit
        def rhs(t, y_prev, idx):
            denom = v_sqrt[idx] + self.epsilon
            return self.f(t, y_interp(t)) - (g_vals[idx] / denom)

        return self.__solve_ode__(rhs)

    # -----------------------   estimadores globales  ---------------------
    @staticmethod
    @jax.jit
    def __global_error__(y_new: jnp.ndarray, y_guess: jnp.ndarray) -> float:
        return jnp.linalg.norm(y_new - y_guess)

    @staticmethod
    @jax.jit
    def __next_y__(smoothing_factor: float,
                   y_current: jnp.ndarray,
                   y_guess:   jnp.ndarray) -> jnp.ndarray:
        return (smoothing_factor * y_current) + \
               ((DTYPE(1.0) - smoothing_factor) * y_guess)

    # ====================================================================
    #   Bucle de resolución externo
    # ====================================================================
    def solve(self):
        """
        Ejecuta la relajación fija hasta que ||y_{k+1}-y_k|| < tol.
        Devuelve (t, y_final).
        """
        self.iteration      = 0
        self.v_sqrt_history = []

        # ----------  iteración 0  ----------------------------------------
        y_current = self.__initial_solution__()
        y_guess   = self.__rhs_with_integral_part__(y_current)
        self.v_sqrt_history.append(list(zip(self.t, self.v_sqrt)))

        current_error = self.__global_error__(y_current, y_guess)
        if self.verbose:
            print(f"Iteration {self.iteration} advanced. "
                  f"Current error: {current_error}.")

        # ----------  control de NaNs / Infs ------------------------------
        if jnp.isnan(current_error) or jnp.isinf(current_error):
            print("Error diverged (NaN or Inf) in initial guess. Stopping.")
            self.y            = y_guess
            self.global_error = current_error
            print(f"Last iteration: {self.iteration}. "
                  f"Final error: {current_error}")
            return self.t, self.y

        # ----------  bucle principal  ------------------------------------
        last_error = current_error
        while current_error > self.global_error_tolerance:

            # relajación
            y_new   = self.__next_y__(self.smoothing_factor,
                                      y_current, y_guess)
            y_guess = self.__rhs_with_integral_part__(y_new)
            self.v_sqrt_history.append(list(zip(self.t, self.v_sqrt)))

            current_error = self.__global_error__(y_new, y_guess)
            y_current     = y_new
            self.iteration += 1

            # --- diverge? -------------------------------------------------
            if jnp.isnan(current_error) or jnp.isinf(current_error):
                print("Error diverged (NaN or Inf). Stopping.")
                break

            # --- empeora -> aumentar factor de suavizado -----------------
            if current_error > last_error:
                if self.max_value_index:
                    print("Maximum smoothing factor reached. "
                          "Stopping without reaching tolerance. "
                          f"Error: {current_error}.")
                    break

                try:
                    next_factor = self.increments[
                        jnp.searchsorted(self.increments,
                                         self.smoothing_factor,
                                         side='right')]
                except IndexError:
                    next_factor = self.smoothing_factor_max
                    print('Smoothing factor is at maximum value.')
                    self.max_value_index = True

                self.smoothing_factor = \
                    jnp.minimum(self.smoothing_factor_max, next_factor)
            last_error = current_error

            # --- trazas de depuración cada 20 iteraciones ----------------
            if self.verbose and self.iteration % 20 == 0:
                print(f"Iteration {self.iteration} advanced. "
                      f"Current error: {current_error}.")

            # --- límite de iteraciones -----------------------------------
            if self.iteration >= self.max_iteration:
                print(f"Maximum number of iterations reached. "
                      f"Current error: {current_error}.")
                break

        # -----------------------------------------------------------------
        print(f"Last iteration: {self.iteration}. "
              f"Final error: {current_error}")

        self.y            = y_guess
        self.global_error = current_error
        return self.t, self.y


# ------------------------------------------------------------
#  NonlocalSolverAdaGrad   
# ------------------------------------------------------------

class NonlocalSolverAdaGrad:
    """
    Solver no-local con término de optimización AdaGrad
    (acumulador de gradientes al cuadrado sin decaimiento).

    La interfaz es la misma que la de NonlocalSolverMomentumAdam y
    NonlocalSolverMomentumRMSProp para que se puedan intercambiar.
    """

    # --------------------------------------------------------
    def __init__(self, f: Callable, dL: Callable, t_span: list, y0,
                 lr_decay: float = 0.0,        # factor de decaimiento del LR
                 alpha: float = 0.01,
                 lambda_: float = 0.0,
                 verbose: bool = True):

        # --- hiperpárametros ------------------------------------------------
        self.y0      = jnp.asarray(y0,    dtype=DTYPE)
        self.alpha   = jnp.asarray(alpha, dtype=DTYPE)
        self.lambda_ = jnp.asarray(lambda_, dtype=DTYPE)
        self.lr_decay= jnp.asarray(lr_decay, dtype=DTYPE)

        # --- malla temporal -------------------------------------------------
        self.t_span = t_span
        self.t      = jnp.arange(t_span[0], t_span[1], self.alpha, dtype=DTYPE)

        # --- funciones dinámicas / gradiente -------------------------------
        self.f  = jax.jit(f)  if not hasattr(f,  "lower") else f
        self.dL = jax.jit(dL) if not hasattr(dL, "lower") else dL

        # --- ε para evitar división por cero --------------------------------
        self.epsilon = DTYPE(1e-8)

        # --- LR adaptativo de AdaGrad  (decay opcional) ---------------------
        #     α_t(t) = 1 / (1 + t * lr_decay / α)
        self.alpha_t = lambda t: DTYPE(1.0) / (DTYPE(1.0) +
                                               t * (self.lr_decay / self.alpha))

        # --- parámetros de relajación --------------------------------------
        self.smoothing_factor     = jnp.asarray(0.5,    dtype=DTYPE)
        self.smoothing_factor_max = jnp.asarray(0.9999, dtype=DTYPE)
        self.increments = jnp.linspace(
            self.smoothing_factor, self.smoothing_factor_max,
            num=int(1e3), dtype=DTYPE)
        self.max_value_index = False

        # --- control de iteraciones / tolerancia ---------------------------
        self.max_iteration          = int(1e10)
        self.global_error_tolerance = 1e-4
        self.verbose = verbose

    # ======================================================================
    #   utilidades privadas
    # ======================================================================

    # --------------------------------------------------------
    def __initial_solution__(self) -> jnp.ndarray:
        """Integración inicial sin término integral."""
        return self.__solve_ode__(lambda t, y, idx=None: self.f(t, y))

    # --------------------------------------------------------
    def __solve_ode__(self, rhs_ode: Callable) -> jnp.ndarray:
        """Euler explícito vectorizado con lax.scan."""
        t_values = self.t
        idxs     = jnp.arange(len(t_values), dtype=jnp.int32)
        y0_scalar = self.y0.ravel()[0]

        args = jnp.stack((t_values[:-1], idxs[:-1]), axis=1)

        def step(y_prev, args):
            t, idx = args
            idx = idx.astype(jnp.int32)
            y_next = y_prev + self.alpha * rhs_ode(t, y_prev, idx)
            return y_next, y_next

        _, y_hist = jax.lax.scan(step, y0_scalar, args)
        return jnp.concatenate([jnp.array([y0_scalar], dtype=DTYPE), y_hist])

    # --------------------------------------------------------
    def __rhs_with_integral_part__(self, y: jnp.ndarray) -> jnp.ndarray:
        """
        RHS(t) = f(t, y(t)) - α_t(t) · g(t) / √(G(t)+ε)

        donde:
            g(t) = dL(y(t)) + (λ/2)·y(t)
            G(t) = Σ_{τ ≤ t} g(τ)²          (acumulador AdaGrad)
        """

        # --- interpolador cúbico sobre la malla --------------------------------
        @jax.jit
        def y_interp(t):
            return interp1d(t, self.t, y, method="cubic")

        # --- g(t) (gradiente + regularización) ---------------------------------
        g_vals = jax.vmap(
            lambda t: self.dL(y_interp(t)) + DTYPE(0.5)*self.lambda_*y_interp(t)
        )(self.t)

        # --- acumulador AdaGrad: G(t) = Σ g² ------------------------------------
        G_vals   = jnp.cumsum(g_vals**2)
        G_sqrt   = jnp.sqrt(G_vals)                    # √G(t)
        alphas   = jax.vmap(self.alpha_t)(self.t)      # α_t(t)

        # Historia para inspección / trazas
        self.G_sqrt = G_sqrt

        # --- RHS final ----------------------------------------------------------
        @jax.jit
        def rhs(t, y_prev, idx):
            denom = G_sqrt[idx] + self.epsilon
            return self.f(t, y_interp(t)) - alphas[idx] * (g_vals[idx] / denom)

        return self.__solve_ode__(rhs)

    # -----------------------  estimadores globales ------------------------
    @staticmethod
    @jax.jit
    def __global_error__(y_new: jnp.ndarray, y_guess: jnp.ndarray) -> float:
        return jnp.linalg.norm(y_new - y_guess)

    @staticmethod
    @jax.jit
    def __next_y__(smoothing_factor: float,
                   y_current: jnp.ndarray,
                   y_guess:   jnp.ndarray) -> jnp.ndarray:
        return (smoothing_factor * y_current) + \
               ((DTYPE(1.0) - smoothing_factor) * y_guess)

    # ======================================================================
    #   Bucle externo de resolución
    # ======================================================================
    def solve(self):
        """
        Ejecuta las iteraciones de relajación fija hasta que la norma
        de la diferencia global sea menor que el umbral especificado.
        Devuelve la tupla (t, y_final).
        """
        self.iteration     = 0
        self.G_sqrt_history = []

        # ---------------  iteración 0  ------------------------------------
        y_current = self.__initial_solution__()
        y_guess   = self.__rhs_with_integral_part__(y_current)
        self.G_sqrt_history.append(list(zip(self.t, self.G_sqrt)))

        current_error = self.__global_error__(y_current, y_guess)
        if self.verbose:
            print(f"Iteration {self.iteration} advanced. "
                  f"Current error: {current_error}.")

        # --- controlar divergencias --------------------------------------
        if jnp.isnan(current_error) or jnp.isinf(current_error):
            print("Error diverged (NaN or Inf) in initial guess. Stopping.")
            self.y            = y_guess
            self.global_error = current_error
            print(f"Last iteration: {self.iteration}. "
                  f"Final error: {current_error}")
            return self.t, self.y

        # ---------------  bucle principal  --------------------------------
        last_error = current_error
        while current_error > self.global_error_tolerance:

            # relajación
            y_new   = self.__next_y__(self.smoothing_factor,
                                      y_current, y_guess)
            y_guess = self.__rhs_with_integral_part__(y_new)
            self.G_sqrt_history.append(list(zip(self.t, self.G_sqrt)))

            current_error = self.__global_error__(y_new, y_guess)
            y_current     = y_new
            self.iteration += 1

            # --- NaN / Inf -------------------------------------------------
            if jnp.isnan(current_error) or jnp.isinf(current_error):
                print("Error diverged (NaN or Inf). Stopping.")
                break

            # --- si el error empeora -> aumenta factor de suavizado -------
            if current_error > last_error:
                if self.max_value_index:
                    print("Maximum smoothing factor reached. "
                          "Stopping without reaching tolerance. "
                          f"Error: {current_error}.")
                    break

                try:
                    next_factor = self.increments[
                        jnp.searchsorted(self.increments,
                                         self.smoothing_factor,
                                         side='right')]
                except IndexError:
                    next_factor = self.smoothing_factor_max
                    print('Smoothing factor is at maximum value.')
                    self.max_value_index = True

                self.smoothing_factor = \
                    jnp.minimum(self.smoothing_factor_max, next_factor)
            last_error = current_error

            # --- trazas de depuración -------------------------------------
            if self.verbose and self.iteration % 20 == 0:
                print(f"Iteration {self.iteration} advanced. "
                      f"Current error: {current_error}.")

            # --- límite de iteraciones ------------------------------------
            if self.iteration >= self.max_iteration:
                print(f"Maximum number of iterations reached. "
                      f"Current error: {current_error}.")
                break

        # -----------------------------------------------------------------
        print(f"Last iteration: {self.iteration}. "
              f"Final error: {current_error}")

        self.y            = y_guess
        self.global_error = current_error
        return self.t, self.y

    
