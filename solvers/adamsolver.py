import numpy as np
from typing import Callable
from typing import Tuple
from jax import lax
import jax.numpy as jnp
from interpax import interp1d
from .base.common import _NonlocalSolverBase, DTYPE, fixed_quad_jax

try:
    import jax
    import jax.numpy as jnp
    device = jax.devices()[0]
    print(f"[Solver] Usando JAX en: {device.device_kind} ({device.platform})")
except ImportError:
    raise ImportError("[Solver] JAX no está instalado. Instálalo para usar este optimizador.")

class AdamMomentum:
    def __init__(self, dL: Callable, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8, 
                 weight_decay: float = 0, lambda_l2: float = 0, epochs: int = 1000):

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.lambda_l2 = lambda_l2
        if not hasattr(dL, 'lower'):
            self.dL = jax.jit(dL)
        else:
            self.dL = dL
        self.epochs = epochs
        self.global_error_tolerance = 1e-5
        self.reset_state()

    def reset_state(self):
        self.m = 0.0
        self.v = 0.0
        self.iteration = 1
        self.theta_result = []
        self.m_result = []
        self.v_result = []

    @staticmethod
    @jax.jit
    def __global_error__(theta_new, theta_old):
        import jax.numpy as jnp
        diff = theta_new - theta_old
        return jnp.abs(diff)

    def solve(self, theta_initial):
        self.reset_state()
        theta = jnp.array(theta_initial)
        while self.iteration <= self.epochs:
            self.theta_result.append(float(theta))
            self.m_result.append(float(self.m))
            self.v_result.append(float(self.v))

            theta_old = theta
            dL_value = self.dL(theta)
            if self.lambda_l2 != 0:
                dL_value += self.lambda_l2 / 2 * theta

            self.m = self.beta1 * self.m + (1 - self.beta1) * dL_value
            self.v = self.beta2 * self.v + (1 - self.beta2) * (dL_value ** 2)
            m_hat = self.m / (1 - self.beta1 ** self.iteration)
            v_hat = self.v / (1 - self.beta2 ** self.iteration)
            update = self.lr * m_hat / (jnp.sqrt(v_hat) + self.epsilon)

            if self.weight_decay != 0:
                theta = (1 - self.weight_decay) * theta - update
            else:
                theta -= update

            global_error = self.__global_error__(theta_new=theta, theta_old=theta_old)

            if self.iteration % 50 == 0:
                print(f'Epoch: {self.iteration}, Error: {float(global_error)}.')

            self.iteration += 1

        print(f'Last epoch: {self.iteration-1}, Error: {float(global_error)}.')

        return self.theta_result, self.m_result, self.v_result, self.iteration
    

    
class NonlocalSolverMomentumAdam(_NonlocalSolverBase):
    """
    θ̇(t) = -α(t) · m̄(t) / ( √v̄(t) + ε(t) )
    con   m̄, v̄  dados por los núcleos exponenciales K₁, K₂
          Kₐ(t) = (1-βₐ)/α · e^{-(1-βₐ) t / α},     a = 1,2.
    """
    def __init__(self, *args,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps_base: float = 1e-8,
                 **kw):
        super().__init__(*args, **kw)
        self.beta1, self.beta2 = map(DTYPE, betas)
        self.eps_base = DTYPE(eps_base)

        # Factores de corrección (5)-(6) de la proposición
        self._alpha_t = lambda t: jnp.where(
            t <= 1e-12,
            1.,
            jnp.sqrt(1. - self.beta2 ** (t / self.alpha)) / (1. - self.beta1 ** (t / self.alpha))
        )
        self._eps_t = lambda t: jnp.where(
            t <= 1e-12,
            self.eps_base,                                   
            self.eps_base * jnp.sqrt(1. - self.beta2 ** (t / self.alpha))
        )

        # --- pre-cálculo de matrices de pesos ---
        dt        = self.t[:, None] - self.t[None, :]    # t_i - t_j
        self._tri = dt >= 0                              # máscara causal

        self.lam1 = (1. - self.beta1) / self.alpha       # λ₁ = (1-β₁)/α
        self.lam2 = (1. - self.beta2) / self.alpha       # λ₂ = (1-β₂)/α

        # e^{-λ (t_i - t_j)}  (máscara causal se aplica en _build_stats)
        self._exp1 = jnp.exp(-self.lam1 * dt)
        self._exp2 = jnp.exp(-self.lam2 * dt)


    # ------------------------------------------------------------------
    def _build_stats(self, y):
        if self.verbose:
            jax.debug.print("¿NaNs en y?: {}", jnp.isnan(y).any())

        interp = self._interp(y)
        lam1 = (1. - self.beta1) / self.alpha
        lam2 = (1. - self.beta2) / self.alpha
        nGL  = int(1e3)

        def g_fun(tau):
            return self.dL(interp(tau)) + 0.5*self.lambda_ * interp(tau)
        
        @jax.jit
        def _moments_single(t,lam):
            def _compute(_t):
                ker = lambda tau: jnp.exp(-lam * (_t - tau))
                f_m = lambda tau: lam * ker(tau) * g_fun(tau)
                f_v = lambda tau: lam * ker(tau) * g_fun(tau)**2
                m_k = fixed_quad_jax(f_m, 1e-12, _t, nGL, verbose=self.verbose)
                v_k = fixed_quad_jax(f_v, 1e-12, _t, nGL, verbose=self.verbose)
                if self.verbose:
                    jax.debug.print("step values → nGL ={}  t={}  m={}  v={}", nGL, t, m_k, v_k)
                return m_k, v_k
            return lax.cond(t < 1e-12, lambda _: (DTYPE(0.), DTYPE(0.)), _compute, t)

        # vectorizado sobre la malla
        m = jax.vmap(lambda t: _moments_single(t,lam1)[0])(self.t)
        v = jax.vmap(lambda t: _moments_single(t,lam2)[1])(self.t)

        if self.verbose:                             
            # imprime 3 primeros y el último punto como muestra
            jax.debug.print("m[0]={:.3e}, v[0]={:.3e}", m[0], v[0])
            jax.debug.print("m[1]={:.3e}, v[1]={:.3e}", m[1], v[1])
            jax.debug.print("m[-1]={:.3e}, v[-1]={:.3e}", m[-1], v[-1])

        self._last_m = jnp.stack((self.t, m), axis=1)
        self._last_v = jnp.stack((self.t, v), axis=1)

        v_sqrt = jnp.sqrt(v)
        a_t    = jax.vmap(self._alpha_t)(self.t)
        eps_t  = jax.vmap(self._eps_t)(self.t)

        return (y, m, v_sqrt, a_t, eps_t)

    # ------------------------------------------------------------------
    def _rhs(self, t, y_prev, idx, y_fix, m, v_sqrt, a_t, eps_t):
        """
        θ̇_i(t) = - α(t) * m_i(t) / ( √v_i(t) + ε(t) )
        """
        y_val = interp1d(t, self.t, y_fix, method="cubic", extrap=True)
        denom = v_sqrt[idx] + eps_t[idx]
        return self.f(t, y_val) - a_t[idx] * (m[idx] / denom)