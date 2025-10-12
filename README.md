# Goal

This repository is associated with the article _"Modeling AdaGrad, RMSProp, and Adam with Integro-Differential Equations"_. The primary aim of this project is to study the dynamics of these optimization algorithms as nonlocal models.

## Repository Structure

The repository is organized into two main components:

### 1. **`solvers/` Package**

Contains modular Python implementations of both discrete and continuous-time optimization algorithms:

- **`adagradsolver.py`**: Implements `AdaGrad` (discrete optimizer) and `NonlocalSolverAdaGrad` (continuous nonlocal dynamics).
- **`adamsolver.py`**: Implements `AdamMomentum` (discrete optimizer) and `NonlocalSolverMomentumAdam` (continuous nonlocal dynamics).
- **`rmspropsolver.py`**: Implements `RMSPropMomentum` (discrete optimizer) and `NonlocalSolverMomentumRMSProp` (continuous nonlocal dynamics).
- **`base/common.py`**: Contains the shared base class `_NonlocalSolverBase` and common utilities including:
  - `fixed_quad_jax`: Fixed-order Gauss-Legendre quadrature for numerical integration
  - Time grid construction and explicit Euler time-stepping
  - Fixed-point relaxation with under-relaxation (smoothing)
  - Global error monitoring and convergence control

### 2. **`simulations/` Folder**

Contains Jupyter notebooks that reproduce all results from the article:

**Convex Cases:**
- `Numerical_Simulations_AdaGrad.ipynb`
- `Numerical_Simulations_Adam.ipynb`
- `Numerical_Simulations_RMSPROP.ipynb`

**Non-Convex Cases:**
- `Numerical_Simulations_AdaGrad_Nonconvex.ipynb`
- `Numerical_Simulations_Adam_Nonconvex.ipynb`
- `Numerical_Simulations_RMSPROP_Nonconvex.ipynb`

**Generated Figures:**
- `figures/`: Contains 84 PNG images generated from simulation results

## Solver Architecture

### Discrete Optimizers

Each algorithm (`AdaGrad`, `AdamMomentum`, `RMSPropMomentum`) provides a from-scratch implementation of the discrete optimization algorithm with:

- **`__init__`**: Initializes optimizer parameters (learning rate, momentum coefficients, epsilon, weight decay, L2 regularization, epochs).
- **`solve(theta_initial)`**: Runs the optimization algorithm for a specified number of epochs, returning the trajectory and accumulated statistics.

### Continuous Nonlocal Solvers

All continuous solvers (`NonlocalSolverAdaGrad`, `NonlocalSolverMomentumAdam`, `NonlocalSolverMomentumRMSProp`) inherit from `_NonlocalSolverBase` and implement:

- **`_build_stats(y)`**: Precomputes nonlocal statistics (integrals, moments) needed for the right-hand side evaluation.
- **`_rhs(t, y_prev, idx, *stats)`**: Computes the right-hand side of the integro-differential equation at time `t`.
- **`solve()`**: Runs the fixed-point iteration with under-relaxation to solve the nonlocal dynamics.

### Algorithm for Nonlocal Solvers (`solve` Method)

The `solve` method implements a fixed-point outer loop with under-relaxation:

```plaintext
Initialize iteration count to 0
Compute initial solution y_current using the original differential equation
Compute initial guess y_guess including the integral part
Calculate the initial global error between y_current and y_guess

WHILE global error is greater than tolerance:
    Compute new solution y_new using a smoothing factor between y_current and y_guess
    Update y_guess with the right-hand side of the ODE including the integral part
    Calculate the current global error between y_new and y_guess

    IF current error is greater than the last error:
        IF maximum smoothing factor reached:
            Exit loop without achieving desired tolerance
        ELSE:
            Update smoothing factor to the next value

    Update y_current to y_new
    Increment iteration count

    IF iteration count exceeds maximum limit:
        Exit loop

Set final solution y to y_guess
Return time values and corresponding solution y
```

## Libraries Used

- **`jax` / `jax.numpy`**: High-performance numerical computing with automatic differentiation and JIT compilation.
- **`interpax`**: JAX-compatible interpolation library for cubic interpolation.
- **`numpy`**: Fundamental package for numerical computations (used for Gauss-Legendre node/weight computation).
- **`plotly`**: Interactive visualization library for generating figures.
- **`sklearn`**: Used for parameter grid generation in simulations.

## Objective

The goal of this repository is to explore and model the dynamics of popular optimization algorithms (AdaGrad, RMSProp, and Adam) as nonlocal models. By leveraging integro-differential equations, we aim to provide deeper insights into how these algorithms operate and optimize in high-dimensional spaces.

## Installation and Dependencies

### Requirements

- Python 3.8+
- JAX (with CPU or GPU support)
- interpax
- numpy
- plotly
- scikit-learn
- jupyter

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd numerical_simulations_nonlocal
   ```

2. Install dependencies:
   ```bash
   pip install jax jaxlib interpax numpy plotly scikit-learn jupyter
   ```

   For GPU support, follow [JAX installation instructions](https://github.com/google/jax#installation).

## How to Use

### Running Simulations

1. **Navigate to the simulations folder:**
   ```bash
   cd simulations
   ```

2. **Open a Jupyter notebook:**
   ```bash
   jupyter notebook
   ```

3. **Select a simulation notebook:**
   - For convex problems: `Numerical_Simulations_[Algorithm].ipynb`
   - For non-convex problems: `Numerical_Simulations_[Algorithm]_Nonconvex.ipynb`
   
   where `[Algorithm]` is one of: `AdaGrad`, `Adam`, or `RMSPROP`.

4. **Run all cells** to reproduce the results and generate figures.


## Additional Files

- **`checks_pytorch/`**: Contains validation notebooks comparing JAX implementations with PyTorch.
- **`LICENSE`**: MIT License for the project.


## License

This repository is released under the [MIT License](LICENSE).
