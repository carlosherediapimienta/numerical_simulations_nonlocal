# Collaboration with Dr. Hidenori Tanaka

This repository is a collaboration with Dr. Hidenori Tanaka from Harvard University and is associated with the article _"Modeling AdaGrad, RMSProp, and Adam with Integro-Differential Equations"_. The main focus of this project is to study the dynamics of these optimization algorithms as nonlocal models.

## Repository Structure

The repository is primarily organized around the `Solver` folder, which contains two Python files:

1. **`NonLocalSolver.py`**: This file contains the implementations for solving integro-differential equations using IDESolver Method.
2. **`Solver.py`**: This file includes the algorithms for AdaGrad, Adam, and RMSProp implemented from scratch.

## Classes Overview

All classes (`NonlocalSolverMomentumAdam`, `NonlocalSolverMomentumRMSProp`, `NonlocalSolverAdaGrad`) share a similar structure to handle the dynamics of optimization algorithms as nonlocal models:

- **`__init__`**: Initializes solver parameters (dynamics function, gradient, time span, initial conditions, specific optimization parameters, etc.).
- **`__solve_ode__`**: Solves the ordinary differential equation using a numerical method.
- **`__rhs_with_integral_part__`**: Computes the right-hand side of the ODE, including integral components.
- **`solve`**: Runs the solution algorithm, managing convergence and error control.

### Pseudocode for `solve` Function

The `solve` function implements an iterative approach to solve the integro-differential equation while ensuring convergence and managing errors.

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

- `scipy.integrate.fixed_quad`: Utilized for numerical integration with fixed quadrature.
- `scipy.interpolate.interp1d`: Provides interpolation for constructing smooth functions.
- `concurrent.futures.ThreadPoolExecutor`: Manages parallel computations to speed up integral calculations.
- `numba.njit`: JIT compiler for optimizing performance-critical functions.
- `numpy`: Fundamental package for numerical computations in Python.

## Objective

The goal of this repository is to explore and model the dynamics of popular optimization algorithms (AdaGrad, RMSProp, and Adam) as nonlocal models. By leveraging integro-differential equations, we aim to provide deeper insights into how these algorithms operate and optimize in high-dimensional spaces.

## How to Use

1. Clone the repository to your local machine.
2. Navigate to the `Solver` folder.
3. Run the appropriate Python file (`NonLocalSolver.py` or `Solver.py`) to explore different algorithms and their dynamics.

## License

This repository is released under the [MIT License](LICENSE).
