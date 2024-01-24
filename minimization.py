import numpy as np


def fmin_beres(valueF, gradF, hessF, S, f_presc, v, n, L, nn, epsilon):
    """
    Minimizes a function using a modified Newton's method.

    Args:
    - S, f_presc, v, n, L: Parameters for the function F.
    - nn: Number of samples for the Latin Hypercube sampling.
    - epsilon: Tolerance for the stopping criterion in Newton's method.

    Returns:
    - x: The point at which the minimum is found.
    - it: The number of iterations used in the final call to mini_newton.
    """

    focal_length = 0.008
    # Define the function F and its derivatives
    def F(x): return valueF(x, v, n, L, focal_length, S * 1e-3, f_presc)
    def dF(x): return gradF(x, v, n, L, focal_length, S * 1e-3, f_presc)
    def ddF(x): return hessF(x, v, n, L, focal_length, S * 1e-3, f_presc)

    # Define bounds and generate initial points
    bounds = np.array([50e-3, 50e-3, 50e-3, 20 * np.pi / 180, 20 * np.pi / 180, 20 * np.pi / 180])
    points = bounds * latin_hypercube(nn, 6)

    # Evaluate F at initial points and refine using mini_newton
    F_vals = np.zeros(nn)
    for i in range(nn):
        F_loc = F(points[:, i])
        if F_loc < 1:
            x, _ = mini_newton(points[:, i], F, dF, ddF, 3, 1e-1)
            x[3:6] = np.mod(x[3:6], 2 * np.pi)
            x[3:6] -= 2 * np.pi * (x[3:6] > 0.5)
            x[3:6] += 2 * np.pi * (x[3:6] < -0.5)
            points[:, i] = x
            F_loc = F(x)
        F_vals[i] = F_loc

    # Sort and select top samples
    sort_idx = np.argsort(F_vals)
    points = points[:, sort_idx]
    for i in range(int(np.ceil(nn / 100))):
        x, it = mini_newton(points[:, i], F, dF, ddF, 100, epsilon)
        points[:, i] = x
        F_vals[i] = F(x)

    sort_idx = np.argsort(F_vals)
    points = points[:, sort_idx]
    x = points[:, 0]
    x[3:6] = np.mod(x[3:6], 2 * np.pi)
    x[3:6] -= 2 * np.pi * (x[3:6] > 0.5)
    x[3:6] += 2 * np.pi * (x[3:6] < -0.5)

    return x


def mini_newton(x0, F, dF, ddF, maxit, eps):
    """
    Modified Newton's method for optimization.

    Args:
    - x0: Initial guess for the minimum.
    - F: Function to be minimized.
    - dF: Gradient of the function.
    - ddF: Hessian of the function.
    - maxit: Maximum number of iterations.
    - eps: Tolerance for convergence.

    Returns:
    - x: Estimated position of the minimum.
    - it: Number of iterations used.
    """
    x = x0
    it = 0
    for it in range(1, maxit + 1):
        g = dF(x)
        H = ddF(x)
        g_norm = np.linalg.norm(g)
        d = -np.linalg.solve(H, g)
        alpha = golden_section_line_search(x, d, F, 0, 1, 1e-3)
        x = x + alpha * d
        if g_norm < eps:
            break

    return x, it


def golden_section_line_search(x, d, F, a, b, eps):
    """
    Golden section line search to find the minimum of a unimodal function along a given direction.

    Args:
    - x: Current point.
    - d: Direction for line search.
    - F: Function to be minimized.
    - a, b: Endpoints of the initial interval.
    - eps: Tolerance for convergence.

    Returns:
    - alpha: Step size that approximately minimizes F along direction d.
    """
    rho = (np.sqrt(5) - 1) / 2  # Golden ratio
    a0, b0 = a, b
    a1 = b0 - rho * (b0 - a0)
    b1 = a0 + rho * (b0 - a0)
    F_a1 = F(x + a1 * d)
    F_b1 = F(x + b1 * d)

    while abs(b0 - a0) > eps:
        if F_a1 < F_b1:
            b0 = b1
            b1 = a1
            F_b1 = F_a1
            a1 = b0 - rho * (b0 - a0)
            F_a1 = F(x + a1 * d)
        else:
            a0 = a1
            a1 = b1
            F_a1 = F_b1
            b1 = a0 + rho * (b0 - a0)
            F_b1 = F(x + b1 * d)

    alpha = (a0 + b1) / 2 if F_a1 < F_b1 else (a1 + b0) / 2

    return alpha


def latin_hypercube(num_samples, num_dimensions):
    """
    Generates a Latin Hypercube Sampling.

    Args:
    - num_samples: Number of samples to generate.
    - num_dimensions: Number of dimensions.

    Returns:
    - samples: Generated samples as a numpy array.
    """
    samples = np.zeros((num_samples, num_dimensions))
    intervals = np.linspace(-1, 1, num_samples + 1)

    for i in range(num_dimensions):
        permuted_intervals = np.random.permutation(intervals[:num_samples])
        mid_points = permuted_intervals + (intervals[1] - intervals[0]) / 2
        samples[:, i] = mid_points

    samples = (samples.T + (np.random.rand(num_dimensions, num_samples) - 0.5) * (2 / num_samples)).T
    return samples
