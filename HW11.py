import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

# Given differential equation: y'' = -(x+1)y' + 2y + (1-x²)e^(-x)
# Boundary conditions: y(0) = 1, y(1) = 2
# Domain: 0 ≤ x ≤ 1
# Step size: h = 0.1

def differential_equation(x, y, dy_dx):
    """
    The differential equation: y'' = -(x+1)y' + 2y + (1-x²)e^(-x)
    """
    d2y_dx2 = -(x + 1) * dy_dx + 2 * y + (1 - x**2) * np.exp(-x)
    return d2y_dx2

# Method 1: Shooting Method
def shooting_method():
    """
    Shooting method to solve the boundary value problem
    We'll guess the initial slope y'(0) and adjust until y(1) = 2
    """
    print("=" * 60)
    print("METHOD 1: SHOOTING METHOD")
    print("=" * 60)
    
    h = 0.1
    x_span = np.arange(0, 1.01, h)
    
    def system_ode(t, y):
        """Convert 2nd order ODE to system of 1st order ODEs"""
        y1, y2 = y  # y1 = y, y2 = y'
        dy1_dt = y2
        dy2_dt = -(t + 1) * y2 + 2 * y1 + (1 - t**2) * np.exp(-t)
        return [dy1_dt, dy2_dt]
    
    def solve_with_slope(slope):
        """Solve IVP with given initial slope"""
        y0 = [1, slope]  # y(0) = 1, y'(0) = slope
        sol = solve_ivp(system_ode, [0, 1], y0, t_eval=x_span, method='RK45')
        return sol.y[0][-1]  # Return y(1)
    
    def objective(slope):
        """Objective function: |y(1) - 2|"""
        return abs(solve_with_slope(slope) - 2)
    
    # Find the correct initial slope using optimization
    from scipy.optimize import minimize_scalar
    result = minimize_scalar(objective, bounds=(-10, 10), method='bounded')
    optimal_slope = result.x
    
    # Solve with optimal slope
    y0 = [1, optimal_slope]
    sol = solve_ivp(system_ode, [0, 1], y0, t_eval=x_span, method='RK45')
    
    x_shoot = sol.t
    y_shoot = sol.y[0]
    dy_shoot = sol.y[1]
    
    print(f"Boundary conditions: y(0) = 1, y(1) = 2")
    print(f"Optimal initial slope: y'(0) = {optimal_slope:.6f}")
    print(f"Achieved final value: y(1) = {y_shoot[-1]:.6f}")
    print(f"Error in boundary condition: {abs(y_shoot[-1] - 2):.8f}")
    print("\nSolution points:")
    for i in range(len(x_shoot)):
        print(f"x = {x_shoot[i]:.1f}, y = {y_shoot[i]:.6f}, y' = {dy_shoot[i]:.6f}")
    
    return x_shoot, y_shoot, dy_shoot

# Method 2: Finite Difference Method
def finite_difference_method():
    """
    Finite difference method to solve the boundary value problem
    """
    print("\n" + "=" * 60)
    print("METHOD 2: FINITE DIFFERENCE METHOD")
    print("=" * 60)
    
    h = 0.1
    n = int(1/h) + 1  # Number of grid points
    x = np.linspace(0, 1, n)
    
    # Create coefficient matrix A and right-hand side vector b
    A = np.zeros((n, n))
    b = np.zeros(n)
    
    # Apply boundary conditions
    A[0, 0] = 1
    b[0] = 1  # y(0) = 1
    
    A[n-1, n-1] = 1
    b[n-1] = 2  # y(1) = 2
    
    # Apply finite difference scheme for interior points
    # y'' ≈ (y[i+1] - 2*y[i] + y[i-1])/h²
    # y' ≈ (y[i+1] - y[i-1])/(2*h)
    # Rearranging: y'' + (x+1)y' - 2y = (1-x²)e^(-x)
    for i in range(1, n-1):
        xi = x[i]
        # Coefficient of y[i-1]
        A[i, i-1] = 1/h**2 + (xi + 1)/(2*h)
        # Coefficient of y[i]
        A[i, i] = -2/h**2 - 2
        # Coefficient of y[i+1]
        A[i, i+1] = 1/h**2 - (xi + 1)/(2*h)
        # Right-hand side
        b[i] = -(1 - xi**2) * np.exp(-xi)
    
    # Solve the linear system
    y_fd = np.linalg.solve(A, b)
    
    print(f"Grid points: {n}")
    print(f"Step size: h = {h}")
    print("Boundary conditions: y(0) = 1, y(1) = 2")
    print("\nSolution points:")
    for i in range(len(x)):
        print(f"x = {x[i]:.1f}, y = {y_fd[i]:.6f}")
    
    return x, y_fd

# Method 3: Variational Approach (Galerkin Method)
def variational_method():
    """
    Variational approach using Galerkin method with polynomial basis functions
    """
    print("\n" + "=" * 60)
    print("METHOD 3: VARIATIONAL APPROACH (GALERKIN METHOD)")
    print("=" * 60)
    
    # Use polynomial basis functions that satisfy boundary conditions
    # Let y(x) = 1 + x + sum(c_i * x^i * (1-x))
    # This ensures y(0) = 1 and contributes to y(1) = 2
    
    def basis_function(x, i):
        """Modified basis function that satisfies boundary conditions"""
        if i == 0:
            return 1 + x  # Linear function: y(0) = 1, y(1) = 2
        else:
            return x**i * (1 - x)  # Polynomial terms that vanish at boundaries
    
    def basis_derivative(x, i, order=1):
        """Derivative of basis function"""
        if order == 1:
            if i == 0:
                return np.ones_like(x)
            else:
                return i * x**(i-1) * (1 - x) - x**i
        elif order == 2:
            if i == 0:
                return np.zeros_like(x)
            elif i == 1:
                return -2 * np.ones_like(x)
            else:
                return i * (i-1) * x**(i-2) * (1 - x) - 2 * i * x**(i-1)
    
    n_basis = 5  # Number of basis functions
    x_quad = np.linspace(0, 1, 101)  # Quadrature points
    
    # The approximation is y(x) = φ_0(x) + sum(c_i * φ_i(x))
    # where φ_0 ensures boundary conditions are met
    
    A_var = np.zeros((n_basis-1, n_basis-1))
    b_var = np.zeros(n_basis-1)
    
    # Galerkin conditions: ∫ φ_j * [L[y] - f] dx = 0 for j = 1,2,...,n-1
    # where L[y] = y'' + (x+1)y' - 2y and f = (1-x²)e^(-x)
    
    for i in range(n_basis-1):  # Test functions
        for j in range(n_basis-1):  # Trial functions
            # Integrate φ_{i+1} * [L[φ_{j+1}]] dx
            L_phi = (basis_derivative(x_quad, j+1, 2) + 
                    (x_quad + 1) * basis_derivative(x_quad, j+1, 1) - 
                    2 * basis_function(x_quad, j+1))
            integrand = basis_function(x_quad, i+1) * L_phi
            A_var[i, j] = np.trapz(integrand, x_quad)
        
        # Right-hand side: ∫ φ_{i+1} * [f - L[φ_0]] dx
        L_phi0 = (basis_derivative(x_quad, 0, 2) + 
                 (x_quad + 1) * basis_derivative(x_quad, 0, 1) - 
                 2 * basis_function(x_quad, 0))
        f_val = (1 - x_quad**2) * np.exp(-x_quad)
        rhs_integrand = basis_function(x_quad, i+1) * (f_val - L_phi0)
        b_var[i] = np.trapz(rhs_integrand, x_quad)
    
    # Solve for coefficients
    c = np.linalg.solve(A_var, b_var)
    
    # Evaluate the approximate solution
    x_var = np.linspace(0, 1, 11)
    y_var = basis_function(x_var, 0)  # Start with φ_0
    
    for i in range(len(c)):
        y_var += c[i] * basis_function(x_var, i+1)
    
    print(f"Number of basis functions: {n_basis}")
    print(f"Coefficients for higher-order terms: {c}")
    print(f"Boundary conditions satisfied: y(0) = {y_var[0]:.6f}, y(1) = {y_var[-1]:.6f}")
    print("\nSolution points:")
    for i in range(len(x_var)):
        print(f"x = {x_var[i]:.1f}, y = {y_var[i]:.6f}")
    
    return x_var, y_var, c

# Main execution
def main():
    print("SOLVING BOUNDARY VALUE PROBLEM: y'' = -(x+1)y' + 2y + (1-x²)e^(-x)")
    print("Boundary conditions: y(0) = 1, y(1) = 2")
    print("Domain: 0 ≤ x ≤ 1, Step size: h = 0.1")
    
    # Solve using all three methods
    x1, y1, dy1 = shooting_method()
    x2, y2 = finite_difference_method()
    x3, y3, c = variational_method()


if __name__ == "__main__":
    main()