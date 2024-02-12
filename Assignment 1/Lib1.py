import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy import sparse
#===========================================================================================
#Q1

def g(x):
    return math.exp(-x)

def is_close_enough(x, prev_x):
    return abs(x - prev_x) < 1e-4

def fixed_point_iteration(x0):
    prev_x = x0 - 1
    x = g(x0)
    n = 0
    while not is_close_enough(x, prev_x):
        prev_x = x
        x = g(x)
        n += 1
    return x, n
    
#===========================================================================================
#Q2

#Simpson
# Function for approximate integral
def simpsons_(func, ll, ul, n ):
 
    # Calculating the value of h
    h = ( ul - ll )/n
 
    # List for storing value of x and f(x)
    x = list()
    fx = list()
     
    # Calculating values of x and f(x)
    i = 0
    while i<= n:
        x.append(ll + i * h)
        fx.append(func(x[i]))
        i += 1
 
    # Calculating result
    res = 0
    i = 0
    while i<= n:
        if i == 0 or i == n:
            res+= fx[i]
        elif i % 2 != 0:
            res+= 4 * fx[i]
        else:
            res+= 2 * fx[i]
        i+= 1
    res = res * (h / 3)
    return res
    
#Gaussian Quadrature
def gaussian_quadrature(f, a, b, n):
    # Compute the sample points and weights from legendre polynomials
    x, w = np.polynomial.legendre.leggauss(n)
    # Change of variables
    t = 0.5 * (x + 1) * (b - a) + a
    return np.sum(w * f(t)) * 0.5 * (b - a)
    
#===============================================================================================
#Q3

def f(x, y):
    return (5 * x**2 - y) / math.exp(x + y)

def rk4_step(x, y, h):
    k1 = h * f(x, y)
    k2 = h * f(x + h/2, y + k1/2)
    k3 = h * f(x + h/2, y + k2/2)
    k4 = h * f(x + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

def solve_ode_rk4(initial_x, initial_y, interval_size, num_steps):
    solution = [(initial_x, initial_y)]
    x = initial_x
    y = initial_y
    for _ in range(num_steps):
        y = rk4_step(x, y, interval_size)
        x += interval_size
        solution.append((x, y))
    return solution
    
#==================================================================================================
#Q4

def crank_nicolson(g, L, n, T, dt):
    dx = L / (n + 1)
    alpha = dt / (dx**2)
    A = np.zeros((n, n))
    B = np.zeros((n, n))

    # Create tridiagonal matrices A and B
    for i in range(n):
        A[i, i] = 2 + 2 * alpha
        B[i, i] = 2 - 2 * alpha
        for j in range(n):
            if j == i + 1 or j == i - 1:
                A[i, j] = -alpha
                B[i, j] = alpha

    x = [0]
    for i in range(n - 1):
        x.append(x[i] + dx)

    # Initialize initial values of the vector v0 using g(x)
    v0 = np.array([g(xi) for xi in x])
    v0[-1] = v0[0]

    v = v0.copy()
    solution_at_each_time = [v0.copy()]

    # Time-stepping using Crank-Nicolson method
    for _ in range(int(T / dt)):
        C = np.matmul(np.linalg.inv(A), B)
        v = np.matmul(C, v)
        solution_at_each_time.append(v.copy())

    return solution_at_each_time, x
    
#=====================================================================================================
