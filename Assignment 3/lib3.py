import numpy as np
import scipy as scipy
import math as math
from scipy.optimize import root
import pandas as pd
import scipy.stats as stats
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

def fileread(s,n=6):
    # Read the matrix A from the file
    A = np.loadtxt(s, max_rows=n)

    # Read the vector B from the file
    B = np.loadtxt(s, skiprows=n)

    return A, B

def power_method(A :list,x0: list,tol = 1e-6):
    
    A = np.array(A)
    x0 = np.array(x0)
    x_copy = np.copy(x0)
    lam_0 = np.matmul(np.matmul(np.linalg.matrix_power(A,2),x0).T,np.matmul(np.linalg.matrix_power(A,1),x0))/np.matmul(np.matmul(np.linalg.matrix_power(A,1),x0).T,np.matmul(np.linalg.matrix_power(A,1),x0))
    lam_1 = np.matmul(np.matmul(np.linalg.matrix_power(A,3),x0).T,np.matmul(np.linalg.matrix_power(A,2),x0))/np.matmul(np.matmul(np.linalg.matrix_power(A,2),x0).T,np.matmul(np.linalg.matrix_power(A,2),x0))
    i=3
    while abs(lam_1-lam_0)>tol:
        lam_0 = lam_1
        lam_1 = np.matmul(np.matmul(np.linalg.matrix_power(A,i+1),x0).T,np.matmul(np.linalg.matrix_power(A,i),x0))/np.matmul(np.matmul(np.linalg.matrix_power(A,i),x0).T,np.matmul(np.linalg.matrix_power(A,i),x0))
        i+=1

    eigval = lam_1
    eigvec = np.matmul(np.linalg.matrix_power(A,i-1),x_copy)
    norm = np.linalg.norm(eigvec)
    eigvec = eigvec/norm
    return eigval,eigvec,i  

def QR(A,tolerance = 1e-6):
    A = np.array(A)
    copy_A = np.copy(A)
    Q,R = QR_factorize(A)
    A = np.matmul(R,Q)
    i=1
    while np.linalg.norm(A-copy_A)>tolerance:
        copy_A = np.copy(A)
        Q,R = QR_factorize(A)
        A = np.matmul(R,Q)
        i+=1
    return np.diag(A),i

def QR_factorize(A):
    A = np.array(A) if type(A) != np.ndarray else A
    Q = np.zeros(A.shape)
    R = np.zeros(A.shape)
    for i in range(A.shape[1]):
        u_i = A[:,i]
        sum = 0
        for j in range(i):
            sum += np.dot(A[:,i],Q[:,j])*Q[:,j]
        u_i = u_i - sum
        Q[:,i] = u_i/np.linalg.norm(u_i)
        for j in range(i+1):
            R[j,i] = np.dot(A[:,i],Q[:,j])
            
    return Q,R

def read_file(filename: str,delimiter: str = '\t'):

    matrices = []
    current_matrix = []

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()  
            if not line or line.startswith("#"):
                if current_matrix: 
                    matrices.append(current_matrix)
                    current_matrix = []  
                continue
            
            try:
                row = [float(num) for num in line.split(delimiter)]
                current_matrix.append(row)
            except ValueError:
                # print("Skipping non-numeric line:", line)
                pass
        if current_matrix:
            matrices.append(current_matrix)
    return matrices

def polynomial_fit(xlist: list,ylist: list,sigma_list: list,degree: int,tol=1e-6):
    xlist = np.array(xlist)
    ylist = np.array(ylist)
    sigma_list = np.array(sigma_list)
    A_matrix = np.zeros((degree+1,degree+1))

    for i in range(degree+1):
        for j in range(degree+1):
            A_matrix[i][j] = np.sum((xlist**(i+j))/(sigma_list**2))
    B_matrix = np.zeros(degree+1)
    for i in range(degree+1):
        B_matrix[i] = np.sum((ylist*(xlist**i))/(sigma_list**2))
    # a = Gauss_seidel_solve(A_matrix.tolist(),B_matrix.tolist(),T=tol)
    a = np.linalg.solve(A_matrix,B_matrix)    
    return a,A_matrix

def poly_fn(x,coefflist):
    sum = 0
    for i in range(len(coefflist)):
        sum += coefflist[i]*x**i
    return sum  

def polynomial_fit_mod_chebyshev(xlist: list,ylist: list,sigma_list: list,degree: int):
    # Defining the modified chebyshev polynomial
    def modified_chebyshev_polynomial(x,degree):
        def chebyshev_polynomial(x,degree):
            if degree == 0:
                return 1
            elif degree == 1:
                return x
            else:
                return 2*x*chebyshev_polynomial(x,degree-1) - chebyshev_polynomial(x,degree-2)
        return chebyshev_polynomial(2*x - 1,degree)
    xlist = np.array(xlist)
    ylist = np.array(ylist)
    sigma_list = np.array(sigma_list)
    A_matrix = np.zeros((degree+1,degree+1))

    for i in range(degree+1):
        for j in range(degree+1):
            # Replace the polynomial with the modified chebyshev polynomial
            A_matrix[i][j] = np.sum((modified_chebyshev_polynomial(xlist,i)*modified_chebyshev_polynomial(xlist,j))/(sigma_list**2))
    B_matrix = np.zeros(degree+1)
    for i in range(degree+1):
        B_matrix[i] = np.sum((ylist*(modified_chebyshev_polynomial(xlist,i)))/(sigma_list**2))
    a = np.linalg.solve(A_matrix,B_matrix)    
    return a,A_matrix

def modified_chebyshev_polynomial(x,degree):
    def chebyshev_polynomial(x,degree):
        if degree == 0:
            return 1
        elif degree == 1:
            return x
        else:
            return 2*x*chebyshev_polynomial(x,degree-1) - chebyshev_polynomial(x,degree-2)
    return chebyshev_polynomial(2*x - 1,degree)


def poly_fn_mod(x,coefflist):
    sum = 0
    for i in range(len(coefflist)):
        sum += coefflist[i]*modified_chebyshev_polynomial(x,i)
    return sum    

