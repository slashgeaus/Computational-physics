import numpy as np
from numpy import dot
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

def is_symmetric(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    if rows != cols:
        return False
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] != matrix[j][i]:
                return False
    return True

def gauss_seidel_method(matrix, b, max_iterations=1000, tolerance=1e-6):
    rows = len(matrix)
    x = [0.0] * rows
    for k in range(max_iterations):
        x_new = [0.0] * rows
        for i in range(rows):
            s1 = sum(matrix[i][j] * x_new[j] for j in range(i))
            s2 = sum(matrix[i][j] * x[j] for j in range(i+1, rows))
            x_new[i] = (b[i] - s1 - s2) / matrix[i][i]
        if all(abs(x[i] - x_new[i]) < tolerance for i in range(rows)):
            break
        x = x_new
        #return rounded x upto 6 decimal places
    print("iterations for gauss siedel:",k)
    return [round(i, 6) for i in x]

def cholesky_decomposition(matrix):
    n = len(matrix)
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = np.sqrt(matrix[i][i] - s)
            else:
                L[i][j] = (1.0 / L[j][j] * (matrix[i][j] - s))
    return L

def cholesky_solver(matrix, b):
    L = cholesky_decomposition(matrix)
    y = np.linalg.solve(L, b)
    x = np.linalg.solve(L.T, y)
    return x

#======================================================================================================================
#let's say I have a matrix A and a matrix B:
# then their augmented matrix code is:
def gauss_jordan(a,b):
    a=np.array(a, float)
    b=np.array(b, float)
    n=len(b)
    m=len(a[0])
    for k in range(n):
        for i in range(k+1, n):#swapping rows
            if a[i,k]>a[k,k]:
                for j in range(k,n):
                    a[k,j],a[i,j]=a[i,j],a[k,j]
                b[k],b[i]=b[i],b[k]
        pivot=a[k][k]   #division of the diagonals
        for j in range(k, n):
            a[k,j] /= pivot
        b[k] /= pivot
        for i in range(n):#elimination
            if i==k or a[i,k]==0: continue
            f=a[i,k]
            for j in range(n):
                a[i,j] = a[i,j]-f*a[k,j]
            b[i] = b[i] - f*b[k]
    return a,b

def make_diagonally_dominant(A, B):
    n = len(A)
    for i in range(n):
        max_index = i
        max_val = sum([abs(A[i][j]) for j in range(n)]) - abs(A[i][i])
        for k in range(i+1, n):
            temp_val = sum([abs(A[k][j]) for j in range(n)]) - abs(A[k][i])
            if temp_val > max_val:
                max_val = temp_val
                max_index = k
        if max_index != i:
            A[[i, max_index]] = A[[max_index, i]]
            B[i], B[max_index] = B[max_index], B[i]
    return A, B

#LU decompose
def LU_decompose(A,b):
    n  = len(A)
    #convert the matrix to upper and lower triangular matrix
    for j in range(n):
        for i in range(n):
            if i <= j :
                    sum = 0
                    for k in range(i):
                        sum += A[i][k]*A[k][j]
                    A[i][j] = A[i][j] - sum
            else  :
                    sum = 0
                    for k in range(j):
                        sum += A[i][k]*A[k][j]
                    A[i][j] = (A[i][j] - sum)/A[j][j]
#forward substitution
    for i in range(n):
        sum = 0
        for j in range(i):
            sum += A[i][j]*b[j]
        b[i] = b [i] - sum
#backward substitution
    for i in range(n-1,-1,-1):
        sum = 0
        for j in range(i+1,n):
            sum += A[i][j]*b[j]
        b[i] = (b[i] - sum)/(A[i][i])
    return b

#======================================================================================================================
def dot_product1(A, b):
    B =  b.reshape(-1, 1)
    result = np.zeros((len(A), len(B[0])))
    
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
                
    return result.flatten()

def dot_product2(A, B):

    result = 0
    for i in range(len(A)):
        result += A[i] * B[i]
        
    return result

def cg(A, x0, b, max_iterations, tolerance):
    x = x0
    r = b - A.dot(x)
    p = r
    rsold = r.dot(r)
    
    for i in range(max_iterations):
        Ap = A.dot(p)
        alpha = rsold / p.dot(Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r.dot(r)
        
        if np.sqrt(rsnew) < tolerance:
            break
        
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    
    return x

#======================================================================================================================

def mat(x, y):
    return (delta(x+1, y) + delta(x-1, y) - 2*delta(x, y))*0.5 + 0.04* delta(x, y)
    # return (delta(x+1, y) + delta(x-1, y) + 2*delta(x, y))*0.5
    # return delta(x,y)

def dot2(x):
    n = len(x)
    r = np.zeros(n)
    for row in range(n):
        for i in range(n):
                r[row] += mat(row,i)*x[i]
    return np.array(r)


def delta(x, y):
    if x == y:
        return 1
    else:
        return 0
    
def cg_fly( x0, b, max_iterations, tolerance):
    x = x0
    r = b - dot2(x)
    p = r
    rsold = r.dot(r)
    res = []

    for i in range(max_iterations):
        Ap = dot2(p)
        alpha = rsold / p.dot(Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r.dot(r)
        if np.sqrt(rsnew) < tolerance:
            break
        res.append(rsold)
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x, np.array(res)

#======================================================================================================================
def fileread(s,n=6):
    # Read the matrix A from the file
    A = np.loadtxt(s, max_rows=n)

    # Read the vector B from the file
    B = np.loadtxt(s, skiprows=n)

    return A, B