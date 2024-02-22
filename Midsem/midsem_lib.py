def bracketing(a0,b0,func):
    counter=0
    a0=1.5
    b0=2.5
    while(func(a0)*func(b0)>0):
        c0=abs(a0-b0)/2
        if(abs(func(a0))<abs(func(b0))):
            a0=a0-c0
        else:
            b0=b0+c0
        counter+=1
        print("The appropriate interval for root finding is:")
        return a0,b0

#Function to find root using Regula-Falsi method
def Regula_falsi(fn, a1, b1,e=6):
    a,b = a1, b1
    if (fn(a) * fn(b) >= 0):
        print("You have not assumed right a and b\n")
        return None
    n = 0
    c=a
    c1 = c-1
    while abs(c-c1) > 10**-e or abs(fn(c)) > 10**-e:
        n += 1
        c = (a * fn(b) - b * fn(a))/ (fn(b) - fn(a))
        #if abs(fn(c)) <= 10**-e:
            #break
        if (fn(c)*fn(a)<0):
            b = c
        elif (fn(c)*fn(b)<0):
            a = c
        c1 = c
        print(f"Iteration no: {n} \troot -> {c:{1}.{e}f}")
    print(f"\nThe root in the given interval converges to {c:{1}.{e}f} and the value of function is {fn(c):{1}.{e}f}")
    print("Total no of iterations = ",n)
    #return c

#Function to find root using Newton_raphson method
def Newton_Raphson(fn,d_fn,x = 0.5,e=6):
    '''fn: the function of which we want to find the root,
       d_fn: the derivative of the function
       x: initial guess for the root'''
    h = fn(x)/d_fn(x)
    n = 0
    while abs(h)>10**-e:
        x -= h
        h = fn(x)/d_fn(x)
        n += 1
        print(f"Iteration no: {n} \troot -> {x:{1}.{e}f}")
    print(f"The root converges to {x:{1}.{e}f} and the value of function is {fn(x):{1}.{e}f}")
    print("\nTotal no of iterations = ",n)
    return round(x,e)

#PDE solver
def PDE_Solve(lx,Nx,lt,Nt,lower_x,tot_steps):
    step_arr = [10, 20, 50, 100, 200, 500, 1000, tot_steps]
    hx=(lx/Nx)
    ht=(lt/Nt)
    alpha=ht/(hx)**2
    V0=np.zeros(Nx+1)
    V1=np.zeros(Nx+1)
    x_cor = np.linspace(lower_x, lx, Nx + 1)
    ctr=0 #marker for the value in step_arr
    #if alpha<=0.5:print("Stability can be a problem")
    for i in range(0,Nx+1):
        if lower_x + (hx * i) == 1:#1 as inital 300C at length 1
            V0[i]=300
        else:
            V0[i]=0
        x_cor[i]=(lower_x + hx * i)
    plt.plot(x_cor, V0, label=0)
    #Matrix mult for sparse when only some are multiplied
    for j in range(0,tot_steps+1):#1000 is number of steps taken
        for i in range(0,Nx+1):

            if i==0:
                V1[i]=(1-2*alpha)*V0[i]+alpha*V0[i+1]
            elif i==Nx:
                V1[i]=(1-2*alpha)*V0[i]+alpha*V0[i-1]
            else:
                V1[i]=(1-2*alpha)*V0[i]+alpha*V0[i-1]+alpha*V0[i+1]
        for k in range(0,Nx+1):#Equating array V0 to V1
            V0[k]=V1[k]
        if j==step_arr[ctr]:
            plt.plot(x_cor,V1,label=step_arr[ctr])
            #print(V0[50])
            ctr=ctr+1
    plt.legend()
    return None

import matplotlib.pyplot as plt
import numpy as np
import math

#Quadrature
def int_quad(a,b,f,N=4):
    h=(b-a)/N
    I2=0
    for i in range(1,N+1):
        mid=(a+(2*i-1)*h/2)
        fm2= f(mid)
        I2+= round(fm2*h,8)
    return(I2)
    
#LU
#Function to decompose a square matrix into lower and upper matrix
def LU_decompose(A):
    n = len(A)
    U = [A[i][:] for i in range(n)]
    L = Identity(n)
    for i in range(n):
        for j in range(i+1,n):
            L[j][i] = U[j][i]/U[i][i]
            U[j] = [U[j][k]-L[j][i]*U[i][k] for k in range(n)]
    return L,U

#Function to perform forward substitution
def forward_sub(L,B):
    n = len(L)
    b = [B[i][0] if type(B[i])==list else B[i] for i in range(n)]
    #print(b)
    y = n*[0]
    y[0] = b[0]
    for i in range(1,n):
        y[i] = b[i] - sum(L[i][j]*y[j] for j in range(i))
    return y

#Function to perform backward substitution
def backward_sub(U,y):
    n = len(U)
    x = n*[0]
    x[-1] = y[-1]/U[-1][-1]
    for i in range(n-2,-1,-1):
        x[i] = (y[i] - sum(U[i][j]*x[j] for j in range(i+1,n)))/U[i][i]
    return x

#Function to solve a system of linear equation using LU-Decomposition
def LU(A,B,r=6):
    L,U = LU_decompose(A)
    y = forward_sub(L,B)
    x = backward_sub(U,y)
    return [round(i,r) for i in x]
    
def Identity(n):
    I_n = [n*[0] for i in range(n)]
    for i in range(n):
        I_n[i][i] = 1
    return I_n

    

    
#Shooter
def Shooter(
    y_0: float, 
    y_L: float, 
    y_x: float, 
    N: int
) -> float:
    """
    This function uses the RK4 method to solve the one dimensional diffusion equation
    given the initial conditions, the length of the rod, the position where the 
    temperature is to be found, and the number of iterations. The function then 
    plots the solution and returns the position where the temperature is 100 degrees 
    celsius.

    Parameters:
    -----------
    y_0: float
        The initial temperature at the start of the rod.
    y_L: float
        The final temperature at the end of the rod.
    y_x: float
        The position where the temperature is to be found.
    N: int
        The number of iterations to use in the RK4 method.

    Returns:
    --------
    float
        The position where the temperature is 100 degrees celsius.

    """
    z1 = -0.5  # initial guess
    z1h, y1 = RK4_boundary_value(
        y_0, z1, 0, y_x, 10, 1
    )  # initialiing
    z2 = -2  # initial guess
    z2h, y2 = RK4_boundary_value(
        y_0, z2, 0, y_x, 10, 1
    )  # initialiing
    iter = 0
    while abs(y1 - y_L) >= 0.001 and abs(y2 - y_L) >= 0.001 and iter <= 30:
        iter += 1
        znew = z2 + ((z1 - z2) * (y_L - y2)) / (y1 - y2)
        znew2, ynew = RK4_boundary_value(
            y_0, znew, 0, N, 10, 1
        )  # newton raphson
        # print(ynew, znew)
        if abs(ynew - y_L) < 0.001:
            z, y, x = RK4_boundary_value(
                y_0, znew, 0, N, 10, 0
            )  # final solution
            break
        else:
            if ynew < y_L:
                z2 = znew
                y2 = ynew
            else:
                z1 = znew
                y1 = ynew
    plt.plot(x, y)
    for i in range(0, len(y)):
        if abs(y[i] - 100) < 0.1:  # to get value of position at temperature 100
            out = x[i]
            break
    return out

def RK4_boundary_value(y0, z0,x0, N, end, interpol):
    y_i = y0
    z_i = z0
    step = 0
    yl = [y0]
    zl = [z0]
    xl = [x0]
    h = (end-0)/N
    while step <= end:
        k1y = h*boundary_RK4_dyx(y_i, z_i, step)
        k1z = h*boundary_RK4_dzx(y_i, z_i, step)

        k2y = h*boundary_RK4_dyx(y_i + k1y/2, z_i+k1z/2, step+h/2)
        k2z = h*boundary_RK4_dzx(y_i + k1y/2, z_i+k1z/2, step+h/2)

        k3y = h*boundary_RK4_dyx(y_i+k2y/2, z_i+k2z/2, step+h/2)
        k3z = h*boundary_RK4_dzx(y_i+k2y/2, z_i+k2z/2, step+h/2)

        k4y = h*boundary_RK4_dyx(y_i+k3y, z_i+k3z, step+h)
        k4z = h*boundary_RK4_dzx(y_i+k3y, z_i+k3z, step+h)

        y_i += (k1y+2*k2y+2*k3y+k4y)/6
        z_i += (k1z+2*k2z+2*k3z+k4z)/6
        yl.append(y_i)
        zl.append(z_i)
        step += h
        xl.append(step)

    if interpol == 0:
        return zl, yl, xl
    else:
        return z_i, y_i
    
def boundary_RK4_dyx(y,z,x):
    f = z
    return f

def boundary_RK4_dzx(y,z,x):
    f = 0.01*(y-20)
    return f
    
# File reader   
def finput(inputf):
    data=[]

    with open(inputf) as f:
        for line in f:
            data.append(line.split(","))


    n=len(data)

    a=np.zeros((n,n))#matrix
    b=np.zeros(n)#vector

    for i in range(n):
        b[i]=(int)(data[i][n])
        for j in range(n):
            a[i][j]=(int)(data[i][j])
    return a,b
