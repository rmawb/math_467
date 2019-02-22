import numpy as np
import matplotlib.pyplot as plt

def trap(f, a, b, n):
    """
    Approximates the integral of a function using trapezoidal rule

    args:
        f: function to integrate
        a: lower bound
        b: upper bound
        n: number of partitions

    returns:
        approximation of integral
    """
    s = 0
    x = np.linspace(a, b, n+1)
    step = (b-a) / n
    for i in range(len(x)-1):
        s += step * (f(x[i]) + f((x[i+1]))) / 2
    return s

def gauss_legendre2(f, a, b, n):
    """
    Approximates the integral of a function using 2-point gaussian-legendre rule

    args:
        f: function to integrate
        a: lower bound
        b: upper bound
        n: number of partitions

    returns:
        approximation of integral
    """
    s = 0

    #Hard-coded abscissa and weights
    absc = [-np.sqrt(1/3), np.sqrt(1/3)]
    weights = [1, 1]

    #establishing the partitions
    x = np.linspace(a, b, n+1)

    #maps each partition to [-1, 1] and adds solution to sum
    for i in range(len(x)-1):
        t_1 = (x[i+1] - x[i])/2
        t_2 = (x[i+1] + x[i])/2
        s += t_1 * (weights[0] * f(t_1*absc[0] + t_2) +\
                  weights[1] * f(t_1*absc[1] + t_2))
    return s

def gauss_legendre3(f, a, b, n):
    """
    Approximates the integral of a function using 3-point gaussian-legendre rule

    args:
        f: function to integrate
        a: lower bound
        b: upper bound
        n: number of partitions

    returns:
        approximation of integral
    """
    s = 0

    #Hard-coded abscissa and weights
    absc = [-np.sqrt(3/5), 0, np.sqrt(3/5)]
    weights = [5/9, 8/9, 5/9]

    #establishing the partitions
    x = np.linspace(a, b, n+1)

    #maps each partition to [-1, 1] and adds solution to sum
    for i in range(len(x)-1):
        t_1 = (x[i+1] - x[i])/2
        t_2 = (x[i+1] + x[i])/2
        s += t_1 * (weights[0] * f(t_1*absc[0] + t_2) +\
                  weights[1] * f(t_1*absc[1] + t_2) +\
                  weights[2] * f(t_1*absc[2] + t_2))
    return s

#Set function and bounds here
f = lambda x: np.exp(x)
l_bound = 0
u_bound = 20

#set exact integral here:
f_exact = lambda x: np.exp(x)
exact = f_exact(u_bound) - f_exact(l_bound)

n = [2*i for i in range(1,20)]
trapezoid = []
gauss2 = []
gauss3 = []
for x in n:
    trapezoid.append(abs(trap(f, l_bound, u_bound, x) - exact))
    gauss2.append(abs(gauss_legendre2(f, l_bound, u_bound, x) - exact))
    gauss3.append(abs(gauss_legendre3(f, l_bound, u_bound, x) - exact))
print(trapezoid)
print(gauss2)
print(gauss3)

plt.plot(n, trapezoid, label = 'trap')
plt.plot(n, gauss2, label = 'gauss2')
plt.plot(n, gauss3, label = 'gauss3')
plt.legend()
plt.show()