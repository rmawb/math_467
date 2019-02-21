import numpy as np

def l_reimann(f, a, b, n):
    """
    Approximates the integral of a function using a left Reimann sum.

    args:
        f: function to integrate
        a: lower bound
        b: upper bound
        n: number of partitions

    returns:
        approximation of integral
    """
    s = 0.0
    x = np.linspace(a, b, n+1)
    step = ((abs(a) + abs(b)) / n)
    for h in x[:-1]:
        s += f(h) * step
    return s

def r_reimann(f, a, b, n):
    """
    Approximates the integral of a function using a right Reimann Sum

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
    step = ((abs(a) + abs(b)) / n)
    for h in x[1:]:
        s += f(h) * step
    return s

def m_reimann(f, a, b, n):
    """
    Approximates the integral of a function using a midpoint Reimann Sum

    args:
        f: function to integrate
        a: lower bound
        b: upper bound
        n: number of partitions

    returns:
        approximation of integral
    """
    s = 0
    z = np.linspace(a, b, n+1)
    x = (z[:-1] + z[1:]) / 2
    step = ((abs(a) + abs(b)) / n)
    for h in x:
        s += f(h) * step
    return s

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
    step = ((abs(a) + abs(b)) / n)
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

def integral_solve(f, a, b, n):
    """
    Approximates the integral of a function using various quadrature rules

    args:
        f: function to integrate
        a: lower bound
        b: upper bound
        n: number of partitions

    returns:
        various approximations of integral as list
    """
    quad = []
    quad.append(l_reimann(f, a, b, n))
    quad.append(r_reimann(f, a, b, n))
    quad.append(m_reimann(f, a, b, n))
    quad.append(trap(f, a, b, n))
    quad.append(gauss_legendre2(f, a, b, n))
    quad.append(gauss_legendre3(f, a, b, n))
    return quad

#Set function, bounds, and number of partitions here
f = lambda x: np.log(x)
l_bound = 1
u_bound = 2
partitions = 12

names = ['Left Reimann sum', 'Right Reimann sum', 'Midpoint Reimann sum',\
         'Trapezoidal rule', '2-point gauss', '3-point gauss']
print("Using " + str(partitions) + " partitions:")
for name, approx in zip(names, integral_solve(f, l_bound, u_bound, partitions)):
    print(name + " approximated the integral to be: " + str(approx))