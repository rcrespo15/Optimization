from __future__ import division
import time
import numpy as np
from operator import itemgetter
from cvxpy import *
from cvxpy.atoms.atom import Atom

################################################################################
# KKT Conditions
#
# In this exercise, we will solve our quadratic optimization problem with the
# bisection algorithm and compare the method to the quadratic solver of CVX
# (or another optimizer of your choice). Although throughout the problem we refer
# to the CVX optimizer, you are free to replace CVX with another solver of your
# choice.
################################################################################

data = np.load('data.npz')
d, r, a = itemgetter('d', 'r', 'a')(data)

################################################################################
#Solve with cvx
################################################################################
def solve_cvx(d, r, a):
    """
    Solve the quadratic program

    min_x   1/2 * x^T D x + r^T x
     s.t.   -1 <= x <= 1
            a^T x = 1

    using an optimizer of your choice (CVXOpt, CVXPy, scipy.optimize, etc.)

    Args:
        d (numpy.ndarray): the values that form the diagonal entries of the D matrix
        r (numpy.ndarray): the values that form the r vector
        a (numpy.ndarray): the values that form the a vector

    Returns:
        the optimal solution, x_opt
        the objective value, obj_val
    """
    n,_ = r.shape
    D = d*np.eye(n)
    x = Variable(n)
    objective = Minimize((1/2)*quad_form(x,D) +r.T*x) #1/2)*x.T*D*x +
    constraints = [x >= -1]
    constraints += [x <= 1]
    constraints += [a.T*x == 1]
    emd = Problem(objective, constraints)
    emd.solve()
    x_opt = x.value
    obj_val = emd.value
    # TODO: compute x_opt and obj_val using CVX
    return x_opt, obj_val

start = time.time()
x_opt_cvx, obj_val_cvx = solve_cvx(d, r, a)
end = time.time()
solve_time_cvx = end - start
print('CVX objective value: {}'.format(obj_val_cvx))
print('CVX solve time: {}'.format(solve_time_cvx))


################################################################################
#Solve with bisection
################################################################################

def get_x_star(m,a,r,d):
    n,d = a.shape
    lambda_ = (m*a+r-d)/2
    dummy = np.zeros((n,2))
    dummy[:,0] = lambda_.T
    num = (-m*a-r)
    den = (d+2*np.amax(dummy,axis = 1))
    x_star = num[:,0]/den
    return x_star

def bisection_step(d,r,a,mu_l,mu_r):
    mu = (mu_r+mu_l)/2
    x_star = get_x_star(mu,a,r,d)
    h = np.dot(a.T,x_star)-1
    if h > 0:
        mu_l = mu
    else:
        mu_r = mu
    new_mu = (mu_r+mu_l)/2
    return mu_l,mu_r,new_mu,x_star

def solve_bisection(d, r, a, mu_l=-200., mu_r=10., eps=1e-6):
    """
    Solve the quadratic program

    min_x   1/2 * x^T D x + r^T x
     s.t.   -1 <= x <= 1
            a^T x = 1

    using the bisection method

    Args:
        d (numpy.ndarray): the values that form the diagonal entries of the D matrix
        r (numpy.ndarray): the values that form the r vector
        a (numpy.ndarray): the values that form the a vector
        mu_l (float): lower bound of initial interval for mu
        mu_r (float): upper bound of initial interval for mu
        eps (float): epsilon value for termination condition

    Returns:
        the optimal solution, x_opt
        the objective value, obj_val
    """
    n,_ = r.shape
    D = d*np.eye(n)
    mu = mu_r+mu_l
    x_star = 0
    condition = True
    count = 0
    while count <= 28:
        new_mu_l,new_mu_r,new_mu,new_x_star = bisection_step(d,r,a,mu_l,mu_r)
        mu_l = new_mu_l
        mu_r = new_mu_r
        x_star = new_x_star
        count +=1
    x_opt = new_x_star
    obj_val = np.dot(np.dot(x_opt.T,D),x_opt) + np.dot(r.T,x_opt)
    # TODO: compute x_opt and obj_val using the bisection method
    return x_opt, obj_val

start = time.time()
x_opt_bisection, obj_val_bisection = solve_bisection(d, r, a)
end = time.time()
solve_time_bisection = end - start
print('Bisection objective value: {}'.format(obj_val_bisection))
print('Bisection solve time: {}'.format(solve_time_bisection))

################################################################################
### Compare CVX and bisection algorithm
################################################################################
euclidean_distance = np.linalg.norm(x_opt_bisection - x_opt_cvx)
solve_time_ratio = solve_time_cvx / solve_time_bisection
print('CVX objective value: {}'.format(obj_val_cvx))
print('Bisection objective value: {}'.format(obj_val_bisection))
print('CVX solve time: {}'.format(solve_time_cvx))
print('Bisection solve time: {}'.format(solve_time_bisection))
print('Euclidean distance between CVX solution and bisection solution: {}'.format(euclidean_distance))
print('Ratio of cxv solve time to bisection solve time: {}'.format(solve_time_ratio))
