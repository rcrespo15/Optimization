import numpy as np
from scipy.io import loadmat
from operator import itemgetter
import matplotlib.pyplot as plt
from cvxpy import *

mat = loadmat('cost_vec_data2.mat')
A, b, c_hat, rho, deltaC = itemgetter('A', 'b', 'c', 'rho', 'deltaC')(mat)
b, c_hat, rho = b[0], c_hat[0], rho[0][0]

##############
### Part 1 ###
##############

def solve_nominal(A, b, c_hat):
    """
    Solves the nominal optimization problem which ignores uncertainty in c.

    Returns:
        x_nom: The nominal solution
        cost_nom: The cost of the nominal solution
    """
    x = Variable(4)
    objective = Minimize(c_hat*x)
    constraints = [x[0] >= 0]
    constraints += [x[1] >= 0]
    constraints += [x[2] >= 0]
    constraints += [x[3] >= 0]
    constraints += [A[0,:] * x <= b[0]]
    constraints += [A[1,:] * x <= b[1]]
    constraints += [A[2,:] * x <= b[2]]
    constraints += [A[3,:] * x <= b[3]]
    constraints += [A[4,:] * x <= b[4]]

    emd = Problem(objective, constraints)
    emd.solve()
    cost_nom = emd.value
    x_nom = x.value

    return x_nom, cost_nom

def worst_cost(x, c_hat, rho, deltaC):
    """
    Computes the worst case cost of a decision x.

    Returns:
        The worst case cost
    """
    big_C = c_hat + rho*deltaC
    cost_worst = np.max(np.sum(big_C*x.T,axis =1))

    return cost_worst

x_nom, cost_nom = solve_nominal(A, b, c_hat)
print('x_nom:\n{}'.format(x_nom))
print('profit_nom: {}'.format(-1. * cost_nom))
print('worst_profit: {}'.format(-1. * worst_cost(x_nom, c_hat, rho, deltaC)))

##############
### Part 2 ###
##############

def solve_robust(A, b, c_hat, rho, deltaC):
    """
    Solves the robust optimization problem which considers uncertainty in c.

    Returns:
        x_rob: The robust solution
        cost_rob: The cost of the robust solution
    """
    big_C = c_hat + rho*deltaC
    n_c,m_c = big_C.shape
    n_a,m_a = A.shape
    x = Variable(4)
    alpha = Variable(1)
    objective = Minimize(alpha)
    constraints = [x[0] >= 0]
    constraints = [alpha >= big_C[0,:]*x]
    for i in range(n_c):
        constraints += [alpha >= big_C[i,:]*x]
    for i in range(m_a):
        constraints += [x[i] >= 0]
    for i in range(n_a):
        constraints += [A[i,:] * x <= b[i]]

    emd = Problem(objective, constraints)
    emd.solve()
    cost_rob = emd.value
    x_rob = x.value
    return x_rob, cost_rob

x_rob, cost_rob = solve_robust(A, b, c_hat, rho, deltaC)
print('x_rob:\n{}'.format(x_rob))
print('profit_rob: {}'.format(-1. * cost_rob))
print('worst_profit: {}'.format(-1. * worst_cost(x_rob, c_hat, rho, deltaC)))

##############
### Part 3 ###
##############

rhos = np.linspace(0.03, 0.09, 200)
profits = []
for i in range(len(rhos)):
    _,value = solve_robust(A, b, c_hat, rhos[i], deltaC)
    profits.append(value)
plt.figure()
plt.title('Profit vs. rho')
plt.plot(rhos, profits)
plt.xlim(0.03, 0.09)
plt.ylim(-1000, 6000)
plt.show()
plt.close()
