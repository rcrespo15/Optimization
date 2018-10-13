import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linprog

def print_discrete_prob_distribution(p, color="blue"):
    plt.bar(range(l), p, 1, color=color, alpha=1)
    plt.ylim(0, 0.5)
    plt.show()

l = 10

mu = np.array([0, 0, 0, 0, 4, 5, 8, 10, 13, 10])
nu = np.array([14, 15, 16, 10, 4, 1, 0, 0, 0, 0])
mu = mu / np.sum(mu)
nu = nu / np.sum(nu)

###############################################################################
# Visualize probability distributions μμ and  ν
###############################################################################
print("mu:")
print_discrete_prob_distribution(mu, color="blue")

print("nu:")
print_discrete_prob_distribution(nu, color="green")

###############################################################################
# Euclidean barycenter between μμ and  ν
###############################################################################
ts = [0, 0.25, 0.5, 0.75, 1]

for t in ts:
    print("barycenter: t=" + str(t))
    print_discrete_prob_distribution(t * mu + (1-t) * nu, color="green")

###############################################################################
# Compute Wasserstein barycenter between μμ and  ν
###############################################################################
C = [[0 for _ in range(l)] for _ in range(l)]

for i in range(l):
    for j in range(l):
        C[i][j] = abs(range(l)[i] - range(l)[j])**2

ts = [0, 0.25, 0.5, 0.75, 1]
A_r = np.zeros([l,l,l])
A_d = np.zeros([l,l,l])
for i in range(l):
    A_r[i,i,:] = np.ones(l)
    A_d[i,:,i] = np.ones(l)
a_r = np.array(A_r).reshape((l,l**2))
a_d = np.array(A_d).reshape((l,l**2))
#
c = np.array(C).reshape((l**2))
from cvxpy import *
for t in ts:
    P_1 = Variable(100)
    P_2 = Variable(100)
    x = Variable(10)
    objective = Minimize(t*(c.T*P_1) + (1-t)*(c.T*P_2))
    #dummy constraint
    constraints = [x[0] <= 10000 ]
    for i in range(l):
        constraints += [ a_r[i,:].T*P_1 == mu[i]]
        constraints += [ a_d[i,:].T*P_1 == x[i]]
        constraints += [ a_d[i,:].T*P_2 == nu[i]]
        constraints += [ a_r[i,:].T*P_2 == x[i]]

    for i in range(l**2):
        constraints += [P_1[i] >= 0 ]
        constraints += [P_2[i] >= 0 ]
    emd = Problem(objective, constraints)
    emd.solve()

        # Uncomment this line to visualize barycenter results.
    print_discrete_prob_distribution(x.value, color="green")
