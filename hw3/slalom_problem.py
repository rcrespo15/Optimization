from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import pdb
from cvxpy import *
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.linalg import sqrtm

data = np.array([[0,4,0],[4,5,3],[8,4,2],[12,6,2],[16,5,1],[20,7,2],[24,4,0]])

y = Variable(7)
objective = Minimize(   square(y[0]-y[1])+square(y[1]-y[2])+ square(y[2]-y[3]) +
                        square(y[3]-y[4])+square(y[4]-y[5])+square(y[5]-y[6]))
constraints = [y[0] == 4]
constraints += [y[6] == 4]
for i in range(len(data)):
    constraints += [y[i] <= data[i,1] + .5* data[i,2]]
    constraints += [y[i] >= data[i,1] - .5* data[i,2]]

emd = Problem(objective, constraints)
emd.solve()

z = y.value
x = np.zeros([len(data),2])
for i in range(len(data)):
    x[i,0] = data[i,0]
    x[i,1] = z[i]

np.linalg.norm(x)
