from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import pdb
from cvxpy import *
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.linalg import sqrtm

data = np.array([   [1,2047.6,2028],
                    [2,2046.0,2008],
                    [3,2002.6,2035],
                    [4,2036.9,0],
                    [5,2013.5,2019],
                    [6,2021.1,0],
                    [7,2027.4,0],
                    [8,2047.1,0],
                    [9,2020.9,2044],
                    [10,2049.9,2044],
                    [11,2015.1,0],
                    [12,2035.1,0],
                    [13,2033.3,0],
                    [14,2027.0,2043],
                    [15,2034.9,0],
                    [16,2033.3,0],
                    [17,2008.9,0],
                    [18,2006.4,0],
                    [19,2050.0,2030],
                    [20,2008.6,2025],
                    [21,2001.6,0],
                    [22,2028.1,2045]])
x = Variable(22)
constraints = [x[4]+x[0]-x[5]+x[9] == 0]
constraints += [x[5]-x[1]-x[6]-x[10] == 0]
constraints += [x[6]+x[2]-x[7]+x[11] == 0]
constraints += [x[7]-x[3]-x[8]-x[12] == 0]
constraints += [-x[13]-x[9]+x[14]+x[18] == 0]
constraints += [-x[14]+x[10]+x[15]-x[19] == 0]
constraints += [-x[15]-x[11]+x[16]+x[20] == 0]
constraints += [-x[16]+x[12]+x[17]-x[21] == 0]
constraints += [x[0] == 2028]
constraints += [x[1] == 2008]
constraints += [x[2] == 2035]
constraints += [x[4] == 2019]
constraints += [x[8] == 2044]
constraints += [x[13] == 2043]
constraints += [x[18] == 2030]
constraints += [x[19] == 2025]
constraints += [x[21] == 2045]

objective = Minimize(pnorm(x-data[:,1],2))

emd = Problem(objective, constraints)
emd.solve()
