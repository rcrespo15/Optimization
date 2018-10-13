from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import pdb
from cvxpy import *
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.linalg import sqrtm
#Given information

r = np.array([.12,.1,.07,.03])
sigma = np.array([[.0064,.0008,-.0011,0],[.0008,.0025,0,0],[-.0011,0,.0004,0],[0,0,0,0]])
sigma_sqrt = sqrtm(sigma)
upper_limit = .4
lower_limit = .05
risk_free_limit = .2
q = -.03
e = 1e-4
fs = 12
def optimal_portfolio(epsilon):
    # Optimization problem
    #Optimization variable
    x = Variable(4)
    #Objective function maximize r*x
    objective = Minimize(-r*x)

    #Constraints
    constraints = [x[3]<=.2]
    for i in range(len(r)):
        constraints += [x[i] <= .40]
        constraints += [x[i] >= .05]
    constraints += [sum(x) ==1]
    constraints += [pnorm((sigma_sqrt@x),2) <= (1/norm.ppf(epsilon))*(q-r.T*x)]

    emd = Problem(objective, constraints)
    return(emd.solve(),x.value)

es = np.linspace(1e-6,1e-3,100)
value = np.zeros(len(es))
convination = np.zeros([len(es),len(r)])

for i in range(len(es)):
    value[i],convination[i,:] = optimal_portfolio(es[i])

##############################################################################
#                             Part 2
##############################################################################
# #Figure 2.a
# plt.plot(es,1-value,label="Optimal Return vs Risk")
# plt.xlabel('epsilon', fontsize = (fs))
# plt.ylabel('Optimal Return', fontsize = (fs))
# name_for_graphs = ('Question 4c: Optimal Return vs Risk')
# plt.title(name_for_graphs, fontsize= (fs))
# plt.show()
#
# #Figure 2.b
# x = es
# fig = plt.subplots()
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
#
# ax1.plot(x, convination[:,0], lw=1, label='stock a', color='r')
# ax1.plot(x, convination[:,1], lw=1, label='stock b', color='b')
# ax1.plot(x, convination[:,2], lw=1, label='stock c', color='green')
# ax1.plot(x, convination[:,3], lw=1, label='stock d', color='purple')
# ax1.fill_between(x, 0, convination[:,0], facecolor='r', alpha=0.2)
# ax1.fill_between(x, convination[:,0], convination[:,1], facecolor="b", alpha=.2)
# ax1.fill_between(x, convination[:,1], convination[:,2], facecolor="green", alpha=.2)
# ax1.fill_between(x, convination[:,2], convination[:,3], facecolor="purple", alpha=.2)
# ax1.legend(loc='upper right')
# ax1.set_xlabel('risk value')
# ax1.set_ylabel('percentage of portfolio')
# plt.show()

##############################################################################
#                             Part 3
##############################################################################
samples = 1000
value_1,convination_1 = optimal_portfolio(es[i])
random_samples = np.random.multivariate_normal(r,sigma,samples)
earnings_1 = np.dot(convination_1,random_samples.T)
mean_earnings_1 = np.mean(earnings_1)
std_earnings_1 = np.std(earnings_1)
loss_earnings_percentage = (np.sum(earnings_1 < 0)/samples)

plt.hist(1+earnings_1,normed=True)      #use this to draw histogram of your data
plt.xlabel('Return', fontsize = (fs))
plt.ylabel('Frequency', fontsize = (fs))
plt.show()
