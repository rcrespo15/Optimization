import numpy as np
import scipy.io as sio
from scipy.stats import ortho_group
import matplotlib.pyplot as plt
import math
import scipy.io as sio
import numpy as np
import time
import timeit

###########################
### EXACT LOW RANK DATA ###
###########################
# This is the template code to get you started. You should fill out the code
# for 2 places:
# 1 -> direct_solver
# 2 -> low_rank_solver
#
# Feel help to write any additional functions and helper code. Note that the data
#  matrix is a 1000 by 1000 matrix with rank 10

def rank_decompose(A, r):
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    return U[:,:r], s[:r], Vh[:r, :]

data = sio.loadmat('low_rank_mtx.mat')
A = data['predictors']
y = data['responses'].T
lmbda = data['penalty_param']

m, n = A.shape

U, s, Vh = rank_decompose(A, np.linalg.matrix_rank(A))
L = U * s
R = Vh.T

def direct_solver(A, y, lmbda):
    '''
    Takes in the data matrix A and vector y, scalar lmbda
    and returns the minimizer for the regularized LS problem
    '''
    m,n = A.shape
    identity = np.eye(m,n)
    x = np.dot(np.linalg.inv(np.dot(A.T,A) + identity*lmbda),np.dot(A.T,y))
    return x

def low_rank_solver(L, R, y, lmbda):
    '''
    Takes in the matrices L and R, with LR^T = A, scalar lmbda,
    and vector y, and returns the minimizer for the low rank
    regularized LS problem
    '''
    # your implementation here
    lrr = np.dot(L,np.dot(R.T,R))
    n,m = R.shape
    identity = np.eye(m,m) * lmbda
    v = np.dot(np.linalg.inv(np.dot(lrr.T,lrr)+ np.dot(identity,np.dot(R.T,R))),np.dot(lrr.T,y))
    x = np.dot(R,v)
    return x

# compare their runtime
print('Direct solver:')
start = timeit.default_timer()
soln_direct = direct_solver(A, y, lmbda)
stop = timeit.default_timer()
print('Time for direct solver: ', stop - start)

print('\nLow rank solver')
start = timeit.default_timer()
soln_low_rank = low_rank_solver(L, R, y, lmbda)
stop = timeit.default_timer()
print('Time for low rank solver: ', stop - start)

# check their solutions to be the same (less than 1e-5 is great !)
print(np.median(np.abs(soln_direct - soln_low_rank) / np.abs(soln_direct)))
#

#################################
### APPROXIMATE LOW RANK DATA ###
#################################

# load in the prepared data
data = sio.loadmat('low_rank_mtx.mat')
A_approx = data['approx_mtx']
n = A_approx.shape[0]
y_appox = data['approx_responses'].T
lmbda = data['penalty_param'][0][0]

def error_fn(A, x, y, lmbda):
    return np.linalg.norm(A @ x - y)**2 + lmbda * np.linalg.norm(x)**2

def rank_decompose(A, r):
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    return U[:,:r], s[:r], Vh[:r, :]

def rank_decompose_LR(A, r):
    U, s, Vh = rank_decompose(A, r)
    return U * s, Vh

def solve_with_rank_r(A, y, lmbda, r):

    # decompose A
    L, RT = rank_decompose_LR(A, r)
    R = RT.T

    #solve using reduced lr method
    soln = low_rank_solver(L, R, y, lmbda)

    # compute error
    error = error_fn(A, soln, y, lmbda)

    return error

rank_lst = [int(i) for i in np.linspace(1, n)]
error_lst = [solve_with_rank_r(A_approx, y_appox, lmbda, r) for r in rank_lst]
plt.plot(rank_lst, error_lst)
plt.xlabel("rank used for approximation")
plt.ylabel("residual error")
plt.show()
