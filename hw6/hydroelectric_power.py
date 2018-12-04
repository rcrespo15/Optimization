################################
#Hydroelectric Power Generatino#
################################

import numpy as np
from cvxpy import *

##############
### Part 1 ###
##############


K = 3
L = 2
J = 3
w_l = np.array([20, 15])
lam = np.array([10, 30, 8])
rho = np.array([30./47, 43./58, 50./72])
V_0 = np.array([120, 130])
V_min = np.zeros((L, K))
V_max = np.array([[95000, 95000, 95000], [11000, 11000, 11000]])
A = np.array([[15, 15, 15], [12, 12, 12]])
T_max = np.array([[55, 55, 55], [65, 65, 65], [80, 80, 80]])
T_min = np.zeros((J, K))

def solve(K, L, J, w_l, lam, rho, V_0, V_min, V_max, A, T_min, T_max):
    """Solve the optimization problem described in part 1 using a solver of your choice.

    Args:
        K (int): number of timestesp
        L (int): number of reservoirs
        J (int): number of turbines
        etc.

    Returns:
        the optimal water volumes, T_opt
        the optimal objective value, obj_val
    """
    # TODO: your implementation here

    V_f = Variable(2)
    Tjk = Variable((3,3))
    Vlk = Variable((3,2))


    objective = Minimize(sum(w_l.T*(V_0.T-V_f)) - sum(lam * (rho*Tjk.T).T)) #np.dot(lam,np.transpose(np.dot(rho,np.transpose(Tjk)))))
    constraints = [Vlk[0,0] == V_0[0] + A[0,0] - Tjk[0,0]-Tjk[0,1]]
    constraints += [Vlk[0,1] == V_0[1] + A[1,0] + Tjk[0,0] + Tjk[0,1] - Tjk[0,2]]

    constraints += [Vlk[1,0] == Vlk[0,0] + A[0,1] - Tjk[1,0] - Tjk[1,1]]
    constraints += [Vlk[1,1] == Vlk[0,1] + A[1,1] + Tjk[1,0] + Tjk[1,1] - Tjk[1,2]]

    constraints += [Vlk[2,0] == Vlk[1,0] + A[0,2] - Tjk[2,0] - Tjk[2,1]]
    constraints += [Vlk[2,1] == Vlk[1,1] + A[1,2] + Tjk[2,0] + Tjk[2,1] - Tjk[2,2]]

    constraints += [Vlk[2,0] == V_f[0]]
    constraints += [Vlk[2,1] == V_f[1]]

    for i in range(3):
        constraints += [Vlk[i,0] <= V_max[0,i]]
        constraints += [Vlk[i,0] >= V_min[0,i]]
        constraints += [Vlk[i,1] <= V_max[1,i]]
        constraints += [Vlk[i,1] >= V_min[1,i]]

    for i in range(3):
        constraints += [Tjk[i,0] <= T_max[0,i]]
        constraints += [Tjk[i,1] <= T_max[1,i]]
        constraints += [Tjk[i,2] <= T_max[2,i]]
        constraints += [Tjk[i,0] >= 0]
        constraints += [Tjk[i,1] >= 0]
        constraints += [Tjk[i,2] >= 0]


    emd = Problem(objective, constraints)
    emd.solve()

    T_opt = Tjk.value
    obj_val = emd.value
    return (T_opt, obj_val)

# Solve the optimization problem
T_opt, obj_val = solve(K, L, J, w_l, lam, rho, V_0, V_min, V_max, A, T_min, T_max)

# Round values for readability (optional)
obj_val = np.around(obj_val, decimals=7)
T_opt = np.around(T_opt, decimals=0)

# Print results
print("Optimal value:\n{}\n".format(obj_val))
print("Optimal water volumes, T_opt:\n{}".format(T_opt))

##############
### Part 2 ###
##############

K = 3
L = 2
J = 3
w_l = np.array([20, 15])
lam = np.array([12, 30, 4])
rho = np.array([30./47, 43./58, 50./72])
V_0 = np.array([120, 130])
V_min = np.zeros((L, K))
V_max = np.array([[95000, 95000, 95000], [11000, 11000, 11000]])
A = np.array([[15, 15, 15], [12, 12, 12]])
T_min = np.zeros((J, K))
T_max = np.array([[55, 55, 55], [65, 65, 65], [80, 80, 80]])

def solve_regularized(K, L, J, T_hat, w_l, lam, rho, V_0, V_min, V_max, A, T_min, T_max, gamma):
    """Solve the optimization problem described in part 2 using a solver of your choice.

    Args:
        T_hat (numpy.ndarray): optimal solution T_opt from part 1
        gamma (float): regularization multiplier
        (remaining variables have same description as in part 1)

    Returns:
        the optimal water volumes, T_opt_reg
        the optimal objective value, obj_val_reg
    """
    # TODO: your implementation here
    V_f = Variable(2)
    Tjk = Variable((3,3))
    Vlk = Variable((3,2))
    z = Variable(1)
    q1 = Variable(1)
    q2 = Variable(1)
    q3 = Variable(1)


    objective = Minimize(sum(w_l.T*(V_0.T-V_f)) - sum(lam * (rho*Tjk.T).T) + gamma*z) #np.dot(lam,np.transpose(np.dot(rho,np.transpose(Tjk)))))
    constraints = [z == q1 + q2 + q3]
    constraints += [q1 >= abs(Tjk[0,0]-T_hat[0,0])]
    constraints += [q1 >= abs(Tjk[0,1]-T_hat[0,1])]
    constraints += [q1 >= abs(Tjk[0,2]-T_hat[0,2])]

    constraints += [q2 >= abs(Tjk[1,0]-T_hat[1,0])]
    constraints += [q2 >= abs(Tjk[1,1]-T_hat[1,1])]
    constraints += [q2 >= abs(Tjk[1,2]-T_hat[1,2])]

    constraints += [q3 >= abs(Tjk[2,0]-T_hat[2,0])]
    constraints += [q3 >= abs(Tjk[2,1]-T_hat[2,1])]
    constraints += [q3 >= abs(Tjk[2,2]-T_hat[2,2])]

    constraints += [Vlk[0,0] == V_0[0] + A[0,0] - Tjk[0,0]-Tjk[0,1]]
    constraints += [Vlk[0,1] == V_0[1] + A[1,0] + Tjk[0,0] + Tjk[0,1] - Tjk[0,2]]

    constraints += [Vlk[1,0] == Vlk[0,0] + A[0,1] - Tjk[1,0] - Tjk[1,1]]
    constraints += [Vlk[1,1] == Vlk[0,1] + A[1,1] + Tjk[1,0] + Tjk[1,1] - Tjk[1,2]]

    constraints += [Vlk[2,0] == Vlk[1,0] + A[0,2] - Tjk[2,0] - Tjk[2,1]]
    constraints += [Vlk[2,1] == Vlk[1,1] + A[1,2] + Tjk[2,0] + Tjk[2,1] - Tjk[2,2]]

    constraints += [Vlk[2,0] == V_f[0]]
    constraints += [Vlk[2,1] == V_f[1]]

    for i in range(3):
        constraints += [Vlk[i,0] <= V_max[0,i]]
        constraints += [Vlk[i,0] >= V_min[0,i]]
        constraints += [Vlk[i,1] <= V_max[1,i]]
        constraints += [Vlk[i,1] >= V_min[1,i]]

    for i in range(3):
        constraints += [Tjk[i,0] <= T_max[0,i]]
        constraints += [Tjk[i,1] <= T_max[1,i]]
        constraints += [Tjk[i,2] <= T_max[2,i]]
        constraints += [Tjk[i,0] >= 0]
        constraints += [Tjk[i,1] >= 0]
        constraints += [Tjk[i,2] >= 0]


    emd = Problem(objective, constraints)
    emd.solve()

    T_opt_reg = Tjk.value
    obj_val_reg = emd.value
    return T_opt_reg, obj_val_reg

# Solve the optimization problem
T_opt_reg, obj_val_reg = solve_regularized(K, L, J, T_opt, w_l, lam, rho, V_0, V_min, V_max, A, T_min, T_max, gamma=np.power(10, 10, dtype=np.float32))

# Round values for readability (optional)
obj_val_reg = np.around(obj_val_reg, decimals=7)
T_opt_reg = np.around(T_opt_reg, decimals=0)

# Print results
print("Part3")
print("Optimal value:\n{}\n".format(obj_val_reg))
print("Optimal water volumes, T_opt_reg:\n{}".format(T_opt_reg))
