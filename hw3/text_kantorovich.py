from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import pdb
from cvxpy import *

document_1 = ["media", "Illinois", "speaks", "Obama"]
document_2 = ["President", "greets", "press", "Chicago"]

#Load word embedding dictionary
word_embedding = np.load("word_embedding.npy").item()

def distance(word1, word2):
    return np.linalg.norm(word_embedding[word1] - word_embedding[word2])

# TODO: Construct C matrix where C_ij = dist(document_1[i], document_2[j]).
d = len(document_2)
n = len(document_1)
C = np.zeros([n,d])

for i in range(len(document_2)):
    for j in range(len(document_1)):
        C[i][j] = distance(document_1[i], document_2[j])

#Compute the transportation plan as well as the transportation cost (aka EMD)

# We assume that the length of document one and two are equal.
l = len(document_1)

# Compute normalized frequency vectors for sentence one and two.
mu = [1. / l for _ in range(l)]
nu = [1. / l for _ in range(l)]
#
# To formulate and solve the LP, the C and P matrices need to reshaped
# to vectors of length lxl
c = np.array(C).reshape((l**2))
ones = np.ones(l)
# Construct matrices of ones, A_r and A_t, which when multiplied by P
# reshaped to lxl vector gives us the equality contraints.
# Where row i of A_r equals sum of entries of P_i and row i of A_t
# equals sum of entries of row i of (P^T).
A_r = np.zeros((l, l, l))
A_t = np.zeros((l, l, l))

for i in range(l):
    A_r[i,i,:] = np.ones(4)
    A_t[i,:,i] = np.ones(4)
# # TODO: Solve LP with objective C^Tx, constraints Ax = b.
P = Variable(16)

objective = Minimize(c.T*P)

constraints = [ P[0] + P[1] + P[2] + P[3] == mu[0] ]
constraints += [ P[4] + P[5] + P[6] + P[7] == mu[1] ]
constraints += [ P[8] + P[9] + P[10] + P[11] == mu[2] ]
constraints += [ P[12] + P[13] + P[14] + P[15] == mu[3]]
constraints += [ P[0] + P[4] + P[8] + P[12] == nu[0]]
constraints += [ P[1] + P[5] + P[9] + P[13] == nu[1]]
constraints += [ P[2] + P[6] + P[10] + P[14] == nu[2]]
constraints += [ P[3] + P[7] + P[11] + P[15] == nu[3]]
for i in range(16):
    constraints += [P[i] >=0]
emd = Problem(objective, constraints)
emd.solve()

P1 = P.value
P1 = P1.reshape((4,4))
print("EMD: " + str(emd))

print("Visualize P transportation plan: ")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.matshow(P1)
ax.set_xticks(np.arange(len(document_1)))
ax.set_yticks(np.arange(len(document_2)))
ax.set_xticklabels(document_2)
ax.set_yticklabels(document_1)
plt.show()

###############################################################################
#Question 4
###############################################################################
a = word_embedding["media"]
# fname = ("text1.txt","text2.txt","text3.txt","text4.txt")
# V = ("aid", "kill", "deal", "president", "tax", "china")
# count = np.zeros([4,6])
# for t in range(len(fname)):
#     with open(fname[t], 'r') as f:
#         for line in f:
#             words = line.split()
#             for word in words:
#                 for i in range(len(V)):
#                     if V[i] == word:
#                         count[t,i] += 1
# x,_ = count.shape
# for i in range(x):
#     count[i,:] = np.true_divide(count[i,:],np.sum(count[i,:]))
#
# print (count)
#
#
# for i in range(x):
#     t1 = (np.sum(np.multiply(count[2,:],count[i,:]))/
#             (LA.norm(count[2,:])*LA.norm(count[i,:])))
#     print (t1)
