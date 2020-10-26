import numpy as np
from algorithms.gradient_descent import gradient_descent
from algorithms.accelerated_gradient_descent import accelerated_gradient_descent
from algorithms.accelerated_gradient_descent_adaptive_restart import (
    accelerated_gradient_descent_adaptive_restart,
)
from algorithms.gradient_descent_line_search import gradient_descent_line_search
from algorithms.accelerated_gradient_descent_line_search import (
    accelerated_gradient_descent_line_search,
)
from algorithms.accelerated_gradient_descent_adaptive_restart_line_search import (
    accelerated_gradient_descent_adaptive_restart_line_search,
)
from algorithms.conjugate_gradient import conjugate_gradient
from algorithms.power_iteration import power_iteration

from utils.plot_results import plot_results
from utils.list2matrix import list2matrix

datasetname = "Wiki-Vote.txt"  # Change ??? appropriately to load data.
# datasetname = 'Slashdot0811.txt'  # Change ??? appropriately to load data
# datasetname = 'web-Stanford.txt'  # Change ??? appropriately to load data
# datasetname = 'web-Google.txt'  # Change ??? appropriately to load data

name = datasetname[0:-4]
datasetname = "../data/" + datasetname
E = list2matrix(
    datasetname
)  # I normalize the columns after reading the edge list
n = E.shape[0]
p = 0.15
# Damping factor.
Mx = (
    lambda x: (1.0 - p) * E.dot(x) + p * np.sum(x) / n
)  # An efficient way of implementing M*x
MminITx = lambda x: (
    ((1.0 - p) * E.T.dot(x)) + p * np.sum(x) / n - x
)  # An efficient way of implementing (M-I)'*x  # where M is PageRank matrix
MminIx = lambda x: (
    ((1.0 - p) * E.dot(x)) + p * np.sum(x) / n - x
)  # An efficient way of implementing (M-I)*x

penaltyparameter = 1.0  # You can vary penalty parameter
sigma = 1e-5  # Penalty parameter on the  l2 norm of the parameter

# Evaluate the Lipschitz constant and strong convexity parameter.
parameter = {}
parameter["Lips"] = 4.0 + n * penaltyparameter + sigma
parameter["strcnvx"] = 1 + sigma

# Set parameters and solve numerically with GD, AGD, AGDR, LSGD, LSAGD, LSAGDR.
print("Numerical solution process is started: \n")
fx = (
    lambda x: (
        0.5 * np.linalg.norm(Mx(x) - x) ** 2
        + 0.5 * penaltyparameter * (np.sum(x) - 1) ** 2
    )
    + 0.5 * sigma * np.linalg.norm(x) ** 2
)
gradf = lambda x: (MminITx(Mx(x) - x) + penaltyparameter * (np.sum(x) - 1)) + sigma * x
parameter["x0"] = np.ones(n) * 1.0 / n
parameter["tolx"] = 1e-8  # You can vary tolx and maxit
parameter["maxit"] = 4e3  # to achieve the convergence.
x = {}
info = {}
x["GD"], info["GD"] = gradient_descent(fx, gradf, parameter, verbose=1)
x["AGD"], info["AGD"] = accelerated_gradient_descent(fx, gradf, parameter, verbose=1)
parameter["kappa"] = 1.0
x["LSGD"], info["LSGD"] = gradient_descent_line_search(fx, gradf, parameter, verbose=1)
x["LSAGD"], info["LSAGD"] = accelerated_gradient_descent_line_search(
    fx, gradf, parameter, verbose=1
)
x["AGDR"], info["AGDR"] = accelerated_gradient_descent_adaptive_restart(
    fx, gradf, parameter, verbose=1
)
x["LSAGDR"], info["LSAGDR"] = accelerated_gradient_descent_adaptive_restart_line_search(
    fx, gradf, parameter, verbose=1
)

# Solve numerically with CG.
Phix = (
    lambda x: (MminITx(Mx(x) - x) + penaltyparameter * (np.sum(x))) + sigma * x
)  # Implements Phi_sigma * x for CG method
y = penaltyparameter  # vector y for CG algorithm
x["CG"], info["CG"] = conjugate_gradient(fx, Phix, y, parameter)
# Solve numerically with PageRank algorithm (Power method).
x["PR"], info["PR"] = power_iteration(fx, Mx, parameter)

print("Numerical solution process is completed. \n")
options = {"dir": "../figs", "name": "Pagerank-" + name}
plot_results(x, info, options, fmin=0)
