"""
**************************************************************************
*********Python implementation of ridgeregression.m **********************
**************************************************************************
"""
import numpy as np
import scipy

from gradient_descent import gradient_descent
from accelerated_gradient_descent import accelerated_gradient_descent
from accelerated_gradient_descent_adaptive_restart import (
    accelerated_gradient_descent_adaptive_restart,
)
from gradient_descent_line_search import gradient_descent_line_search
from accelerated_gradient_descent_line_search import (
    accelerated_gradient_descent_line_search,
)
from accelerated_gradient_descent_adaptive_restart_line_search import (
    accelerated_gradient_descent_adaptive_restart_line_search,
)
from conjugate_gradient import conjugate_gradient
import plotresults as plotresults

# Parameters for synthetic data.
cfg = {}
cfg["n"] = int(1e3)
# number of features
cfg["p"] = int(1e3)
# number of dimensions
cfg["noisestd"] = 1e-6
# standard deviation of additive iid gaussian noise (0 for noiseless)
cfg["strcnvx"] = False
# false = not strongly convex
# true  = strongly convex with, lambda = 0.01*norm(A'*A)

# Methods to be checked.
chk = {
    "GD": True,
    "AGD": True,
    "AGDR": True,
    "LSGD": True,
    "LSAGD": True,
    "LSAGDR": True,
    "CG": True,
}
# Generate synthetic data.
A = np.random.random((cfg["n"], cfg["p"]))
# Generate s-sparse vector.
xtrue = np.random.randn(cfg["p"])
# Take (noisy) samples.
noise = cfg["noisestd"] * np.random.randn(cfg["n"])
b = np.dot(A, xtrue) + noise


# Strongly convex OR Convex?
if cfg["strcnvx"]:
    cfg["lambda"] = 0.01 * np.linalg.norm(A)
else:
    cfg["lambda"] = 0.0

# Evaluate the Lipschitz constant and strong convexity parameter.
parameter = {}
parameter["Lips"] = np.linalg.norm(np.dot(A.T, A) + cfg["lambda"] * np.eye(cfg["p"]))
parameter["strcnvx"] = cfg["lambda"]

# Set parameters and solve numerically.
print("Numerical solution process is started: \n")
fx = lambda x: (
    0.5 * np.linalg.norm((np.dot(A, x) - b)) ** 2
    + 0.5 * cfg["lambda"] * np.linalg.norm(x) ** 2
)
gradf = lambda x: (np.dot(A.T, np.dot(A, x) - b) + cfg["lambda"] * x)
phi = lambda x: (np.dot(A.T, np.dot(A, x)) + cfg["lambda"] * x)
y = np.dot(A.T, b)
parameter["x0"] = np.zeros((cfg["p"]))
parameter["tolx"] = 1e-5  # You can vary tolx and maxit
parameter["maxit"] = 4e2  # to achieve the convergence.


x = {}
info = {}
if chk["GD"]:
    x["GD"], info["GD"] = gradient_descent(fx, gradf, parameter, verbose=1)
if chk["AGD"]:
    x["AGD"], info["AGD"] = accelerated_gradient_descent(
        fx, gradf, parameter, verbose=1
    )
if chk["AGDR"]:
    x["AGDR"], info["AGDR"] = accelerated_gradient_descent_adaptive_restart(
        fx, gradf, parameter, verbose=1
    )
if chk["LSGD"]:
    parameter["kappa"] = 1.0
    x["LSGD"], info["LSGD"] = gradient_descent_line_search(
        fx, gradf, parameter, verbose=1
    )
if chk["LSAGD"]:
    x["LSAGD"], info["LSAGD"] = accelerated_gradient_descent_line_search(
        fx, gradf, parameter, verbose=1
    )
if chk["LSAGDR"]:
    x["LSAGDR"], info[
        "LSAGDR"
    ] = accelerated_gradient_descent_adaptive_restart_line_search(
        fx, gradf, parameter, verbose=1
    )
if chk["CG"]:
    x["CG"], info["CG"] = conjugate_gradient(fx, phi, y, parameter, verbose=1)

print("Numerical solution process is completed. \n")
# Find x^* and f^* if noisy to plot data.
fmin = 0.0
if cfg["noisestd"] != 0 and cfg["n"] >= cfg["p"]:
    xmin = np.dot(
        np.linalg.pinv(np.dot(A.T, A) + cfg["lambda"] * np.eye(cfg["p"])),
        np.dot(A.T, b),
    )
    fmin = fx(xmin)
# Plot the results.
options = {"name": "RidgeRegression"}
plotresults.plotresults(x, info, options, fmin=0)
