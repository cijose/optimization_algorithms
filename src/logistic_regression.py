import numpy as np
import scipy
from sklearn.datasets import load_svmlight_file

from algorithms.accelerated_gradient_descent import \
    accelerated_gradient_descent
from algorithms.accelerated_gradient_descent_adaptive_restart import \
    accelerated_gradient_descent_adaptive_restart
from algorithms.accelerated_gradient_descent_adaptive_restart_line_search import \
    accelerated_gradient_descent_adaptive_restart_line_search
from algorithms.accelerated_gradient_descent_line_search import \
    accelerated_gradient_descent_line_search
from algorithms.conjugate_gradient import conjugate_gradient
from algorithms.gradient_descent import gradient_descent
from algorithms.gradient_descent_line_search import \
    gradient_descent_line_search
from algorithms.newtons_method import newtons_method
from algorithms.quasi_newton import quasi_newton
from utils.plot_results import plot_results

cfg = {}
dataset_name = "w5a"
"""
A,b = svmutils.load_svmlight_file('./data/'+dataset_name+'.tr')
cfg['n'],cfg['p'] = A.shape
b= np.asarray(b)
b[b>1] = -1.
perm = np.random.permutation(cfg['n'])
A = A[perm, :]
b = b[perm]
ntr = round(cfg['n']*3./4.)
Atr = A[0:ntr, :]
btr = b[0:ntr]
Ate = A[ntr:-1, :]
bte = b[ntr:-1]
"""
data_dir = "../data/"
Atr, btr = load_svmlight_file(data_dir + dataset_name + ".tr")
# btr[btr>1] = -1.
cfg["n"], cfg["p"] = Atr.shape
Ate, bte = load_svmlight_file(data_dir + dataset_name + ".te")
sigma = cfg["n"] * 1e-2
sigmoid = lambda x: 1.0 / (1 + np.exp(-btr * (Atr.dot(x[0:-1]) + x[-1])))
fx = lambda x: (
    np.sum(-np.log(sigmoid(x))) + 0.5 * sigma * np.linalg.norm(x[0:-1]) ** 2
)


if dataset_name == "mnist":
    btr[btr < 5] = -1
    btr[btr != -1] = 1
    bte[bte < 5] = -1
    bte[bte != -1] = 1
    Atr = Atr[:, :775]
    Ate = Ate[:, :775]
    cfg["n"], cfg["p"] = Atr.shape


def gradf(x):
    y = sigmoid(x)
    gx = sigma * x[0:-1] - Atr.T.dot(btr * (1.0 - y))
    gx = np.append(gx, -np.sum(btr * (1 - y)))
    return gx


def hessf(x):
    y = sigmoid(x)
    y = y * (1 - y)
    Hr = np.zeros((len(x), len(x)))
    Hr[0:-1, 0:-1] = (
        sigma * np.eye(len(x) - 1)
        + ((Atr.T * scipy.sparse.spdiags(y, 0, len(y), len(y)) * Atr)).toarray()
    )
    Hr[-1, -1] = np.sum(y)
    y = y.T * Atr
    Hr[-1, 0:-1] = y
    Hr[0:-1, -1] = y
    return Hr


# Methods to be checked.
chk = {
    "QNM": 1,
    "NM": 1,
    "GD": 1,
    "AGD": 1,
    "AGDR": 1,
    "LSGD": 1,
    "LSAGD": 1,
    "LSAGDR": 1,
    "CG": 1,
}

# Evaluate the Lipschitz constant and strong convexity parameter.
parameter = {}
Asq = Atr
Asq.data **= 2
Asq = Asq.sum(axis=1)
parameter["Lips"] = 0.25 * np.sum((Asq)) + cfg["n"] + sigma
del Asq
parameter["strcnvx"] = sigma
# Set parameters and solve numerically.
print("Numerical solution process is started: \n")

parameter["x0"] = np.zeros((cfg["p"] + 1))
parameter[
    "tolx"
] = 1e-6  # You don't need very high precision solution to machine learnin problems
parameter["maxit"] = 40  # to achieve the convergence.
x = {}
info = {}
if chk["QNM"]:
    parameter["kappa"] = 0.1
    x["QNM"], info["QNM"] = quasi_newton(fx, gradf, parameter, verbose=1)
if chk["NM"]:
    parameter["kappa"] = 0.1
    x["NM"], info["NM"] = newtons_method(fx, gradf, hessf, parameter, verbose=1)
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
    parameter["kappa"] = 0.1
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

if chk["GD"]:
    btrpred = np.sign(Atr * x["GD"][0:-1] + x["GD"][-1])
    btepred = np.sign(Ate * x["GD"][0:-1] + x["GD"][-1])
    print("Training Accuracy GD " + str(np.sum(btrpred == btr) * 100.0 / len(btr)))
    print("Testing  Accuracy GD " + str(np.sum(btepred == bte) * 100.0 / len(bte)))
if chk["LSGD"]:
    btrpred = np.sign(Atr * x["LSGD"][0:-1] + x["LSGD"][-1])
    btepred = np.sign(Ate * x["LSGD"][0:-1] + x["LSGD"][-1])
    print("Training Accuracy LSGD " + str(np.sum(btrpred == btr) * 100.0 / len(btr)))
    print("Testing  Accuracy LSGD " + str(np.sum(btepred == bte) * 100.0 / len(bte)))
if chk["AGD"]:
    btrpred = np.sign(Atr * x["AGD"][0:-1] + x["AGD"][-1])
    btepred = np.sign(Ate * x["AGD"][0:-1] + x["AGD"][-1])
    print("Training Accuracy AGD " + str(np.sum(btrpred == btr) * 100.0 / len(btr)))
    print("Testing  Accuracy AGD " + str(np.sum(btepred == bte) * 100.0 / len(bte)))
if chk["LSGD"]:
    btrpred = np.sign(Atr * x["LSAGD"][0:-1] + x["LSAGD"][-1])
    btepred = np.sign(Ate * x["LSAGD"][0:-1] + x["LSAGD"][-1])
    print("Training Accuracy LSAGD " + str(np.sum(btrpred == btr) * 100.0 / len(btr)))
    print("Testing  Accuracy LSAGD " + str(np.sum(btepred == bte) * 100.0 / len(bte)))
if chk["AGDR"]:
    btrpred = np.sign(Atr * x["AGDR"][0:-1] + x["AGDR"][-1])
    btepred = np.sign(Ate * x["AGDR"][0:-1] + x["AGDR"][-1])
    print("Training Accuracy AGDR " + str(np.sum(btrpred == btr) * 100.0 / len(btr)))
    print("Testing  Accuracy AGDR " + str(np.sum(btepred == bte) * 100.0 / len(bte)))
if chk["LSAGDR"]:
    btrpred = np.sign(Atr * x["LSAGDR"][0:-1] + x["LSAGDR"][-1])
    btepred = np.sign(Ate * x["LSAGDR"][0:-1] + x["LSAGDR"][-1])
    print("Training Accuracy LSAGDR " + str(np.sum(btrpred == btr) * 100.0 / len(btr)))
    print("Testing  Accuracy LSAGDR " + str(np.sum(btepred == bte) * 100.0 / len(bte)))
if chk["NM"]:
    btrpred = np.sign(Atr * x["NM"][0:-1] + x["NM"][-1])
    btepred = np.sign(Ate * x["NM"][0:-1] + x["NM"][-1])
    print("Training Accuracy NM " + str(np.sum(btrpred == btr) * 100.0 / len(btr)))
    print("Testing  Accuracy NM " + str(np.sum(btepred == bte) * 100.0 / len(bte)))
if chk["QNM"]:
    btrpred = np.sign(Atr * x["QNM"][0:-1] + x["QNM"][-1])
    btepred = np.sign(Ate * x["QNM"][0:-1] + x["QNM"][-1])
    print("Training Accuracy QNM " + str(np.sum(btrpred == btr) * 100.0 / len(btr)))
    print("Testing  Accuracy QNM " + str(np.sum(btepred == bte) * 100.0 / len(bte)))

print("Numerical solution process is completed.")
# Plot the results.
options = {"dir": "../figs", "name": "LogReg-" + dataset_name}
plot_results(x, info, options, fmin=0)
