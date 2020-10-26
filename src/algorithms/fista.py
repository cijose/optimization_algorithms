"""
 Function:  x, info = fistaR(fx, gradf, parameter)       
 Purpose:   Implementation of fista.     
 Parameter: x0         - Initial estimate.
            maxit      - Maximum number of iterations.
            tolx       - Error toleration for stopping condition.
            Lips       - Lipschitz constant for gradient.
            strcnvx    - Strong convexity parameter of f(x).

 Author: Cijo Jose.  [josevancijo@gmail.com] 
"""
import time

import numpy as np


def fista(fx, gradf, gx, proxg, parameter, verbose=0):
    x = parameter["x0"]
    y = parameter["x0"]
    t = 1.0
    info = {"itertime": [], "fx": [], "iter": None, "time": None, "totaltime": "None"}
    # Set the clock.
    timestart = time.time()
    # Main loop.
    for iter in range(int(parameter["maxit"])):
        # Compute error and save data to be plotted later on.
        info["itertime"].append(time.time() - timestart)
        fs = fx(x) + gx(x)
        info["fx"].append(fs)
        # Print the information.
        if verbose:
            print("Iter = %4d, f(x) = %5.3e\n" % (iter, info["fx"][iter]))
        # Start the clock.
        fk = 0
        timestart = time.time()
        while True:
            x_next = proxg(y - gradf(y) / parameter["Lips"])
            if fs >= fx(x_next) + gx(x_next) or fk > 0:
                break
            t = 1.0
            y = x
            fk += 1
            # Update the next iteration.
        t_next = 0.5 * (1.0 + np.sqrt(4.0 * t * t + 1.0))
        y = x_next + (t - 1.0) * (x_next - x) / t_next
        # Check stopping criterion.
        if np.linalg.norm(x_next - x) <= parameter["tolx"]:
            break
        x = x_next
        t = t_next
    # Finalization.
    info["iter"] = iter
    info["time"] = np.cumsum(info["itertime"])
    info["totaltime"] = info["time"][iter]
    return x, info


if __name__ == "__main__":
    alpha = 10
    # Parameters for synthetic data.
    cfg = {}
    cfg["n"] = 50
    # number of features
    cfg["p"] = 15
    # number of dimensions
    cfg["noisestd"] = 1e-6
    # standard deviation of additive iid gaussian noise (0 for noiseless)
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

    fx = lambda x: np.linalg.norm(np.dot(A, x) - b)
    gradf = lambda x: 2 * np.dot(A.T, np.dot(A, x) - b)
    gx_l1 = lambda x: alpha * np.sum(np.abs(x))
    prox_l1 = lambda x: np.max(np.abs(x) - alpha, 0) * np.sign(x)

    parameter = dict()
    parameter["Lips"] = np.linalg.norm(np.dot(A.T, A))
    parameter["x0"] = np.zeros((cfg["p"]))
    parameter["tolx"] = 1e-10  # You can vary tolx and maxit
    parameter["maxit"] = 1e3  # to achieve the convergence.
    x, info = fista(fx, gradf, gx_l1, prox_l1, parameter)
