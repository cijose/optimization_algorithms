"""
 Function:  x, info  =  quasi_newton(fx, gradf, parameter)       
 Purpose:   Implementation of the BFGS algorithm.     
 Parameter: x0         - Initial estimate.
            maxit      - Maximum number of iterations.
            tolx       - Error toleration for stopping condition.
            Lips       - Lipschitz constant for gradient.
            strcnvx    - Strong convexity parameter of f(x).
            kappa      - Line search parameter

 Author: Cijo Jose.  [josevancijo@gmail.com] 

"""
import time

import numpy as np


def quasi_newton(fx, gradf, parameter, verbose=0, xtrue=None):
    # print '%s\n', repmat('*', 1, 68);
    # Initialize x1.
    x = parameter["x0"]
    d = len(x)
    grad_next = gradf(x)
    B = 1.0 * np.eye(d)
    kappa = parameter["kappa"]
    info = {"itertime": [], "fx": [], "iter": None, "time": None, "totaltime": "None"}
    # Set the clock.
    timestart = time.time()
    # Main loop.
    for iter in range(int(parameter["maxit"])):
        # Compute error and save data to be plotted later on.
        info["itertime"].append(time.time() - timestart)
        fc = fx(x)
        info["fx"].append(fc)
        # Print the information.
        if verbose:
            print("Iter = %4d, f(x) = %5.3e" % (iter, info["fx"][iter]))
        # Start the clock.
        timestart = time.time()
        gd = grad_next
        p = np.dot(B, -gd)
        ddg = np.dot(p.T, gd)
        # Update the next iteration.
        alpha = 1.0
        while True:
            x_next = x + alpha * p
            if fx(x_next) <= fc + kappa * alpha * ddg:
                break
            alpha *= 0.5
        if np.sum((alpha * p) ** 2) <= parameter["tolx"]:
            break
        grad_next = gradf(x_next)
        v = grad_next - gd
        bv = np.dot(B, v)
        qbv = np.sum(v * bv)
        B += -np.outer(bv, bv.T) / qbv + alpha * np.outer(p, p.T) / np.sum(p * v)
        x = x_next
    # Finalization.
    info["iter"] = iter
    info["time"] = np.cumsum(info["itertime"])
    info["totaltime"] = info["time"][iter]
    return x, info
