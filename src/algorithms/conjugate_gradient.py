"""

 Function:  x, info = conjugate_gradient(fx, gradf, parameter)       
 Purpose:   Implementation of the congugate gradient algorithm.     
 Parameter: x0         - Initial estimate.
            maxit      - Maximum number of iterations.
            tolx       - Error toleration for stopping condition.
            Lips       - Lipschitz constant for gradient.
            strcnvx    - Strong convexity parameter of f(x).

 Author: Cijo Jose.  [cijose-at-idiap.ch] 

"""
import time

import numpy as np


def conjugate_gradient(fx, phi, y, parameter, verbose=0, xtrue=None):
    # Initialize x1.
    x = parameter["x0"]
    r = phi(x) - y
    p = -r
    info = {"itertime": [], "fx": [], "iter": None, "time": None, "totaltime": "None"}
    # Set the clock.
    timestart = time.time()
    # Main loop.
    for iter in range(int(parameter["maxit"])):
        # Compute error and save data to be plotted later on.
        info["itertime"].append(time.time() - timestart)
        info["fx"].append(fx(x))
        # Print the information.
        if verbose:
            print("Iter = %4d, f(x) = %5.3e" % (iter, info["fx"][iter]))
        # Start the clock.
        timestart = time.time()
        phip = phi(p)
        alpha = np.dot(r.T, p) / np.dot(p.T, phip)
        # Update the next iteration.
        x_next = x - alpha * p
        r = phi(x_next) - y
        beta = np.dot(r.T, phip) / np.dot(p.T, phip)
        p = -r + beta * p
        # Check stopping criterion.
        if np.linalg.norm(x_next - x) <= parameter["tolx"]:
            break
        x = x_next
    # Finalization.
    info["iter"] = iter
    info["time"] = np.cumsum(info["itertime"])
    info["totaltime"] = info["time"][iter]
    return x, info
