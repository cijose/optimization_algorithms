"""
 Function:  x, info = accelerated_gradient_descent(fx, gradf, parameter)       
 Purpose:   Implementation of the accelerated gradient descent algorithm.     
 Parameter: x0         - Initial estimate.
            maxit      - Maximum number of iterations.
            tolx       - Error toleration for stopping condition.
            Lips       - Lipschitz constant for gradient.
            strcnvx    - Strong convexity parameter of f(x).

 Author: Cijo Jose.  [josevancijo@gmail.com] 

"""
import time

import numpy as np


def accelerated_gradient_descent(fx, gradf, parameter, verbose=0, xtrue=None):
    # Initialize x1.
    x = parameter["x0"]
    y = parameter["x0"]
    t = 1.0
    if parameter["strcnvx"]:
        sL = np.sqrt(parameter["Lips"])
        sC = np.sqrt(parameter["strcnvx"])
        gamma = (sL - sC) / (sL + sC)
    alpha = 1.0 / parameter["Lips"]
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
        # Update the next iteration.
        x_next = y - gradf(y) * alpha
        t_next = 0.5 * (1.0 + np.sqrt(4.0 * t ** 2 + 1.0))
        if parameter["strcnvx"]:
            y = x_next + gamma * (x_next - x)
            # print gamma
        else:
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
