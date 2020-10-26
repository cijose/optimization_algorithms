"""
 Function:  x, info  = gradient_descent(fx, gradf, parameter)       
 Purpose:   Implementation of the gradient descent algorithm.     
 Parameter: x0         - Initial estimate.
            maxit      - Maximum number of iterations.
            tolx       - Error toleration for stopping condition.
            Lips       - Lipschitz constant for gradient.
            strcnvx    - Strong convexity parameter of f(x).

 Author: Cijo Jose.  [josevancijo@gmail.com]

"""
import time

import numpy as np


def gradient_descent(fx, gradf, parameter, verbose=0, xtrue=None):
    # print '%s\n', repmat('*', 1, 68);
    # Initialize x1.
    x = parameter["x0"]
    if parameter["strcnvx"]:
        alpha = 2.0 / (parameter["Lips"] + parameter["strcnvx"])
    else:
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
        x_next = x - alpha * gradf(x)
        # Check stopping criterion.
        if np.linalg.norm(x_next - x) <= parameter["tolx"]:
            break
        x = x_next
    # Finalization.
    info["iter"] = iter
    info["time"] = np.cumsum(info["itertime"])
    info["totaltime"] = info["time"][iter]
    return x, info
