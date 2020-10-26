"""
 Function:  x, info = accelerated_gradient_descent_adaptive_restart(fx, gradf, parameter)       
 Purpose:   Implementation of the accelerated gradient descent algorithm with adaptive restart.     
 Parameter: x0         - Initial estimate.
            maxit      - Maximum number of iterations.
            tolx       - Error toleration for stopping condition.
            Lips       - Lipschitz constant for gradient.
            strcnvx    - Strong convexity parameter of f(x).

 Author: Cijo Jose.  [josevancijo@gmail.com] 
"""
import time

import numpy as np


def accelerated_gradient_descent_adaptive_restart(
    fx, gradf, parameter, verbose=0, xtrue=None
):
    # print '%s\n', repmat('*', 1, 68);
    # Initialize x1.
    # epsilon = 1e-5
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
        fs = fx(x)
        info["fx"].append(fs)
        # Print the information.
        if verbose:
            print("Iter = %4d, f(x) = %5.3e" % (iter, info["fx"][iter]))
        # Start the clock.
        fk = 0
        timestart = time.time()
        while True:
            x_next = y - gradf(y) / parameter["Lips"]
            if fs >= fx(x_next) or fk > 0:
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
