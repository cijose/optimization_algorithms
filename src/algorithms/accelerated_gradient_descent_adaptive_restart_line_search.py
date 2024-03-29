"""

 Function:  x, info = accelerated_gradient_descent_adaptive_restart_line_search(fx, gradf, parameter)       
 Purpose:   Implementation of the accelerated gradient descent algorithm with line search and adaptive restart.     
 Parameter: x0         - Initial estimate.
            maxit      - Maximum number of iterations.
            tolx       - Error toleration for stopping condition.
            Lips       - Lipschitz constant for gradient.
            strcnvx    - Strong convexity parameter of f(x).

 Author: Cijo Jose.  [josevancijo@gmail.com] 

"""
import time

import numpy as np


def accelerated_gradient_descent_adaptive_restart_line_search(
    fx, gradf, parameter, verbose=0, xtrue=None
):
    # print '%s\n', repmat('*', 1, 68);
    # Initialize x1.
    x = parameter["x0"]
    y = parameter["x0"]
    fy = fs = fx(x)
    L = parameter["Lips"]
    t = 1.0
    info = {"itertime": [], "fx": [], "iter": None, "time": None, "totaltime": "None"}
    # Set the clock.
    timestart = time.time()
    # Main loop.
    for iter in range(int(parameter["maxit"])):
        # Compute error and save data to be plotted later on.
        info["itertime"].append(time.time() - timestart)
        info["fx"].append(fs)
        # Print the information.
        if verbose:
            print("Iter = %4d, f(x) = %5.3e" % (iter, info["fx"][iter]))
        # Start the clock.
        timestart = time.time()
        while True:
            gradfy = gradf(y)
            dnrmy_2 = np.linalg.norm(gradfy) ** 2
            L_next = parameter["Lips"] / 64.0
            while True:
                x_next = y - gradfy / L_next
                fc = fx(x_next)
                q = fy - 0.5 * dnrmy_2 / L_next
                if fc <= q:
                    break
                L_next = L_next * 2.0
            if fs >= fc:
                break
            t = 1.0
            y = x
            fy = fs
        tau = L_next / L
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * tau * t ** 2))
        y = x_next + (x_next - x) * (t - 1.0) / t_next
        # Check stopping criterion.
        if np.linalg.norm(x_next - x) <= parameter["tolx"]:
            break
        x = x_next
        t = t_next
        L = L_next
        fs = fc
        fy = fx(y)
    # Finalization.
    info["iter"] = iter
    info["time"] = np.cumsum(info["itertime"])
    info["totaltime"] = info["time"][iter]
    return x, info
