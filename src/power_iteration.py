"""
 Function:  x, info  =  power_iteration(fx, gradf, parameter)       
 Purpose:   Implementation of the power method algorithm.     
 Parameter: x0         - Initial estimate.
            maxit      - Maximum number of iterations.
            tolx       - Error toleration for stopping condition.

 Author: Cijo Jose.  [josevancijo@gmail.com] 
"""

import time
import numpy as np


def power_iteration(fx, Mx, parameter, verbose=0, xtrue=None):
    # Initialize x1.
    x = parameter["x0"]
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
            print('Iter = %4d, f(x) = %5.3e\n' %(iter,  info['fx'][iter]))
        # Start the clock.
        timestart = time.time()
        # Update the next iteration.
        x_next = Mx(x)
        # Check stopping criterion.
        if np.linalg.norm(x_next - x) <= parameter["tolx"]:
            break
        x = x_next
    # Finalization.
    info["iter"] = iter
    info["time"] = np.cumsum(info["itertime"])
    info["totaltime"] = info["time"][iter]
    return x, info


"""
**************************************************************************
 END OF THE IMPLEMENTATION.
**************************************************************************
"""
