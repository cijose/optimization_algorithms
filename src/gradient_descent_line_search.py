"""
 Function:  x, info  = gradient_descent_line_search(fx, gradf, parameter)       
 Purpose:   Implementation of the gradient descent algorithm with line search.     
 Parameter: x0         - Initial estimate.
            maxit      - Maximum number of iterations.
            tolx       - Error toleration for stopping condition.
            Lips       - Lipschitz constant for gradient.
            strcnvx    - Strong convexity parameter of f(x).
            kappa      - Line search parameter for GD

 Author: Cijo Jose.  [josevancijo@gmail.com] 

"""
import time
import numpy as np


def gradient_descent_line_search(fx, gradf, parameter, verbose=0, xtrue=None):
    # Initialize x1.
    x = parameter["x0"]
    fc = fx(x)
    kappa = parameter["kappa"] * 0.5
    info = {"itertime": [], "fx": [], "iter": None, "time": None, "totaltime": "None"}
    # Set the clock.
    timestart = time.time()
    # Main loop.
    for iter in range(int(parameter["maxit"])):
        # Compute error and save data to be plotted later on.
        info["itertime"].append(time.time() - timestart)
        info["fx"].append(fc)
        # Print the information.
        if verbose:
            print("Iter = %4d, f(x) = %5.3e\n" % (iter, info["fx"][iter]))
        # Start the clock.
        L = parameter["Lips"] / 64.0
        timestart = time.time()
        gradfc = gradf(x)
        nrmgrad = np.dot(gradfc.T, gradfc)
        while True:
            x_next = x - gradfc / L
            fs = fx(x_next)
            q = fc - kappa * nrmgrad / L
            if fs <= q:
                break
            L = L * 2.0
        # Check stopping criterion.
        if np.linalg.norm(x_next - x) <= parameter["tolx"]:
            break
        x = x_next
        fc = fs
    # Finalization.
    info["iter"] = iter
    info["time"] = np.cumsum(info["itertime"])
    info["totaltime"] = info["time"][iter]
    return x, info
