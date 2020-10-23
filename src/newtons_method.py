"""
 Function:  x, info  =  newtons_method(fx, gradf, hessf, parameter)       
 Purpose:   Implementation of the gradient descent algorithm.     
 Parameter: x0         - Initial estimate.
            maxit      - Maximum number of iterations.
            tolx       - Error toleration for stopping condition.
            Lips       - Lipschitz constant for gradient.
            strcnvx    - Strong convexity parameter of f(x).
            kappa      - Line search parameter.

 Author: Cijo Jose.  [josevancijo@gmail.com] 

"""
import time
import numpy as np
import scipy as sp
import scipy.sparse.linalg


def newtons_method(fx, gradf, hessf, parameter, verbose=0, xtrue=None):
    # print '%s\n', repmat('*', 1, 68);
    # Initialize x1.
    x = parameter["x0"]
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
            print("Iter = %4d, f(x) = %5.3e\n" % (iter, info["fx"][iter]))
        # Start the clock.
        timestart = time.time()
        gf = gradf(x)
        p = sp.sparse.linalg.cgs(hessf(x), -gf, maxiter=1000, tol=1e-10)[0]
        ddg = np.sum(p * gf)
        alpha = 1.0
        while True:
            x_next = x + alpha * p
            if fx(x_next) <= fc + kappa * alpha * ddg:
                break
            alpha *= 0.5
        if np.linalg.norm(p * alpha) <= parameter["tolx"]:
            break
        x = x_next
    # Finalization.
    info["iter"] = iter
    info["time"] = np.cumsum(info["itertime"])
    info["totaltime"] = info["time"][iter]
    return x, info
