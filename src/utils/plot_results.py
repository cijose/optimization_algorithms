import colorsys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def distinguishable_colors(num_colors):
    colors = []
    for i, ii in enumerate(np.arange(0.0, 360.0, 360.0 / num_colors)):
        hue = ii / 360.0
        lightness = (50 + 20.0 * (i % 2 == 0)) / 100.0
        saturation = (100 - 50 * (i % 3 == 0)) / 100.0
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors


def plot_results(x, info, options, fmin=None):
    num_points = 17
    font = {"family": "normal", "weight": "bold", "size": 7}
    plt.rc("font", **font)
    plt.rc("text", usetex=True)
    # With respect to number of iterations.
    plt.figure(1)
    plt.clf()
    methodsdone = list(x.keys())
    allMarkers = ["s", "d", "v", "^", ">", "<", "p", "h", "+", "o", "*", "x"]
    colors = distinguishable_colors(len(methodsdone))
    ax = plt.gca()

    if fmin > 0:
        ylbl = r"$ \big( f(\mathbf{x}) - f^* \big) / f^* $"
    elif fmin == None:
        fmin = []
        for count in range(len(methodsdone)):
            fmin.append(min(info[methodsdone[count]]["fx"]))
        fmin = min(fmin)
        ylbl = r"$ \big( f(\mathbf{x}) - f^*_{aprx} \big) / f^*_{aprx} $"
        print(
            "f^* is approximated by the minimum over the methods you have implemented. \n"
        )
    if fmin == 0:
        ffmin = []
        for count in range(len(methodsdone)):
            ffmin.append(min(info[methodsdone[count]]["fx"]))
        ffmin = min(ffmin)

        ylbl = "$f(\mathbf{x})$"
        for count in range(len(methodsdone)):
            info[methodsdone[count]]["error"] = info[methodsdone[count]]["fx"]
        print("f^* is 0 so you will get f(x) vs number of iterations and time. \n")
    else:
        for count in range(len(methodsdone)):
            info[methodsdone[count]]["error"] = (
                info[methodsdone[count]]["fx"] - fmin
            ) / fmin
    for count in range(len(methodsdone)):
        ypoints = np.asarray(info[methodsdone[count]]["error"])
        xpoints = np.asarray(np.arange(len(ypoints)))
        marker_loc = [int(i) for i in np.linspace(0, len(xpoints) - 1, num_points)]
        plt.semilogy(xpoints, ypoints, linewidth=2, color=colors[count], label=None)
        plt.semilogy(
            xpoints[marker_loc],
            ypoints[marker_loc],
            color=colors[count],
            marker=allMarkers[count],
            markeredgecolor=colors[count],
            markerfacecolor=colors[count],
            linestyle="-",
            markersize=7,
            label=methodsdone[count],
        )
    plt.text(
        0.01,
        0.01,
        "$f^*_{\mbox{aprx}} = %5.9e$" % (ffmin),
        fontsize=17,
        transform=ax.transAxes,
    )
    plt.xlabel("Number of Iterations")
    plt.ylabel(ylbl)
    plt.legend(numpoints=1, markerscale=1.0)
    plt.title(options["name"])
    plt.savefig(
        options["dir"] + "/" + options["name"] + "_iter.eps", format="eps", dpi=1000
    )
    # With respect to time.
    plt.figure(2)
    plt.clf()
    for count in range(len(methodsdone)):
        ypoints = np.asarray(info[methodsdone[count]]["error"])
        xpoints = np.asarray(info[methodsdone[count]]["time"])
        marker_loc = [int(i) for i in np.linspace(0, len(xpoints) - 1, num_points)]
        plt.semilogy(xpoints, ypoints, linewidth=2, color=colors[count], label=None)
        plt.semilogy(
            xpoints[marker_loc],
            ypoints[marker_loc],
            color=colors[count],
            marker=allMarkers[count],
            linestyle="-",
            markeredgecolor=colors[count],
            markerfacecolor=colors[count],
            markersize=11,
            label=methodsdone[count],
        )
    plt.text(
        0.01,
        0.01,
        "$f^*_{\mbox{aprx}} = %5.9e$" % (ffmin),
        fontsize=17,
        transform=ax.transAxes,
    )
    plt.xlabel("Time (s)")
    plt.ylabel(ylbl)
    plt.legend(numpoints=1, markerscale=1.0)
    plt.title(options["name"])
    plt.savefig(
        options["dir"] + "/" + options["name"] + "_time.eps", format="eps", dpi=1000
    )
    plt.figure(3)
    plt.clf()
    for count in range(len(methodsdone)):
        ypoints = np.asarray(info[methodsdone[count]]["error"])
        xpoints = np.asarray(np.arange(len(ypoints)))
        marker_loc = [int(i) for i in np.linspace(0, len(xpoints) - 1, num_points)]
        plt.loglog(xpoints, ypoints, linewidth=2, color=colors[count], label=None)
        plt.loglog(
            xpoints[marker_loc],
            ypoints[marker_loc],
            color=colors[count],
            marker=allMarkers[count],
            markeredgecolor=colors[count],
            markerfacecolor=colors[count],
            linestyle="-",
            markersize=7,
            label=methodsdone[count],
        )
    plt.text(
        0.01,
        0.01,
        "$f^*_{\mbox{aprx}} = %5.9e$" % (ffmin),
        fontsize=17,
        transform=ax.transAxes,
    )
    plt.xlabel("Number of Iterations")
    plt.ylabel(ylbl)
    plt.legend(numpoints=1, markerscale=1.0)
    plt.title(options["name"] + "-loglog")
    plt.savefig(
        options["dir"] + "/" + options["name"] + "_log_log_iter.eps",
        format="eps",
        dpi=1000,
    )
    # With respect to time.
    plt.figure(4)
    plt.clf()
    for count in range(len(methodsdone)):
        ypoints = np.asarray(info[methodsdone[count]]["error"])
        xpoints = np.asarray(info[methodsdone[count]]["time"])
        marker_loc = [int(i) for i in np.linspace(0, len(xpoints) - 1, num_points)]
        plt.loglog(xpoints, ypoints, linewidth=2, color=colors[count], label=None)
        plt.loglog(
            xpoints[marker_loc],
            ypoints[marker_loc],
            color=colors[count],
            marker=allMarkers[count],
            linestyle="-",
            markeredgecolor=colors[count],
            markerfacecolor=colors[count],
            markersize=11,
            label=methodsdone[count],
        )
    plt.text(
        0.01,
        0.01,
        "$f^*_{\mbox{aprx}} = %5.9e$" % (ffmin),
        fontsize=17,
        transform=ax.transAxes,
    )
    plt.xlabel("Time (s)")
    plt.ylabel(ylbl)
    plt.legend(numpoints=1, markerscale=1.0)
    plt.title(options["name"] + "-loglog")
    plt.savefig(
        options["dir"] + "/" + options["name"] + "_log_log_time.eps",
        format="eps",
        dpi=1000,
    )

    plt.show()
