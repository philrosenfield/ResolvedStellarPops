import numpy as np
import pylab as pl

def scatter_contour(x, y,
                    levels=10,
                    threshold=100,
                    log_counts=False,
                    histogram2d_args={},
                    scatter_args={},
                    contour_args={},
                    contour_lw={},
                    hist_bins = 10,
                    ax=None):
    """Scatter plot with contour over dense regions

    Parameters
    ----------
    x, y : arrays
        x and y data for the contour plot
    levels : integer or array (optional, default=10)
        number of contour levels, or array of contour levels
    threshold : float (default=100)
        number of points per 2D bin at which to begin drawing contours
    log_counts :boolean (optional)
        if True, contour levels are the base-10 logarithm of bin counts.
    histogram2d_args : dict
        keyword arguments passed to numpy.histogram2d
        see doc string of numpy.histogram2d for more information
    scatter_args : dict
        keyword arguments passed to pylab.scatter
        see doc string of pylab.scatter for more information
    contourf_args : dict
        keyword arguments passed to pylab.contourf
        see doc string of pylab.contourf for more information
    hist_bins: int
        number of bins for the 2d histogram
    ax: mpl axis instance
        used to update an exisiting plot.
    
    """ 
    H, xbins, ybins = np.histogram2d(x, y, hist_bins, **histogram2d_args)
    
    Nx = len(xbins)
    Ny = len(ybins)

    if log_counts:
        H = np.log10(1 + H)
        threshold = np.log10(1 + threshold)

    levels = np.asarray(levels)
    
    if levels.size == 1:
        levels = np.linspace(threshold, H.max(), levels)

    # only plot points which fall outside contoured region
    x_i = np.digitize(x, xbins) - 1
    y_i = np.digitize(y, ybins) - 1
    x_i[x_i < 0] = 0
    x_i[x_i >= H.shape[0]] = H.shape[0] - 1
    y_i[y_i < 0] = 0
    y_i[y_i >= H.shape[1]] = H.shape[1] - 1
    flag = (H[x_i, y_i] <= levels[1])
    
    if ax == None:
        sc = pl.scatter(x[flag], y[flag], **scatter_args)

        co = pl.contourf(H.T, levels,
                    extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
                    **contour_args)
        if len(contour_lw) != 0:
           cf = pl.contour(H.T, levels,
                           extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
                           **contour_lw)
    else:
        sc = ax.scatter(x[flag], y[flag], **scatter_args)

        co = ax.contourf(H.T, levels,
                         extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
                         **contour_args)
        if len(contour_lw) != 0:
            cf = ax.contour(H.T, levels,
                            extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
                            **contour_lw)

    return sc,co