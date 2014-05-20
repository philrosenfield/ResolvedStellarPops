""" Missing documentation """
from __future__ import print_function
import numpy as np
from scipy.interpolate import interp1d
from matplotlib.path import Path

__all__ = ['between', 'bin_up', 'brighter', 'closest_match', 'closest_match2d',
           'double_gaussian', 'extrap1d', 'find_peaks', 'gaussian',
           'get_verts', 'hist_it_up', 'inside', 'is_numeric', 'min_dist2d',
           'mp_double_gauss', 'points_inside_poly', 'smooth', 'spread_bins',
           'count_uncert_ratio']


def count_uncert_ratio(numerator, denominator):
    '''
    combining poisson error for taking a ratio
    '''
    n = float(numerator)
    d = float(denominator)
    return (n / d) * (1./np.sqrt(n) + 1./np.sqrt(d))


def points_inside_poly(points, all_verts):
    """ Proxy to the correct way with mpl """
    return Path(all_verts, close=True).contains_points(points)


def hist_it_up(mag2, res=0.1, threash=10):
    # do fine binning and hist
    bins = np.arange(np.nanmin(mag2), np.nanmax(mag2), res)
    hist = np.histogram(mag2, bins=bins)[0]
    # drop the too fine bins
    binds = spread_bins(hist, threash=threash)
    # return the hist, bins.
    if len(binds) == 0:
        binds = np.arange(len(bins))
    return np.histogram(mag2, bins=bins[binds])


def spread_bins(hist, threash=10):
    '''
    goes through in index order of hist and returns indices such that
    each bin will add to at least threash.
    Returns the indices of the new bins to use (should then re-hist)
    ex:
    bins is something set that works well for large densities
    h = np.histogram(mag2, bins=bins)[0]
    binds = spread_bins(h)
    hist, bins = np.histogram(mag2, bins=bins[binds])
    '''
    b = []
    i = 1
    j = 0
    while i < len(hist):
        while np.sum(hist[j:i]) < threash:
            i += 1
            #print j, i, len(hist)
            if i > len(hist):
                break
        j = i
        b.append(j)
    return np.array(b[:-1])


def brighter(mag2, trgb, inds=None):
    ''' number of stars brighter than trgb, make sure mag2 is
        the same filter as trgb!'''
    i, = np.nonzero(mag2 < trgb)
    if inds is not None:
        i = np.intersect1d(i, inds)
    return i


def extrap1d(x, y, xout_arr):
    '''
    linear extapolation from interp1d class, a way around bounds_error.
    Adapted from:
    http://stackoverflow.com/questions/2745329/how-to-make-scipy-interpolate-give-an-extrapolated-result-beyond-the-input-range
    Returns the interpolator class and yout_arr
    '''
    # Interpolator class
    f = interp1d(x, y)
    xo = xout_arr
    # Boolean indexing approach
    # Generate an empty output array for "y" values
    yo = np.empty_like(xo)

    # Values lower than the minimum "x" are extrapolated at the same time
    low = xo < f.x[0]
    yo[low] = f.y[0] + (xo[low] - f.x[0]) * (f.y[1] - f.y[0]) / (f.x[1] - f.x[0])

    # Values higher than the maximum "x" are extrapolated at same time
    high = xo > f.x[-1]
    yo[high] = f.y[-1] + (xo[high] - f.x[-1]) * (f.y[-1] - f.y[-2]) / (f.x[-1] - f.x[-2])

    # Values inside the interpolation range are interpolated directly
    inside = np.logical_and(xo >= f.x[0], xo <= f.x[-1])
    yo[inside] = f(xo[inside])
    return f, yo


def find_peaks(arr):
    '''
    from
    http://stackoverflow.com/questions/4624970/finding-local-maxima-minima-with-numpy-in-a-1d-numpy-array
    '''
    gradients = np.diff(arr)
    #print gradients

    maxima_num = 0
    minima_num = 0
    max_locations = []
    min_locations = []
    count = 0
    for i in gradients[:-1]:
        count += 1

        if ((cmp(i, 0) > 0) & (cmp(gradients[count], 0) < 0) &
           (i != gradients[count])):
            maxima_num += 1
            max_locations.append(count)

        if ((cmp(i, 0) < 0) & (cmp(gradients[count], 0) > 0) &
           (i != gradients[count])):
            minima_num += 1
            min_locations.append(count)

    turning_points = {'maxima_number': maxima_num,
                      'minima_number': minima_num,
                      'maxima_locations': max_locations,
                      'minima_locations': min_locations}

    return turning_points


def min_dist2d(xpoint, ypoint, xarr, yarr):
    '''index and distance of point in [xarr, yarr] nearest to [xpoint, ypoint]'''
    dist = np.sqrt((xarr - xpoint) ** 2 + (yarr - ypoint) ** 2)
    return np.argmin(dist), np.min(dist)


def closest_match2d(ind, x1, y1, x2, y2, normed=False):
    '''
    finds closest point between of arrays x2[ind], y2[ind] and x1, y1.
    Just minimizes
    the radius of a circle.
    '''
    if normed is True:
        dist = np.sqrt((x1 / np.max(x1) - x2[ind] / np.max(x2)) ** 2 + (y1 / np.max(y1) - y2[ind] / np.max(y2)) ** 2)
    else:
        dist = np.sqrt((x1 - x2[ind]) ** 2 + (y1 - y2[ind]) ** 2)
    return np.argmin(dist), np.min(dist)


def closest_match(num, somearray):
    index = -1
    somearray = np.nan_to_num(somearray)
    difference = np.abs(num - somearray[0])
    for i in range(len(somearray)):
        if difference > np.abs(num - somearray[i]):
            difference = np.abs(num - somearray[i])
            index = i
    return index, difference


def between(arr, mdim, mbrt, inds=None):
    '''
    returns indices between two floats, mdim and mbrt. A quick check exists so
    this will work with regular numbers, not just stupid magnitudes.
    if inds, will return intersection of between and inds
    '''
    if mdim < mbrt:
        mtmp = mbrt
        mbrt = mdim
        mdim = mtmp
    i, = np.nonzero((arr < mdim) & (arr > mbrt))
    if inds is not None:
        i = np.intersect1d(i, inds)
    return i


def inside(x, y, u, v, verts=False, get_verts_args={}):
    """
    returns the indices of u, v that are within the boundries of x, y.
    """
    if not verts:
        verts = get_verts(x, y, **get_verts_args)
    else:
        verts = np.column_stack((x, y))
    points = np.column_stack((u, v))
    mask = points_inside_poly(points, verts)
    ind, = np.nonzero(mask)
    return ind


def get_verts(x, y, **kwargs):
    '''
    simple edge detection returns n, 2 array
    '''
    dx = kwargs.get('dx')
    dy = kwargs.get('dy')
    nbinsx = kwargs.get('nbinsx', 10)
    nbinsy = kwargs.get('nbinsy', 10)
    smooth = kwargs.get('smooth')

    if smooth is None:
        smooth = 0
    else:
        smooth = 1
    ymin = y.min()
    ymax = y.max()
    xmin = x.min()
    xmax = x.max()

    if dx is None and dy is None:
        dx = (xmax - xmin) / nbinsx
        dy = (ymax - ymin) / nbinsy

    if nbinsx is None and nbinsy is None:
        nbinsy = (ymax - ymin) / dy
        nbinsx = (xmax - xmin) / dx

    ymid = []
    min_x = []
    max_x = []
    for j in range(nbinsy):
        yinner = ymin + j * dy
        youter = ymin + (j + 1) * dy
        # counter intuitive because I'm dealing with mags...
        ind = np.nonzero((y > yinner) & (y < youter))[0]
        if len(ind) > 0:
            if smooth == 1:
                min_x.append(np.average(x[ind]) - 3. * np.std(x[ind]))
                max_x.append(np.average(x[ind]) + 3. * np.std(x[ind]))
                ymid.append((yinner + youter) / 2.)
            else:
                min_x.append(min(x[ind]))
                max_x.append(max(x[ind]))
                ymid.append((yinner + youter) / 2.)

    max_x.reverse()
    ymidr = ymid[:]
    ymidr.reverse()

    # close polygon
    max_x.append(min_x[0])

    # close polygon
    ymidr.append(ymid[0])

    # get verticies of polygon
    xs = np.concatenate((min_x, max_x))
    ys = np.concatenate((ymid, ymidr))
    verts = np.column_stack((xs, ys))

    return verts


def is_numeric(lit):
    """
    Return value of numeric literal string or ValueError exception
    From http://rosettacode.org/wiki/Determine_if_a_string_is_numeric#Python
    """
    # Empty String
    if len(lit) <= 0:
        return lit
    # Handle '0'
    if lit == '0':
        return 0
    # Hex/Binary
    if len(lit) > 1:  # sometimes just '-' means no data...
        litneg = lit[1:] if lit[0] == '-' else lit
        if litneg[0] == '0':
            if litneg[1] in 'xX':
                return int(lit, 16)
            elif litneg[1] in 'bB':
                return int(lit, 2)
            else:
                try:
                    return int(lit, 8)
                except ValueError:
                    pass
    # Int/Float/Complex
    try:
        return int(lit)
    except ValueError:
        pass
    try:
        return float(lit)
    except ValueError:
        pass
    try:
        return complex(lit)
    except ValueError:
        pass
    return lit


def bin_up(x, y, nbinsx=None, nbinsy=None, dx=None, dy=None, **kwargs):
    """
    Adapted from contour_plus writen by Oliver Fraser.
    Commented out a means to bin up by setting dx, dy. I found it more useful
    to choose the number of bins in the x and y direction (nbinsx, nbinsy)

    # xrange = np.arange(xmin, xmax, dx)
    # yrange = np.arange(ymin, ymax, dy)
    # nbinsx = xrange.shape[0]
    # nbinsy = yrange.shape[0]

    Returns Z, xrange, yrange
    """
    npts = len(x)
    xmin = float(x.min())  # cast these for matplotlib's w/o np support
    xmax = float(x.max())
    ymin = float(y.min())
    ymax = float(y.max())
    if nbinsx is None:
        nbinsx = (xmax - xmin) / dx
    if nbinsy is None:
        nbinsy = (ymax - ymin) / dy
    if dx is None:
        dx = (xmax - xmin) / nbinsx
    if dy is None:
        dy = (ymax - ymin) / nbinsy

    xrange = kwargs.get('xrange', np.linspace(xmin, xmax, nbinsx))
    yrange = kwargs.get('yrange', np.linspace(ymin, ymax, nbinsy))
    Z = np.zeros((nbinsy, nbinsx))
    xbin = np.zeros(npts)
    ybin = np.zeros(npts)
    # assign each point to a bin
    for i in range(npts):
        xbin[i] = min(int((x[i] - xmin) / dx), nbinsx - 1)
        ybin[i] = min(int((y[i] - ymin) / dy), nbinsy - 1)
        # and count how many are in each bin
        Z[ybin[i]][xbin[i]] += 1

    return Z, xrange, yrange


def smooth(x, window_len=11, window='hanning'):
    """
    taken from http://www.scipy.org/Cookbook/SignalSmooth
    smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    >>> t = np.linspace(-2, 2, 0.1)
    >>> x = np.sin(t) + np.random.randn(len(t)) * 0.1
    >>> y = smooth(x)

    see also:

    np.hanning, np.hamming, np.bartlett, np.blackman, np.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1: 0: -1], x, x[-1: -window_len: -1]]

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


def gaussian(x, p):
    '''
    gaussian(arr,p): p[0] = norm, p[1] = mean, p[2]=sigma
    '''
    return p[0] * np.exp( -1 * (x - p[1]) ** 2 / (2 * p[2] ** 2))


def double_gaussian(x, p):
    '''
    gaussian(arr,p): p[0] = norm1, p[1] = mean1, p[2]=sigma1
                     p[3] = norm2, p[4] = mean2, p[5]=sigma2
    '''
    return gaussian(x, p[:3]) + gaussian(x, p[3:])


def mp_double_gauss(p, fjac=None, x=None, y=None, err=None):
    '''
    double gaussian for mpfit
    '''
    model = double_gaussian(x, p)
    status = 0
    return [status, (y - model) / err]
