""" General untilities used throughout the package """
from __future__ import print_function
import numpy as np

__all__ = ['add_data', 'count_uncert_ratio', 'between', 'brighter',
           'closest_match', 'closest_match2d', 'double_gaussian', 'extrap1d',
           'find_peaks', 'gaussian', 'get_verts', 'is_numeric', 'min_dist2d',
           'mp_double_gauss', 'points_inside_poly', 'smooth', 'sort_dict',
           'mp_gauss']

def add_data(old_data, names, new_data):
    '''
    use with Starpop, Track, or any object with data attribute that is a
    np.recarray

    add columns to self.data, update self.key_dict
    see numpy.lib.recfunctions.append_fields.__doc__

    Parameters
    ----------
    old_data : recarray
        original data to add columns to

    new_data : array or sequence of arrays
        new columns to add to old_data

    names : string, sequence
        String or sequence of strings corresponding to the names
        of the new_data.

    Returns
    -------
    array with old_data and new_data
    '''
    import numpy.lib.recfunctions as nlr
    data = nlr.append_fields(np.asarray(old_data), names, new_data).data
    data = data.view(np.recarray)
    return data


def sort_dict(dictionary):
    ''' zip(*sorted(dictionary.items(), key=lambda(k,v):(v,k))) '''
    return zip(*sorted(dictionary.items(), key=lambda(k,v): (v,k)))


def count_uncert_ratio(numerator, denominator):
    ''' combine poisson error to calculate ratio uncertainty'''
    n = float(numerator)
    d = float(denominator)
    try:
        cur = (n / d) * (1./np.sqrt(n) + 1./np.sqrt(d))
    except ZeroDivisionError:
        cur = np.nan
    return cur

def points_inside_poly(points, all_verts):
    """ Proxy to the correct way with mpl """
    from matplotlib.path import Path
    return Path(all_verts).contains_points(points)


def brighter(mag2, trgb, inds=None):
    ''' Indices of mag2 or mag2[inds] brighter (<) than trgb'''
    i, = np.nonzero(mag2 < trgb)
    if inds is not None:
        i = np.intersect1d(i, inds)
    return i


def extrap1d(x, y, xout_arr):
    '''
    linear extapolation from interp1d class with a way around bounds_error.
    Adapted from:
    http://stackoverflow.com/questions/2745329/how-to-make-scipy-interpolate-give-an-extrapolated-result-beyond-the-input-range

    Parameters
    ----------
    x, y : arrays
        values to interpolate

    xout_arr : array
        x array to extrapolate to

    Returns
    -------
    f, yo : interpolator class and extrapolated y array
    '''
    from scipy.interpolate import interp1d
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
    find maxs and mins of an array
    from
    http://stackoverflow.com/questions/4624970/finding-local-maxima-minima-with-numpy-in-a-1d-numpy-array
    Parameters
    ----------
    arr : array
        input array to find maxs and mins

    Returns
    -------
    turning_points : dict
        keys:
        maxima_number: int, how many maxima in arr
        minima_number: int, how many minima in arr
        maxima_locations: list, indicies of maxima
        minima_locations: list, indicies of minima
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
    '''
    index and distance of point in [xarr, yarr] nearest to [xpoint, ypoint]

    Parameters
    ----------
    xpoint, ypoint : floats

    xarr, yarr : arrays

    Returns
    -------
    ind, dist : int, float
        index of xarr, arr and distance
    '''
    dist = np.sqrt((xarr - xpoint) ** 2 + (yarr - ypoint) ** 2)
    return np.argmin(dist), np.min(dist)


def closest_match2d(ind, x1, y1, x2, y2, normed=False):
    '''
    find closest point between of arrays x2[ind], y2[ind] and x1, y1.
    by minimizing the radius of a circle.
    '''
    if normed is True:
        dist = np.sqrt((x1 / np.max(x1) - x2[ind] / np.max(x2)) ** 2 + \
                       (y1 / np.max(y1) - y2[ind] / np.max(y2)) ** 2)
    else:
        dist = np.sqrt((x1 - x2[ind]) ** 2 + (y1 - y2[ind]) ** 2)
    return np.argmin(dist), np.min(dist)


def closest_match(num, arr):
    '''index and difference of closet point of arr to num'''
    index = -1
    arr = np.nan_to_num(arr)
    difference = np.abs(num - arr[0])
    for i in range(len(arr)):
        if difference > np.abs(num - arr[i]):
            difference = np.abs(num - arr[i])
            index = i
    return index, difference


def between(arr, mdim, mbrt, inds=None):
    '''indices of arr or arr[inds] between mdim and mbrt'''
    if mdim < mbrt:
        mtmp = mbrt
        mbrt = mdim
        mdim = mtmp
    i, = np.nonzero((arr < mdim) & (arr > mbrt))
    if inds is not None:
        i = np.intersect1d(i, inds)
    return i


def get_verts(x, y, dx=None, dy=None, nbinsx=10, nbinsy=10, smooth=False):
    '''
    simple edge detection returns n, 2 array
    Parameters
    ----------
    x, y : arrays
        x,y points

    dx, dy : floats
        bin size in each direction of grid to draw verts

    nbinsx, nbinsy : ints
        number of bins in each grid direction to draw verts

    smooth : bool
        in each bin, use the average +/- 3 sigma for x values of verts instead
        of the min and max

    Returns
    -------
    verts : 2d array
        verticies of x, y
    '''
    ymin = y.min()
    ymax = y.max()
    xmin = x.min()
    xmax = x.max()

    if dx is None and dy is None:
        dx = (xmax - xmin) / nbinsx
        dy = (ymax - ymin) / nbinsy
    else:
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
            if smooth is True:
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
    value of numeric: literal, string, int, float, hex, binary
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


def smooth(x, window_len=11, window='hanning'):
    """
    taken from http://www.scipy.org/Cookbook/SignalSmooth
    smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    Parameters
    ----------
    x : array
        the input signal
    window_len : int
        the dimension of the smoothing window; should be an odd integer
    window : str
        the type of window from 'flat', 'hanning', 'hamming', 'bartlett',
            'blackman'
        flat window will produce a moving average smoothing.

    Returns:
    y : array
        the smoothed signal

    Example
    -------
    >>> t = np.linspace(-2, 2, 0.1)
    >>> x = np.sin(t) + np.random.randn(len(t)) * 0.1
    >>> y = smooth(x)

    See Also
    --------
    np.hanning, np.hamming, np.bartlett, np.blackman, np.convolve
    scipy.signal.lfilter
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
    '''gaussian(arr,p): p[0] = norm, p[1] = mean, p[2]=sigma'''
    return p[0] * np.exp( -1 * (x - p[1]) ** 2 / (2 * p[2] ** 2))


def double_gaussian(x, p):
    '''
    gaussian(arr,p): p[0] = norm1, p[1] = mean1, p[2]=sigma1
                     p[3] = norm2, p[4] = mean2, p[5]=sigma2
    '''
    return gaussian(x, p[:3]) + gaussian(x, p[3:])

def mp_gauss(p, fjac=None, x=None, y=None, err=None):
    '''double gaussian for mpfit'''
    model = gaussian(x, p)
    status = 0
    return [status, (y - model) / err]

def mp_double_gauss(p, fjac=None, x=None, y=None, err=None):
    '''double gaussian for mpfit'''
    model = double_gaussian(x, p)
    status = 0
    return [status, (y - model) / err]
