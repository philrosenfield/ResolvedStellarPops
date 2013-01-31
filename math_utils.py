import numpy as np
import matplotlib.nxutils as nxutils

def brighter(mag2, trgb, inds=None):
    ''' number of stars brighter than trgb, make sure mag2 is 
        the same filter as trgb!''' 
    i, = np.nonzero(mag2 < trgb)
    if inds is not None:
        i = np.intersect1d(i, inds)
    return i


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

    turning_points = {'maxima_number':maxima_num,
                      'minima_number':minima_num,
                      'maxima_locations':max_locations,
                      'minima_locations':min_locations}  
    
    return turning_points


def closest_match2d(ind, x1, y1, x2, y2):
    '''
    finds closest point between x2[ind], y2[ind] and x1, y1. Just minimizes
    the radius of a circle.
    '''
    dist = np.sqrt((x1/np.max(x1) - x2[ind]/np.max(x2)) ** 2 + (y1/np.max(y1) - y2[ind]/np.max(y2)) ** 2)
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

def bayesian_blocks(t):
    """Bayesian Blocks Implementation

    By Jake Vanderplas.  License: BSD
    Based on algorithm outlined in
    http://adsabs.harvard.edu/abs/2012arXiv1207.5578S

    Parameters
    ----------
    t : ndarray, length N
        data to be histogrammed

    Returns
    -------
    bins : ndarray
        array containing the (N+1) bin edges

    Notes
    -----
    This is an incomplete implementation: it may fail for some
    datasets.  Alternate fitness functions and prior forms can
    be found in the paper listed above.
    """
    # copy and sort the array
    t = np.sort(t)
    N = t.size

    # create length-(N + 1) array of cell edges
    edges = np.concatenate([t[:1], 0.5 * (t[1:] + t[:-1]), t[-1:]])
    block_length = t[-1] - edges

    # arrays needed for the iteration
    nn_vec = np.ones(N)
    best = np.zeros(N, dtype=float)
    last = np.zeros(N, dtype=int)

    #-----------------------------------------------------------------
    # Start with first data cell; add one cell at each iteration
    #-----------------------------------------------------------------
    for K in range(N):
        # Compute the width and count of the final bin for all possible
        # locations of the K^th changepoint
        width = block_length[:K + 1] - block_length[K + 1]
        count_vec = np.cumsum(nn_vec[:K + 1][::-1])[::-1]

        # evaluate fitness function for these possibilities
        fit_vec = count_vec * (np.log(count_vec) - np.log(width))
        fit_vec -= 4  # 4 comes from the prior on the number of changepoints
        fit_vec[1:] += best[:K]

        # find the max of the fitness: this is the K^th changepoint
        i_max = np.argmax(fit_vec)
        last[K] = i_max
        best[K] = fit_vec[i_max]

    #-----------------------------------------------------------------
    # Recover changepoints by iteratively peeling off the last block
    #-----------------------------------------------------------------
    change_points =  np.zeros(N, dtype=int)
    i_cp = N
    ind = N
    while True:
        i_cp -= 1
        change_points[i_cp] = ind
        if ind == 0:
            break
        ind = last[ind - 1]
    change_points = change_points[i_cp:]

    return edges[change_points]


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
    returns the indices of u,v that are within the boundries of x,y.
    """
    if not verts:
        verts = get_verts(x, y, **get_verts_args)
    else:
        verts = np.column_stack((x, y))
    points = np.column_stack((u, v))
    mask = nxutils.points_inside_poly(points, verts)
    ind, = np.nonzero(mask)
    return ind


def get_verts(x, y, **kwargs):
    '''
    simple edge detection returns n,2 array
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
    Commented out a means to bin up by setting dx,dy. I found it more useful
    to choose the number of bins in the x and y direction (nbinsx,nbinsy)

    # xrange = np.arange(xmin, xmax, dx)
    # yrange = np.arange(ymin, ymax, dy)
    # nbinsx = xrange.shape[0]
    # nbinsy = yrange.shape[0]

    Returns Z, xrange,yrange
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

#del np, nxutils
