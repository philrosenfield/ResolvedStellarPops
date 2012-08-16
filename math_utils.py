import numpy as np
import matplotlib.nxutils as nxutils

def inside(x,y,u,v,verts=False,get_verts_args={}):
    """
    returns the indices of u,v that are within the boundries of x,y.
    """
    if verts != True:
        verts = get_verts(x,y,**get_verts_args)
    else:
        verts =  np.column_stack((x,y))
    points = np.column_stack((u,v))
    mask = nxutils.points_inside_poly(points, verts)
    ind, = np.nonzero(mask)
    return ind

def get_verts(x,y,**kwargs):
    '''
    simple edge detection returns n,2 array
    '''
    dx=kwargs.get('dx')
    dy=kwargs.get('dy')
    nbinsx=kwargs.get('nbinsx',10)
    nbinsy=kwargs.get('nbinsy',10)
    smooth=kwargs.get('smooth')


    if smooth == None:
        smooth = 0
    else:
        smooth = 1
    ymin = y.min()
    ymax = y.max()
    xmin = x.min()
    xmax = x.max()
    
    if dx == None and dy == None:
        dx = (xmax-xmin)/nbinsx
        dy = (ymax-ymin)/nbinsy
        
    if nbinsx == None and nbinsy == None:
        nbinsy = (ymax-ymin)/dy
        nbinsx = (xmax-xmin)/dx
    
    ymid = []
    min_x = [] 
    max_x = []
    for j in range(nbinsy):
        yinner = ymin+j*dy
        youter = ymin+(j+1)*dy
        # counter intuitive because I'm dealing with mags...
        ind = np.nonzero((y > yinner) & (y < youter))[0]  
        if len(ind) > 0:
            if smooth == 1:
                min_x.append(np.average(x[ind])-3.*np.std(x[ind]))
                max_x.append(np.average(x[ind])+3.*np.std(x[ind]))
                ymid.append((yinner+youter)/2.)
            else:
                min_x.append(min(x[ind]))
                max_x.append(max(x[ind]))
                ymid.append((yinner+youter)/2.)
    
    max_x.reverse() 
    ymidr = ymid[:]
    ymidr.reverse()
    
    # close polygon
    max_x.append(min_x[0])
    
    # close polygon
    ymidr.append(ymid[0])
    
    # get verticies of polygon
    xs = np.concatenate((min_x,max_x))
    ys = np.concatenate((ymid,ymidr))
    verts = np.column_stack((xs,ys))
    
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
    if lit == '0': return 0
    # Hex/Binary
    if len(lit) > 1: # sometimes just '-' means no data...
        litneg = lit[1:] if lit[0] == '-' else lit
        if litneg[0] == '0':
            if litneg[1] in 'xX':
                return int(lit,16)
            elif litneg[1] in 'bB':
                return int(lit,2)
            else:           
                try:
                    return int(lit,8)
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

def bin_up(x,y,nbinsx=None,nbinsy=None,dx=None,dy=None,**kwargs):
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
    xmin  = float( x.min() ) # cast these for matplotlib's w/o np support
    xmax  = float( x.max() )
    ymin  = float( y.min() )
    ymax  = float( y.max() )
    if nbinsx == None:
        nbinsx = (xmax - xmin) / dx
    if nbinsy == None:
        nbinsy = (ymax - ymin) / dy
    if dx == None:
        dx = (xmax - xmin) / nbinsx
    if dy == None:
        dy = (ymax - ymin) / nbinsy
    
    xrange = kwargs.get('xrange',np.linspace(xmin, xmax, nbinsx))
    yrange = kwargs.get('yrange',np.linspace(ymin, ymax, nbinsy))
    Z    = np.zeros((nbinsy, nbinsx))
    xbin = np.zeros(npts)
    ybin = np.zeros(npts)
    # assign each point to a bin
    for i in range(npts):
        xbin[i] = min(int((x[i] - xmin) / dx), nbinsx-1)
        ybin[i] = min(int((y[i] - ymin) / dy), nbinsy-1)
        # and count how many are in each bin
        Z[ ybin[i] ][ xbin[i] ] += 1
    
    return Z,xrange,yrange

