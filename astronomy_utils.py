import fileIO
import os
import numpy as np
import angst_tables
angst_data = angst_tables.AngstTables()
import graphics.GraphicsUtils as rspgraph
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def points_inside_ds9_polygon(reg_name, ra_points, dec_points):
    '''
    uses read_reg
    '''
    radec = np.column_stack((ra_points, dec_points))
    ras, decs = read_reg(reg_name,shape='polygon')
    verts = np.column_stack((ras, decs))
    mask = nxutils.points_inside_poly(radec, verts)
    inds, = np.nonzero(mask)
    return inds


def read_reg(reg_name, shape='polygon'):
    '''
    Takes a ds9 reg file and loads ra dec into arrays. 
    Only tested on polygon shape. 
    returns ras, decs
    '''
    with open(reg_name, 'r') as f:
        lines = f.readlines()
    xy, = [map(float, line.replace(shape + '(', '').replace(')', '').split('#')[0].split(',')) 
               for line in lines if line.startswith(shape)]
    ras = xy[0::2]
    decs =xy[1::2]
    return ras, decs


def hess(color, mag, binsize, **kw):
    """
    Compute a hess diagram (surface-density CMD) on photometry data.

    INPUT:
       color  
       mag
       binsize -- width of bins, in magnitudes

    OPTIONAL INPUT:
       cbin=  -- set the centers of the color bins
       mbin=  -- set the centers of the magnitude bins
       cbinsize -- width of bins, in magnitudes

    OUTPUT:
       A 3-tuple consisting of:
         Cbin -- the centers of the color bins
         Mbin -- the centers of the magnitude bins
         Hess -- The Hess diagram array
    
    EXAMPLE:
      cbin = out[0]
      mbin = out[1]
      imshow(out[2])
      yticks(range(0, len(mbin), 4), mbin[range(0, len(mbin), 4)])
      xticks(range(0, len(cbin), 4), cbin[range(0, len(cbin), 4)])
      ylim([ylim()[1], ylim()[0]])

    2009-02-08 23:01 IJC: Created, on a whim, for LMC data (of course)
    2009-02-21 15:45 IJC: Updated with cbin, mbin options
    2012 PAR: Gutted and changed it do histogram2d for faster implementation.
    """
    defaults = dict(mbin=None, cbin=None, verbose=False)
    
    for key in defaults:
        if (not kw.has_key(key)):
            kw[key] = defaults[key]
    
    if kw['mbin'] is None:
        mbin = np.arange(mag.min(), mag.max(), binsize)
    else:
        mbin = np.array(kw['mbin']).copy()
    if kw['cbin'] is None:
        cbinsize = kw.get('cbinsize')
        if cbinsize is None: cbinsize=binsize
        cbin = np.arange(color.min(), color.max(), cbinsize)
    else:
        cbin = np.array(kw['cbin']).copy()

    hesst, cbin, mbin = np.histogram2d(color, mag, bins=[cbin, mbin])
    hess = hesst.T
    return (cbin, mbin, hess)


def hess_plot(hess, fig=None, ax=None, colorbar=False, filter1=None, filter2=None,
              imshow_kw={}, imshow=True):
    '''
    Plots a hess diagram with imshow.
    default kwargs passed to imshow:
    default_kw = {'cmap': cm.gray,
                  'interpolation': 'nearest',
                  'extent': [hess[0][0], hess[0][-1],
                             hess[1][-1], hess[1][0]]}
    '''
    if fig is None and ax is None:
        fig = plt.figure(figsize=(9, 9))
        ax = plt.axes()

    default_kw = {'norm': LogNorm(vmin=None, vmax=hess[2].max()),
                  'cmap': cm.gray,
                  'interpolation': 'nearest',
                  'extent': [hess[0][0], hess[0][-1],
                             hess[1][-1], hess[1][0]]}

    imshow_kw = dict(default_kw.items() + imshow_kw.items())
            
    if imshow is True:
        im = ax.imshow(hess[2], **imshow_kw)
        rspgraph.forceAspect(ax, aspect=1)
    else:
        im = ax.contourf(hess[2], **imshow_kw)

    if colorbar is True:    
        fig.colorbar(im, shrink=0.77)
            
    if filter2 is not None and filter1 is not None:
        ax.set_ylabel('$%s$' % (filter2), fontsize=20)
        ax.set_xlabel('$%s-%s$' % (filter1, filter2), fontsize=20)
        ax.tick_params(labelsize=16)

    return ax


def parse_mag_tab(photsys, filter, bcdir=None):
    if not bcdir:
        try:
            bcdir = os.environ['BCDIR']
        except KeyError:
            logger.error('need bcdir environmental variable, or to pass it to parse_mag_tab')

    photsys = photsys.lower()

    tab_mag_dir = os.path.join(bcdir, 'tab_mag_odfnew/')
    tab_mag, = fileIO.get_files(tab_mag_dir, 'tab_mag_%s.dat' % photsys)

    tab = open(tab_mag, 'r').readlines()
    mags = tab[1].strip().split()
    Alam_Av = map(float, tab[3].strip().split())
    try:
        Alam_Av[mags.index(filter)]
    except ValueError:
        logger.error('%s not in list' % filter)
        logger.errot(tab_mag, mags)
    return Alam_Av[mags.index(filter)]


def Mag2mag(Mag, filter, photsys, **kwargs):
    '''
    This uses Leo calculations using Cardelli et al 1998 extinction curve
    with Rv = 3.1

    kwargs:
    target, galaxy id to find in angst survey paper table 5
    Av, 0.0
    dmod, 0.0
    '''
    target = kwargs.get('target', None)
    A = 0.
    if target is not None:
        filter2 = filter
        filter1 = kwargs.get('filter1', None)
        trgb, Av, dmod = angst_data.get_tab5_trgb_av_dmod(target, 
                                                          ','.join((filter1, 
                                                                     filter2)))
    else:
        Av = kwargs.get('Av', 0.0)
        dmod = kwargs.get('dmod', 0.)
        
    if Av != 0.0:
        Alam_Av = parse_mag_tab(photsys, filter)
        A = Alam_Av * Av
    if dmod == 0. and A == 0.:
        logger.warning('Mag2mag did nothing.')
    return Mag+dmod+A


def mag2Mag(mag, filter, photsys, **kwargs):
    '''
    This uses Leo calculations using Cardelli et al 1998 extinction curve
    with Rv = 3.1
    
    kwargs:
    target, galaxy id to find in angst survey paper table 5
    Av, 0.0
    dmod, 0.0
    '''
    
    target = kwargs.get('target', None)
    A = 0.
    if target != None:
        trgb, Av, dmod = get_tab5_trgb_Av_dmod(target)
    else:
        Av = kwargs.get('Av', 0.0)
        dmod = kwargs.get('dmod', 0.)
        
    if Av != 0.0:
        Alam_Av = parse_mag_tab(photsys, filter)
        A = Alam_Av * Av
    
    return mag-dmod-A


def get_dmodAv(gal=None, **kwargs):
    '''
    dmod and Av can be separated only if we have more than one filter to deal
    with.
    
    This will take either a Galaxies.star_pop instance (galaxy, simgalaxy) or
    a pile of kwargs.

    SO:
    mag1 - Mag1 = dmod + Alambda1/Av * Av
    mag2 - Mag2 = dmod + Alambda2/Av * Av
    
    subtract:
    ((mag1 - Mag1) - (mag2 - Mag2)) = Av * (Alambda1/Av - Alambda2/Av)
    
    rearrange:
    Av = ((mag1 - Mag1) - (mag2 - Mag2)) / (Alambda1/Av - Alambda2/Av)
    
    plug Av into one of the first equations and solve for dmod.
    '''
    if gal is None:
        photsys = kwargs.get('photsys')
        filter1 = kwargs.get('filter1')
        filter2 = kwargs.get('filter2')
        mag1 = kwargs.get('mag1')
        mag2 = kwargs.get('mag2')
        Mag1 = kwargs.get('Mag1')
        Mag2 = kwargs.get('Mag2')
    else:
        photsys = gal.photsys
        filter1 = gal.filter1
        filter2 = gal.filter2
        mag1 = gal.mag1
        mag2 = gal.mag2
        Mag1 = gal.Mag1
        Mag2 = gal.Mag2
    
    Al1 = parse_mag_tab(photsys, filter1)
    Al2 = parse_mag_tab(photsys, filter2)
    Av = (mag1 - Mag1 - mag2 + Mag2) / (Al1 - Al2)
    dmod = mag1 - Mag1 - Al1 * Av
    # could do some assert dmods and Avs  are all the same...
    return dmod[0], Av[0]




