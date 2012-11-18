import fileIO
from angst_tables import AngstTables
import os
import numpy as np


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


def parse_mag_tab(photsys, filter, bcdir=None):
    if not bcdir:
        try:
            bcdir = os.environ['BCDIR']
        except KeyError:
            print 'need bcdir environmental variable, or to pass it to parse_mag_tab' 

    photsys = photsys.lower()

    tab_mag_dir = os.path.join(bcdir, 'tab_mag_odfnew/')
    tab_mag, = fileIO.get_files(tab_mag_dir, 'tab_mag_%s.dat' % photsys)

    tab = open(tab_mag, 'r').readlines()
    mags = tab[1].strip().split()
    Alam_Av = map(float, tab[3].strip().split())
    try:
        Alam_Av[mags.index(filter)]
    except ValueError:
        print '%s not in list' % filter
        print tab_mag, mags
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
        trgb, Av, dmod = AngstTables.get_tab5_trgb_av_dmod(target, 
                                                           ', '.join((filter1, 
                                                                     filter2)))
    else:
        Av = kwargs.get('Av', 0.0)
        dmod = kwargs.get('dmod', 0.)
        
    if Av != 0.0:
        Alam_Av = parse_mag_tab(photsys, filter)
        A = Alam_Av * Av
    
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