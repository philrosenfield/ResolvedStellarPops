from __future__ import print_function
import os
import numpy as np

from .. import io
from ..bc_config import BCDIR
from ..angst_tables import angst_data

__all__ = ['Av2Alambda', 'Mag2mag', 'bens_fmt_galaxy_info', 'get_dmodAv',
           'hla_galaxy_info', 'mag2Mag', 'parse_mag_tab', ]


def parse_mag_tab(photsys, filter, bcdir=None):
    bcdir = bcdir or BCDIR
    photsys = photsys.lower()

    tab_mag_dir = os.path.join(bcdir, 'tab_mag_odfnew/')
    tab_mag, = io.get_files(tab_mag_dir, 'tab_mag_{0:s}.dat'.format(photsys))

    tab = open(tab_mag, 'r').readlines()
    mags = tab[1].strip().split()
    Alam_Av = map(float, tab[3].strip().split())
    Alam_Av[mags.index(filter)]
    return Alam_Av[mags.index(filter)]


def Av2Alambda(Av, photsys, filter):
    Alam_Av = parse_mag_tab(photsys, filter)
    Alam = Alam_Av * Av
    return Alam


def Mag2mag(Mag, filterx, photsys, target=None, Av=0., dmod=0., **kwargs):
    '''
    Return an apparent redenned magnitude from absolute magnitude

    This uses Leo calculations using Cardelli et al 1998 extinction curve with
    Rv = 3.1

    Parameters
    ----------
    Mag: float
        absolute magnitude


    filterx:


    photsys:

    target:
        galaxy id to find in angst survey paper table 5

    Av: float
        0.0
    dmod: float
        0.0

    Returns
    -------
    mag: float
        apparent magnitude (added distance and extinction)
    '''
    if target is not None:
        filter2 = kwargs.get('filter2', filterx)
        filter1 = kwargs.get('filter1', None)
        trgb, Av, dmod = angst_data.get_tab5_trgb_av_dmod(target, ','.join((filter1, filter2)))

    if Av != 0.0:
        Alam_Av = parse_mag_tab(photsys, filterx)
        A = Alam_Av * Av
    else:
        A = 0.

    return Mag + dmod + A


def mag2Mag(mag, filterx, photsys, target=None, Av=0., dmod=0., **kwargs):
    '''
    Returns the apparent magnitude from an absolute magnitude

    This uses Leo calculations using Cardelli et al 1998 extinction curve
    with Rv = 3.1

    Parameters
    ----------
    mag: float
        apparent magnitude


    filterx:


    photsys:

    target:
        galaxy id to find in angst survey paper table 5

    Av: float
        0.0
    dmod: float
        0.0

    Returns
    -------
    Mag: float
        absolute magnitude (magnitude corrected for distance and extinction)
    '''

    target = kwargs.get('target', None)
    if target is not None:
        filter2 = kwargs.get('filter2', filterx)
        filter1 = kwargs.get('filter1', None)
        _, Av, dmod = angst_data.get_tab5_trgb_av_dmod(target, ','.join((filter1, filter2)))
    else:
        Av = kwargs.get('Av', 0.0)
        dmod = kwargs.get('dmod', 0.)

    if Av != 0.0:
        Alam_Av = parse_mag_tab(photsys, filterx)
        A = Alam_Av * Av
    else:
        A = 0.

    return mag - dmod - A


def get_dmodAv(gal=None, **kwargs):
    '''
    dmod and Av can be separated only if we have more than one filter to deal
    with.

    This will take either a Galaxies.star_pop instance (galaxy, simgalaxy) or a
    pile of kwargs.

    .. math:
        mag1 - Mag1 = dmod + (Alambda1/Av) * Av
        mag2 - Mag2 = dmod + (Alambda2/Av) * Av

        Av   = ((mag1 - Mag1) - (mag2 - Mag2)) / (Alambda1/Av - Alambda2/Av)
        dmod = mag1 - Mag1 - al1 * Av
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


def hla_galaxy_info(filename):
    """ Parse HLA information from a given file name

    Parameters
    ----------
    filename: str
        name of the file to get info from

    Returns
    -------
    survey: str
        survey name

    propid: str
        proposal id

    target: str
        target of the proposal

    filters: str
        filters

    photsys: str
        photometric system
    """
    name = os.path.split(filename)[1]
    name_split = name.split('_')[1:-2]
    survey, lixo, photsys, pidtarget, filters = name_split
    propid = pidtarget.split('-')[0]
    target = '-'.join(pidtarget.split('-')[1:])
    return survey, propid, target, filters, photsys


def bens_fmt_galaxy_info(filename):
    """ Parse Ben's file names and extract relevant info

    Parameters
    ----------
    filename: str
        name of the file to get info from

    Returns
    -------
    propid: str
        proposal id

    target: str
        target of the proposal

    filter1: str
        name of the 1st filter

    filter2: str
        name of the 2nd filter
    """
    info = os.path.split(filename)[1].split('.')[0].split('_')
    # Why not just split? Sometimes there's an IR right in there for
    # reasons beyond comprehension.
    (propid, target) = info[:2]
    (filter1, filter2) = info[-2:]
    return propid, target, filter1, filter2


def read_galtable(**kwargs):
    fname = kwargs.get('filename')
    br = kwargs.get('br', True)
    tpagb = kwargs.get('tpagb')
    if not fname:
        if br is not None:
            fname = '/Users/phil/research/BRratio/tables/brratio_galtable.dat'
            dtype = [('Target', '|S10'), ('O/H', '<f8'), ('Z', '<f8')]
            kwargs = {'autostrip': 1, 'delimiter': ',', 'dtype': dtype}
        if tpagb is not None:
            pass
    return np.genfromtxt(fname, **kwargs)


def galaxy_metallicity(gal, target, **kwargs):
    '''
    add metallicity to galaxy object.
    '''
    print('galaxy_metallicity is gonna break shit.')
    got = 0
    met_table = read_galtable(**kwargs)
    for i, t in enumerate(met_table['Target']):
        if t.lower() in target:
            z = met_table['Z'][i]
            if met_table['Target'][i] != t:
                print('fuck!!!')
            got = 1
    if got == 0:
        print("{0:s} not found".format(target))
        z = np.nan
    gal.z = z
    return z


def get_fake(target, fake_loc='.'):
    return io.get_files(fake_loc, '*%s*.matchfake' % target.upper())[0]
