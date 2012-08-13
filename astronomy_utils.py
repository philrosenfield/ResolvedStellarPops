import fileIO
from angst_tables import AngstTables
import os


def parse_mag_tab(photsys,filter):
    bcdir = os.environ['BCDIR']
    tab_mag_dir = os.path.join(bcdir,'tab_mag_odfnew/')
    tab_mag = fileIO.get_files(tab_mag_dir,'tab_mag_'+photsys+'.dat')[0]
    
    tab = open(tab_mag,'r').readlines()
    mags = tab[1].strip().split()
    Alam_Av = map(float,tab[3].strip().split())
    return Alam_Av[mags.index(filter)]
    
def Mag2mag(Mag,filter,photsys,**kwargs):
    '''
    This uses Leo calculations using Cardelli et al 1998 extinction curve
    with Rv = 3.1

    kwargs:
    target, galaxy id to find in angst survey paper table 5
    Av, 0.0
    dmod, 0.0
    '''
    target = kwargs.get('target',None)
    A = 0.
    if target != None:
        filter2 = filter
        filter1 = kwargs.get('filter1',None)
        trgb,Av,dmod = AngstTables.get_tab5_trgb_av_dmod(target, ','.join((filter1,filter2)))
    else:
        Av = kwargs.get('Av',0.0)
        dmod = kwargs.get('dmod',0.)
        
    if Av != 0.0:
        Alam_Av = parse_mag_tab(photsys,filter)
        A = Alam_Av * Av
    
    return Mag+dmod+A

def mag2Mag(mag,filter,photsys,**kwargs):
    '''
    This uses Leo calculations using Cardelli et al 1998 extinction curve
    with Rv = 3.1
    
    kwargs:
    target, galaxy id to find in angst survey paper table 5
    Av, 0.0
    dmod, 0.0
    '''
    
    target = kwargs.get('target',None)
    A = 0.
    if target != None:
        trgb,Av,dmod = get_tab5_trgb_Av_dmod(target)
    else:
        Av = kwargs.get('Av',0.0)
        dmod = kwargs.get('dmod',0.)
        
    if Av != 0.0:
        Alam_Av = parse_mag_tab(photsys,filter)
        A = Alam_Av * Av
    
    return mag-dmod-A