import numpy as np
import sys

def convertz(z=None, oh=None, mh=None, feh=None, oh_sun=8.76, z_sun=0.01524,
             y0=.2485, dy_dz=1.80):
    '''
    input:
    metallicity as z
    [O/H] as oh
    [M/H] as mh
    [Fe/H] as feh
    
    initial args can be oh_sun, z_sun, y0, and dy_dz
    
    returns oh, z, y, x, feh, mh where y = He and X = H mass fractions
    '''

    if oh is not None:
        feh = oh - oh_sun
        z = z_sun * 10 **(feh)

    if mh is not None:
        z = (1 - y0) / ((10**(-1. * mh) / 0.0207) + (1. + dy_dz))

    if z is not None:
        feh = np.log10(z / z_sun)
    
    if feh is not None:
        z = z_sun * 10**feh

    oh = feh + oh_sun
    y = y0 + dy_dz * z
    x = 1. - z - y
    if mh is None:
        mh = np.log10((z / x) / 0.0207)
    
    if __name__ == "__main__":
        print '''
                 [O/H] = %2f
                 z = %.4f
                 y = %.4f
                 x = %.4f
                 [Fe/H] = %.4f
                 [M/H] = %.4f''' % (oh, z, y, x, feh, mh)
    return oh, z, y, x, feh, mh
    
if __name__=="__main__":
    '''
    ex:
    python convertz z 0.001
    python convertz oh 7.80
    python convertz mh 1.50
    '''
    try:
        element = sys.argv[1]
        val = float(sys.argv[2])
    except:
        print "__main__".__doc__
    convertz(**{element:val})
