import matplotlib.nxutils as nxutils
import numpy as np

class StellarPop(object):
    def __init__(self,galaxy_object,CMDregion_object):
        self.galaxy = galaxy_object
        self.cmdregion = CMDregion_object
        
    def number_inside_region(self, region):
        cmd = np.column_stack((self.galaxy.color, self.galaxy.mag2))
        inds, = np.nonzero(nxutils.points_inside_poly(cmd, self.cmdregion.regions[region]))
        self.__setattr__('%s_stars'%region, inds)
        return inds
    
    def star_range(self, reg, vals = ['mag1', 'mag2', 'color']):
        '''
        finds the limits of the stars in specified region.
        Could also do this with Mag1 etc.
        adds attribute region_[val]_range
        should loop over reg??
        '''
        if not hasattr(self, '%s_stars'%reg):
            inds = StellarPop.number_inside_region(self, reg)
        else:
            inds = self.__dict__['%s_stars'%reg]
        
        for i in vals:
            arr = self.galaxy.__dict__[i][inds]
            self.__setattr__('%s_%s_range'%(reg,i), (arr.min(), arr.max()))
            
            