import matplotlib.nxutils as nxutils
import numpy as np
from mpfit import mpfit


class StellarPops(object):
    def __init__(self, stellar_pops):
        self.stellarpops = stellar_pops
        self.galaxies = [sp.galaxy for sp in sps]
        self.cmdregions = [sp.cmdregion for sp in sps]


class StellarPop(object):
    def __init__(self, galaxy_object, CMDregion_object):
        self.galaxy = galaxy_object
        self.cmdregion = CMDregion_object

    def number_inside_region(self, region, slice_inds=None):
        '''
        returns the indices of stars inside a region.
        Allows inds, some slice of the galaxy color or mag2 array to only 
        include.
        Won't write slice to attribue, only region...
        '''
        cmd = np.column_stack((self.galaxy.color, self.galaxy.mag2))
        mask = nxutils.points_inside_poly(cmd, self.cmdregion.regions[region])
        inds, = np.nonzero(mask)
        if slice_inds is not None:
            inds = list(set(inds) & set(slice_inds))
        else:
            self.__setattr__('%s_stars' % region, inds)
        return inds

    def star_range(self, reg, vals=['mag1', 'mag2', 'color']):
        '''
        finds the limits of the stars in specified region.
        Could also do this with Mag1 etc.
        adds attribute region_[val]_range
        should loop over reg??
        '''
        d = {}
        if not hasattr(self, '%s_stars' % reg):
            star_inds = StellarPop.number_inside_region(self, reg)
        else:
            star_inds = self.__dict__['%s_stars' % reg]

        for i in vals:
            arr = self.galaxy.__dict__[i][star_inds]
            lims = (arr.min(), arr.max())
            self.__setattr__('%s_%s_range' % (reg, i), lims)
            d[i] = lims
        return d
        
    def call_mpfit(self, funct, dcol, **kwargs):
        '''
        calls mpfit and saves result as new attribute or in gaussians
        dictionary.

        by default, assumes a single gaussian fit.

        input:
        funct, mpfit function
        p0, initial guesses, see mpfit doc.
        functkw, dict of funct keywords, see mpfit doc.

        color_arr, historgram array
        dcol, historgram bin size
        inds, indices to slice color_arr

        kwargs, default
        err, uniform
        reg, None: region name
        colorlimits, if reg: self.[reg]_color_range
                     else: color_arr min, color_arr max

        new_key, None: name for self.gaussians dictionary key to hold mpfit
                       instance

        returns
        if new_key: self.gaussians[new_key] = mpfit result
        else: self.gaussians = mpfit result
        '''
        inds = kwargs.get('inds')
        if inds is not None:
            color = self.galaxy.color[inds]
        else:
            color = color_arr[:]

        # make a color histogram
        colorlimits = kwargs.get('colorlimits')
        if not colorlimits:
            reg = kwargs.get('reg')
            slice_inds = kwargs.get('slice_inds')
            if reg is not None:
                colorlimits = self.__dict__['%s_color_range' % reg]
            else:
                colorlimits = (color.min(), color.max())

        col_bins = np.arange(colorlimits[0], colorlimits[1] + dcol, dcol)

        functkw = kwargs.get('functkw', {})
        hist_in = functkw
        p0 = kwargs.get('p0')

        if len(functkw) == 0:
            err = kwargs.get('err')
            hist, bins = np.histogram(color, bins=col_bins)
            bins = bins[1:]

            # uniform errors by default
            if not err:
                err = np.zeros(len(bins)) + 1.
            if err == 'weight_by_distance':
                err = np.array([abs(c - np.median(colorlimits)) 
                                for c in bins]) + 1.
            if err == 'weight_by_model':
                pass
            # set up inputs
            hist_in = {'x': bins, 'y': hist, 'err': err}
            if not p0:
                p0 = [np.nanmax(hist), np.mean(col_bins) / 2, dcol]

        mp_dg = mpfit(funct, p0, functkw=hist_in, quiet=True)

        new_key = kwargs.get('new_key')
        new_attr = funct.__name__
        if not new_key:
            self.__setattr__(new_attr, mp_dg)
        else:
            if not hasattr(self, new_attr):
                self.__setattr__(new_attr, {})
            if not new_key in self.__dict__[new_attr]:
                self.__dict__[new_attr][new_key] = {}
            self.__dict__[new_attr][new_key] = {'fit': mp_dg,
                                                'nstars': color.size,
                                                'functkw': hist_in}


# del np, nxutils
