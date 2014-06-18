'''
wrapper for lists of galaxy objects, each method returns lists, unless they
are setting attributes.
'''
import numpy as np
import itertools

from .starpop import StarPop

__all__ = ['Galaxies']


class Galaxies(StarPop):
    '''
    wrapper for lists of galaxy objects, each method returns lists, unless they
    are setting attributes.
    '''
    def __init__(self, galaxy_objects):
        self.galaxies = np.asarray(galaxy_objects)
        self.filter1s = np.unique([g.filter1 for g in galaxy_objects])
        self.filter2s = np.unique([g.filter2 for g in galaxy_objects])

    def sum_attr(self, *attrs):
        for attr, g in itertools.product(attrs, self.galaxies):
            g.__setattr__('sum_%s' % attr, np.sum(g.data.get_col(attr)))

    def all_stages(self, *stages):
        '''
        adds the indices of any stage as attributes to galaxy.
        If the stage isn't found, -1 is returned.
        '''
        [g.all_stages(*stages) for g in self.galaxies]
        return

    def squish(self, *attrs, **kwargs):
        '''
        concatenates an attribute or many attributes and adds them to galaxies
        instance -- with an 's' at the end to pluralize them... that might
        be stupid.
        ex
        for gal in gals:
            gal.ra = gal.data['ra']
            gal.dec = gal.data['dec']
        gs =  Galaxies.galaxies(gals)
        gs.squish('color', 'mag2', 'ra', 'dec')
        gs.ras ...
        '''
        inds = kwargs.get('inds', np.arange(len(self.galaxies)))
        new_attrs = kwargs.get('new_attrs', None)

        if new_attrs is not None:
            assert len(new_attrs) == len(attrs), \
                'new attribute titles must be list same length as given attributes.'

        for i, attr in enumerate(attrs):
            # do we have a name for the new attribute?
            if new_attrs is not None:
                new_attr = new_attrs[i]
            else:
                new_attr = '%ss' % attr

            new_list = [g.__getattribute__(attr) for g in self.galaxies[inds]]
            # is attr an array of arrays, or is it now an array?
            try:
                new_val = np.concatenate(new_list)
            except ValueError:
                new_val = np.array(new_list)

            self.__setattr__(new_attr, new_val)

    def finite_key(self, key):
        return [g for g in self.galaxies if np.isfinite(g.__dict__[key])]

    def select_on_key(self, key, val):
        ''' ex filter2 == F814W works great with strings or exact g.key==val.
        rounds z to four places, no error handling.
        '''
        key = key.lower()
        if key == 'z':
            gs = [g for g in self.galaxies if
                  np.round(g.__dict__[key], 4) == val]
        else:
            gs = [g for g in self.galaxies if g.__dict__[key] == val]
        return gs

    def group_by_z(self):
        if not hasattr(self, 'zs'):
            return
        zsf = self.zs[np.isfinite(self.zs)]

        d = {}
        for z in zsf:
            key = 'Z%.4f' % z
            d[key] = self.select_on_key('z', z)
        d['no z'] = [g for g in self.galaxies if np.isnan(g.z)]
        return d

    def intersection(self, **kwargs):
        '''
        ex kwargs = {'filter2':'F814W', 'filter1':'F555W'}
        will return a list of galaxy objects that match all kwarg values.
        '''
        gs_tmp = self.galaxies
        gs = [self.select_on_key(k, v) for k, v in kwargs.items()]
        for i in range(len(gs)):
            gs_tmp = list(set(gs_tmp) & set(gs[i]))
        return gs_tmp
