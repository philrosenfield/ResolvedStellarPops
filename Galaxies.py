from ResolvedStellarPops import angst_tables
from ResolvedStellarPops import fileIO
from ResolvedStellarPops import astronomy_utils
import os
import numpy as np

angst_data = angst_tables.AngstTables()


class galaxies(object):
    '''
    wrapper for lists of galaxy objects, each method returns lists, unless they
    are setting attributes.
    '''
    def __init__(self, galaxy_objects):
        self.galaxies = galaxy_objects
        self.zs = np.unique([np.round(g.z, 4) for g in self.galaxies])
        self.filter1s = np.unique(g.filter1 for g in self.galaxies)
        self.filter2s = np.unique(g.filter2 for g in self.galaxies)
        self.photsyss = np.unique(g.photsys for g in self.galaxies)

    def all_stages(self, *stages):
        '''
        adds the indices of any stage as attributes to galaxy.
        If the stage isn't found, -1 is returned.
        '''
        [g.all_stages(*stages) for g in self.galaxies]
        return

    def squish(self, *attrs):
        '''
        concatenates an attribute or many attributes and adds them to galaxies
        instance.
        No slicing, so not sure how it will be useful besides Color Mag2.
        '''
        for attr in attrs:
            new_list = [g.__getattribute__(attr) for g in self.galaxies]
            new_val = np.concatenate(new_list)
            self.__setattr__('%ss' % attr, new_val)

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
        zsf = self.zs[np.isfinite(self.zs)]
        zsn = self.zs[np.isnan(self.zs)]
        d = {}
        for z in zsf:
            key = 'Z%.4f' % z
            d[key] = galaxies.select_on_key(self, 'z', z)

        d['no z'] = [g for g in gals if np.isnan(g.z)]
        return d

    def intersection(self, **kwargs):
        '''
        ex kwargs = {'filter2':'F814W', 'filter1':'F555W'}
        will return a list of galaxy objects that match all kwarg values.
        '''
        gs_tmp = self.galaxies
        gs = [galaxies.select_on_key(self, k, v) for k, v in kwargs.items()]
        for i in range(len(gs)):
            gs_tmp = list(set(gs_tmp) & set(gs[i]))
        return gs_tmp

    def __str__(self):
        for g in self.galaxies:
            print g.__str__()
        return ''


def galaxy_info(filename):
    name = os.path.split(filename)[1]
    name_split = name.split('_')[1:-2]
    survey, lixo, photsys, pidtarget, filters = name_split
    propid = pidtarget.split('-')[0]
    target = '-'.join(pidtarget.split('-')[1:])
    return survey, propid, target, filters, photsys


class galaxy(object):
    '''
    angst and angrrr galaxy object. data is a ascii tagged file with stages.
    '''
    def __init__(self, fname, filetype='tagged_phot', **kwargs):
        # name spaces
        self.base, self.name = os.path.split(fname)
        self.survey, self.propid, self.target, filts, psys = galaxy_info(fname)
        # photometry
        self.filter1, self.filter2 = filts.upper().split('-')
        self.photsys = psys.replace('-', '_')

        if filetype == 'fitstable':
            self.data = fileIO.read_fits(fname)
            self.mag1 = self.data['mag1_%s' % photsys.split('-')[0]]
            self.mag2 = self.data['mag2_%s' % photsys.split('-')[0]]
        elif filetype == 'tagged_phot':
            self.data = fileIO.read_tagged_phot(fname)
            self.mag1 = self.data['mag1']
            self.mag2 = self.data['mag2']
            self.stage = self.data['stage']
        elif filetype == 'trilegal':
            print 'left off here!!'

        self.color = self.mag1 - self.mag2
        # angst table loads
        self.comp50mag1 = angst_data.get_50compmag(self.target, self.filter1)
        self.comp50mag2 = angst_data.get_50compmag(self.target, self.filter2)
        self.trgb, self.Av, self.dmod = galaxy.trgb_av_dmod(self)
        # Abs mag
        mag2Mag_kwargs = {'Av': self.Av, 'dmod': self.dmod}
        self.Mag1 = astronomy_utils.mag2Mag(self.mag1, self.filter1,
                                            self.photsys, **mag2Mag_kwargs)
        self.Mag2 = astronomy_utils.mag2Mag(self.mag2, self.filter2,
                                            self.photsys, **mag2Mag_kwargs)
        self.Color = self.Mag1 - self.Mag2
        self.Trgb = astronomy_utils.mag2Mag(self.trgb, self.filter2,
                                            self.photsys, **mag2Mag_kwargs)
        # etc
        # this is now done in br_fileIO for lack of a better place.
        # how should this be done for both agb and br?
        self.z = galaxy_metallicity(self, self.target, **kwargs)

    def trgb_av_dmod(self):
        '''
        returns trgb, av, dmod from angst table
        '''
        filters = ','.join((self.filter1, self.filter2))
        return angst_data.get_tab5_trgb_av_dmod(self.target, filters)

    def all_stages(self, *stages):
        '''
        adds the indices of some stage as an attribute.
        '''
        for stage in stages:
            i = stage_inds(self.stage, stage)
            self.__setattr__('i%s' % stage.lower(), i)
        return

    def __str__(self):
        out = (
            "%s data: "
            "   Prop ID: %s"
            "   Target: %s"
            "   dmod: %g"
            "   Av: %g"
            "   Filters: %s - %s"
            "   Camera: %s"
            "   Z: %.4f") % (
                self.survey, self.propid, self.target, self.dmod, self.Av,
                self.filter1, self.filter2, self.photsys, self.z)
        return


def stage_inds(stage, label):
    return np.nonzero(stage == TrilegalUtils.get_stage_label(label))[0]


def read_galtable(**kwargs):
    fname = kwargs.get('filename')
    br = kwargs.get('br')
    tpagb = kwargs.get('tpagb')
    if not fname:
        if br:
            fname = '/Users/phil/research/BRratio/tables/brratio_galtable.dat'
            dtype = [('Target', '|S10'), ('O/H', '<f8'), ('Z', '<f8')]
            kwargs = {'autostrip': 1, 'delimiter': ',', 'dtype': dtype}
        if tpagb:
            pass
    return np.genfromtxt(fname, **kwargs)


def galaxy_metallicity(gal, target, **kwargs):
    '''
    add metallicity to galaxy object.
    '''
    got = 0
    met_table = read_galtable(**kwargs)
    for i, t in enumerate(met_table['Target']):
        if t.lower() in target:
            z = met_table['Z'][i]
            if met_table['Target'][i] != t:
                print 'fuck!!!'
            got = 1
    if got == 0:
        print target, 'not found'
        z = np.nan
    gal.z = z
    return z

#del angst_tables, fileIO, astronomy_utils, os, np
