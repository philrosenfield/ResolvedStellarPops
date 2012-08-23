from ResolvedStellarPops import angst_tables
from ResolvedStellarPops import fileIO
from ResolvedStellarPops import astronomy_utils
from TrilegalUtils import get_stage_label
import os
import sys
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
        else:
            print ('filetype must be fitstable or tagged_phot,',
                    'use simgalaxy for trileagl.')
            sys.exit(2)

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

        self.z = galaxy_metallicity(self, self.target, **kwargs)
        
    def hess(self, binsize, absmag=False, hess_kw = {}):
        '''
        adds a hess diagram of color, mag2 or Color, Mag2. See astronomy_utils
        doc for more information.
        '''
        if absmag:
            hess = astronomy_utils.hess(self.Color, self.Mag2, binsize,
                                        **hess_kw)
            self.Hess = hess 
        else:
            hess = astronomy_utils.hess(self.color, self.mag2, binsize,
                                        **hess_kw)
            self.hess = hess
        return hess

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

    def cut_mag_inds(self, mag2cut, mag1cut=None):
        '''
        a simple function to return indices of magX that are brighter than magXcut.
        '''
        mag2cut, = np.nonzero(self.mag2 <= mag2cut)
        if mag1cut is not None:
            mag1cut, = np.nonzero(self.mag1 <= mag1cut)
            cuts = list(set(mag1cut) & set(mag2cut))
        else:
            cuts = mag2cut
        return cuts

                
class simgalaxy(object):
    '''
    reads a trilegal output catalog 
    there is an issue with mags being abs mag or app mag. If dmod == 0, mags 
    are assumed to be abs mag and get title() attributes. 
    '''
    def __init__(self, trilegal_out, filter1, filter2, photsys=None):
        self.base, self.name = os.path.split(trilegal_out)
        self.data = fileIO.read_table(trilegal_out)
        self.filter1 = filter1
        self.filter2 = filter2
        if photsys is None:
            # assume it's the last _item before extension.
            self.photsys = self.name.split('_')[-1].split('.')[0]
        else:
            self.photsys = photsys
        #self.target = self.name.split('_')[2]
        if self.data.get_col('m-M0')[0] == 0.: 
            absmag = True
        if absmag:
            self.Mag1 = self.data.get_col(self.filter1)
            self.Mag2 = self.data.get_col(self.filter2)
        else:
            self.mag1 = self.data.get_col(self.filter1)
            self.mag2 = self.data.get_col(self.filter2)

        self.stage = self.data.get_col('stage')

        do_slice = simgalaxy.load_ast_corrections(self)
        if do_slice:
            data_to_slice = ['mag1', 'mag2', 'stage', 'ast_mag1', 'ast_mag2']
            slice_inds = self.rec
            simgalaxy.slice_data(self, data_to_slice, slice_inds)    
            self.ast_color = self.ast_mag1 - self.ast_mag2

        if not absmag:
            self.color = self.mag1 - self.mag2
        else:
            self.Color = self.Mag1 - self.Mag2
        simgalaxy.load_ic_mstar(self)
        
    def get_fits(self):
        match_out_dir = os.path.join(os.path.split(self.base)[0], 'match', 'output')
        fit_file_name = '%s_%s_%s.fit'%(self.ID, self.mix, self.model_name)
        try:
            fit_file, = fileIO.get_files(match_out_dir, fit_file_name)
            self.chi2, self.fit = MatchUtils.get_fit(fit_file)
        except ValueError:
            print 'no match output for %s.' % fit_file_name
        return
        
    def load_ast_corrections(self):
        try:
            diff1 = self.data.get_col('diff_' + self.filter1)
            diff2 = self.data.get_col('diff_' + self.filter2)
        except KeyError:
            # there may not be AST corrections... everything is recovered
            self.rec = range(len(self.data.get_col('m-M0')))
            return 0
        recovered1, = np.nonzero(abs(diff1) < 90.)
        recovered2, = np.nonzero(abs(diff2) < 90.)
        self.rec = list(set(recovered1) & set(recovered2))
        if hasattr(self, 'mag1'):
            self.ast_mag1 = self.mag1 + diff1
            self.ast_mag2 = self.mag2 + diff2
        else:
            self.ast_Mag1 = self.Mag1 + diff1
            self.ast_Mag2 = self.Mag2 + diff2
        return 1

    def slice_data(self, data_to_slice, slice_inds):
        '''
        slice already set attributes by some index list.
        '''
        for d in data_to_slice:
            if hasattr(self, d):
                self.__setattr__(d, self.__dict__[d][slice_inds])
            if hasattr(self, d.title()):
                d = d.title()
                self.__setattr__(d, self.__dict__[d][slice_inds])

    def mix_modelname(self,model):
        '''
        give a model, will split into CAF09, modelname
        '''
        self.mix, self.model_name = get_mix_modelname(model)
    
    def delete_data(self):
        '''
        for wrapper functions, I don't want gigs of data stored when they
        are no longer needed.
        '''
        data_names = ['data', 'mag1', 'mag2', 'color', 'stage', 'ast_mag1',
                      'ast_mag2', 'ast_color', 'rec']
        for data_name in data_names:
            if hasattr(data_name):
                self.__delattr__(data_name)
            if hasattr(data_name.title()):
                self.__delattr__(data_name.title())

    def stage_inds(self, stage_name):
        return np.nonzero(self.stage == get_stage_label(stage_name))[0]

    def load_ic_mstar(self):
        co = self.data.get_col('C/O')[self.rec]
        lage = self.data.get_col('logAge')[self.rec]
        mdot = self.data.get_col('logML')[self.rec]
        logl = self.data.get_col('logL')[self.rec]
        
        self.imstar, = np.nonzero((co <= 1) & 
                                  (logl >= 3.3) & 
                                  (mdot<=-5) & 
                                  (self.stage == get_stage_label('TPAGB')))
                                  
        self.icstar, = np.nonzero((co >= 1) & 
                                  (mdot <= -5) & 
                                  (self.stage == get_stage_label('TPAGB')))
    
    def all_stages(self, *stages):
        '''
        adds the indices of some stage as an attribute.
        '''
        for stage in stages:
            i = stage_inds(self.stage, stage)
            self.__setattr__('i%s'%stage.lower(), i)
        return
    
    def convert_mag(self, dmod=0., Av=0., target=None):
        '''
        convert from mag to Mag or from Mag to mag, whichever self doesn't
        already have an attribute.
        
        pass dmod, Av, or use AngstTables to look it up from target.
        '''
        if target is not None:
            self.target = target
            filters = ','.join((self.filter1, self.filter2))
            tad = angst_data.get_tab5_trgb_av_dmod(self.target, filters)
            __, self.Av, self.dmod = tad
        else:
            self.dmod = dmod
            self.Av = Av
        mag_covert_kw = {'Av': self.Av, 'dmod': self.dmod}

        if hasattr(self, 'mag1'):
            self.Mag1 = astronomy_utils.mag2Mag(self.mag1, self.filter1,
                                                self.photsys, **mag_covert_kw)
            self.Mag2 = astronomy_utils.mag2Mag(self.mag2, self.filter2,
                                                self.photsys, **mag_covert_kw)
        elif hasattr(self, 'Mag1'):
            self.mag1 = astronomy_utils.Mag2mag(self.Mag1, self.filter1,
                                                self.photsys, **mag_covert_kw)
            self.mag2 = astronomy_utils.Mag2mag(self.Mag2, self.filter2,
                                                self.photsys, **mag_covert_kw)

def get_mix_modelname(model):
    '''
    separate the mix and model name
    eg cmd_input_CAF09_COV0.5_ENV0.5.dat => CAF09, COV0.5_ENV0.5
    '''
    mix = model.split('.')[0].split('_')[2]
    model_name = '_'.join(model.split('.')[0].split('_')[3:])
    return mix, model_name


def stage_inds(stage, label):
    import TrilegalUtils
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


def spread_cmd(gal, ast_file, hess_kw = {}, **kwargs):
    '''
    not finished...
    '''
    mag1lim = kwargs.get('mag1lim', self.comp50mag1)
    mag2lim = kwargs.get('mag2lim', self.comp50mag2)
    colorlimits = kwargs.get('colorlimits')

    # default input mags, will be corrected if star is not recovered
    amag1, amag2, dmag1, dmag2 = np.loadtxt(ast_file, unpack=True)
    ast_mag1 = amag1 + dmag1
    ast_mag2 = amag2 + dmag2

    # unrecovered stars: stores only input mags
    if dmag1 > 9.98:
        ast_mag1 = amag1
    if dmag2 > 9.98:
        ast_mag2 = amag2


def get_fake(target, fake_loc='.'):
    return fileIO.get_files(fake_loc, '*%s*.matchfake' % target.upper())[0]

#del angst_tables, fileIO, astronomy_utils, os, np
