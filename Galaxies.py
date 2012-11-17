import ResolvedStellarPops as rsp
from TrilegalUtils import get_stage_label
import os
import sys
import numpy as np

angst_data = rsp.angst_tables.AngstTables()

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


def hla_galaxy_info(filename):
    name = os.path.split(filename)[1]
    name_split = name.split('_')[1:-2]
    survey, lixo, photsys, pidtarget, filters = name_split
    propid = pidtarget.split('-')[0]
    target = '-'.join(pidtarget.split('-')[1:])
    return survey, propid, target, filters, photsys

def bens_fmt_galaxy_info(filename):
    info = os.path.split(filename)[1].split('.')[0].split('_')
    # Why not just split? Sometimes there's an IR right in there for
    # reasons beyond comprehension.
    (propid, target) = info[:2]
    (filter1, filter2) = info[-2:]
    return propid, target, filter1, filter2

class galaxy(object):
    '''
    angst and angrrr galaxy object. data is a ascii tagged file with stages.
    '''
    def __init__(self, fname, filetype='tagged_phot', **kwargs):
        hla = kwargs.get('hla', True)
        angst = kwargs.get('angst', True)
        band = kwargs.get('band')
        # name spaces
        self.base, self.name = os.path.split(fname)
        if hla is True:
            self.survey, self.propid, self.target, filts, psys = hla_galaxy_info(fname)
            # photometry
            self.filter1, self.filter2 = filts.upper().split('-')
            self.photsys = psys.replace('-', '_')
        else:
            self.survey = ' '
            self.dmod = kwargs.get('dmod', 0.)
            self.Av = kwargs.get('Av', 0.)
            self.z = kwargs.get('z', -99)
            self.trgb = kwargs.get('trgb', np.nan)
            self.photsys = kwargs.get('photsys')
            if self.photsys is None: 
                self.photsys = 'wfc3'
                print 'assuming this is wfc3 data'
            self.propid, self.target, self.filter1, self.filter2 = bens_fmt_galaxy_info(fname)

        if filetype == 'fitstable':
            self.data = rsp.fileIO.read_fits(fname)
            ext = self.photsys.upper().split('_')[0]
            if band is not None:
                ext = band.upper()
            self.mag1 = self.data['mag1_%s' % ext]
            self.mag2 = self.data['mag2_%s' % ext]
        elif filetype == 'tagged_phot':
            self.data = rsp.fileIO.read_tagged_phot(fname)
            self.mag1 = self.data['mag1']
            self.mag2 = self.data['mag2']
            self.stage = self.data['stage']
        else:
            print ('filetype must be fitstable or tagged_phot,',
                    'use simgalaxy for trilegal.')
            sys.exit(2)

        self.color = self.mag1 - self.mag2
        # angst table loads
        if angst is True:
            self.comp50mag1 = angst_data.get_50compmag(self.target, self.filter1)
            self.comp50mag2 = angst_data.get_50compmag(self.target, self.filter2)
            self.trgb, self.Av, self.dmod = galaxy.trgb_av_dmod(self)
            # Abs mag
            self.AbsMag()
            self.z = galaxy_metallicity(self, self.target, **kwargs)

    def AbsMag(self):
        mag2Mag_kwargs = {'Av': self.Av, 'dmod': self.dmod}
        print mag2Mag_kwargs
        self.Mag1 = rsp.astronomy_utils.mag2Mag(self.mag1, self.filter1,
                                                self.photsys, **mag2Mag_kwargs)
        self.Mag2 = rsp.astronomy_utils.mag2Mag(self.mag2, self.filter2,
                                                self.photsys, **mag2Mag_kwargs)
        self.Color = self.Mag1 - self.Mag2
        self.Trgb = rsp.astronomy_utils.mag2Mag(self.trgb, self.filter2,
                                                self.photsys, **mag2Mag_kwargs)

    def hess(self, binsize, absmag=False, hess_kw = {}):
        '''
        adds a hess diagram of color, mag2 or Color, Mag2. See astronomy_utils
        doc for more information.
        '''
        if absmag:
            hess = rsp.astronomy_utils.hess(self.Color, self.Mag2, binsize,
                                            **hess_kw)
            self.Hess = hess
        else:
            hess = rsp.astronomy_utils.hess(self.color, self.mag2, binsize,
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

    def stage_inds(self, stage_name):
        return get_stage_inds(self.data, stage_name)

    def delete_data(self):
        data_names = ['data', 'mag1', 'mag2', 'color', 'stage']
        [self.__delattr__(data_name) for data_name in data_names]
    
    def __str__(self):
        out = (
            "%s data: \n"
            "   Prop ID: %s\n"
            "   Target: %s\n"
            "   dmod: %g\n"
            "   Av: %g\n"
            "   Filters: %s - %s\n"
            "   Camera: %s\n"
            "   Z: %.4f") % (
                self.survey, self.propid, self.target, self.dmod, self.Av,
                self.filter1, self.filter2, self.photsys, self.z)
        return out

    def cut_mag_inds(self, mag2cut, mag1cut=None):
        '''
        a simple function to return indices of magX that are brighter than
        magXcut.
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
    def __init__(self, trilegal_out, filter1, filter2, photsys=None, count_offset=0.0):
        self.base, self.name = os.path.split(trilegal_out)
        self.data = rsp.fileIO.read_table(trilegal_out)
        self.filter1 = filter1
        self.filter2 = filter2
        self.count_offset = count_offset
        if photsys is None:
            # assume it's the last _item before extension.
            self.photsys = self.name.split('_')[-1].split('.')[0]
            if self.photsys != 'wfpc2':
                self.photsys = 'acs_wfc'
        else:
            self.photsys = photsys
        #self.target = self.name.split('_')[2]
        absmag = False
        if self.data.get_col('m-M0')[0] == 0.:
            absmag = True
        if absmag is True:
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
        match_out_dir = os.path.join(os.path.split(self.base)[0], 'match',
                                     'output')
        fit_file_name = '%s_%s_%s.fit'%(self.ID, self.mix, self.model_name)
        try:
            fit_file, = rsp.fileIO.get_files(match_out_dir, fit_file_name)
            self.chi2, self.fit = MatchUtils.get_fit(fit_file)
        except ValueError:
            print 'no match output for %s.' % fit_file_name
        return

    def load_ast_corrections(self):
        try:
            diff1 = self.data.get_col('%s_cor' % self.filter1)
            diff2 = self.data.get_col('%s_cor' % self.filter2)
        except KeyError:
            # there may not be AST corrections... everything is recovered
            self.rec = range(len(self.data.get_col('m-M0')))
            return 0
        recovered1, = np.nonzero(abs(diff1) < 90.)
        recovered2, = np.nonzero(abs(diff2) < 90.)
        self.rec = list(set(recovered1) & set(recovered2))
        if hasattr(self, 'mag1'):
            self.ast_mag1 = diff1
            self.ast_mag2 = diff2
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
        try:
            co = self.data.get_col('C/O')[self.rec]
        except KeyError:
            print 'warning, no agb stars'
            return
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
        mag_covert_kw = {'Av': self.Av, 'dmod': self.dmod-1.4}

        if hasattr(self, 'mag1'):
            self.Mag1 = rsp.astronomy_utils.mag2Mag(self.mag1, self.filter1,
                                                    self.photsys,
                                                    **mag_covert_kw)
            self.Mag2 = rsp.astronomy_utils.mag2Mag(self.mag2, self.filter2,
                                                    self.photsys,
                                                    **mag_covert_kw)
        elif hasattr(self, 'Mag1'):
            self.mag1 = rsp.astronomy_utils.Mag2mag(self.Mag1, self.filter1,
                                                    self.photsys,
                                                    **mag_covert_kw)
            self.mag2 = rsp.astronomy_utils.Mag2mag(self.Mag2, self.filter2,
                                                    self.photsys,
                                                    **mag_covert_kw)
    
    def get_header(self):
        key_dict = self.data.key_dict
        names = [k[0] for k in sorted(key_dict.items(),
                                      key=lambda (k, v): (v, k))]
        self.header = '# %s' % ' '.join(names)
        return self.header

    def add_data(self, **new_cols):
        '''
        add columns to data
        new_cols: {new_key: new_vals}
        new_vals must have same number of rows as data.
        Ie, be same length as self.data.shape[0]

        adds new data to self.data.data_array and self.data.key_dict
        returns new header string (or -1 if nrows != len(new_vals))
        '''

        data = self.data.data_array.copy()
        nrows = data.shape[0]
        ncols = data.shape[1]
        # new arrays must be equal length as the data
        len_test = np.array([len(v) == nrows
                            for v in new_cols.values()]).prod()
        if not len_test:
            'array lengths are not the same.'
            return -1

        header = self.get_header()
        # add new columns to the data and their names to the header.
        for k,v in new_cols.items():
            header += ' %s' % k
            data = np.column_stack((data, v))

        # update self.data
        self.data.data_array = data
        col_keys =  header.replace('#','').split()
        self.data.key_dict = dict(zip(col_keys, range(len(col_keys))))
        return header

    def normalize_by_stage(self, mag2, stage, stage_lab, magcut=999.,
                           useasts=False, sinds_cut=None):
        '''
        this could be in some simgalaxy/galaxy class. just sayin.
        returns inds, normalization

        input
        assumes self.mag2
        mag2, stage: data arrays of filter2 and the tagged stage
        stage_lab: the label of the stage, probably 'ms' or 'rgb'
        magcut:
        normalization: N_i/N_j
        N_i = number of stars in stage == stage_lab (brighter than magcut)
        N_j = number of simulated stars in stage == stage_lab
        (brighter than magcut)

        inds: random sample of simulated stars < normalization

        mag2 and stage are from observational data.
        '''
        smag2 = self.mag2
        if useasts:
            smag2 = self.data.get_col('%s_cor' % self.filter2)
            # ast corrections keep nans and infs to stay the same length as
            # data
            sinds_cut, = np.nonzero(np.isfinite(smag2))
        new_attr = '%s_norm' % stage_lab
        stage_lab = get_stage_label(stage_lab)

        sinds, = np.nonzero((self.stage == stage_lab) & (smag2 < magcut))
        if len(sinds) == 0:
            print 'no stars with %s < %.2f' % (new_attr, magcut)

        if sinds_cut is not None:
            # inf could be less than magcut, so better keep only finite vals.
            sinds = list(set(sinds) & set(sinds_cut))
        dsinds, = np.nonzero((stage == stage_lab) & (mag2 < magcut))
        normalization = float(len(dsinds)) / float(len(sinds))

        # random sample the data distribution
        rands = np.random.random(len(smag2))
        ind, = np.nonzero(rands < normalization)
        self.__setattr__('%s_inds' % new_attr, ind)
        self.__setattr__('%s' % new_attr, normalization)
        return ind, normalization

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
    return np.nonzero(stage == get_stage_label(label))[0]


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
    return rsp.fileIO.get_files(fake_loc, '*%s*.matchfake' % target.upper())[0]


def ast_correct_trileagl_sim(sgal, fake_file, outfile=None, overwrite=False, 
                             spread_too=False, spread_outfile=None, 
                             savefile=False):
    '''
    convolve trilegal simulation with artificial star tests.
    options to write the a copy of the trilegal simulation with the corrected
    mag columns, and also just the color, mag into a file (spread_too). 
    
    input:
    sgal: simgalaxy instance with convert_mag method already called.
    fake_file: string - matchfake file, or an artifical_star_tests instance.
    outfile: string - ast_file to write
    overwrite: overwrite output files, is also passed to write_spread
    spread_too: call write_spread flag
    spread_outfile: outfile for write_spread

    returns
    adds corrected mags to sgal.data.data_array and updates sgal.data.key_dict
    '''

    if type(fake_file) is str:
        asts = artificial_star_tests(fake_file)
    else:
        asts = fake_file

    if sgal.filter1 != asts.filter1 or sgal.filter2 != asts.filter2:
        print 'bad filter match between sim gal and ast.'
        return -1

    cor_mag1, cor_mag2 = asts.ast_correction(sgal.mag1, sgal.mag2, 
                                             **{'binsize': 0.2})
    new_cols = {'%s_cor' % asts.filter1: cor_mag1, 
                '%s_cor' % asts.filter2: cor_mag2}
    sgal.add_data(**new_cols)

    if savefile is True: 
        if not outfile:
            outfile_name = sgal.name.replace('out', 'ast')
            outfile_base = os.path.join(os.path.split(sgal.base)[0], 'AST')
            rsp.fileIO.ensure_dir(outfile_base)
            outfile = os.path.join(outfile_base, outfile_name)
        if overwrite or not os.path.isfile(outfile):
            write_trilegal_sim(sgal, outfile)
        else:
            print '%s exists, send overwrite=True arg to overwrite' % outfile
    if spread_too:
        write_spread(sgal, outfile=spread_outfile, overwrite=overwrite)

    return


class artificial_star_tests(object):
    '''
    class for reading and using artificial stars. 
    If *filename* is not a matchfake file, MUST create a new method to read in
    artificial stars.

    for load_fake:
    mag1 is assumed to be mag1in
    mag2 is assumed to be mag2in
    mag1diff is assumed to be mag1in-mag1out
    mag2diff is assumed to be mag2in-mag2out

    filename is assumed as matchfake:
    PID_TARGET_FILTER1_FILTER2_....
    this is how attributes target, filter1, and filter2 are assigned.
    
    '''
    def __init__(self, filename):
        self.base, self.name = os.path.split(filename)
        __, self.target, self.filter1, self.filter2, __ = self.name.split('_')
        artificial_star_tests.load_fake(self, filename)

    def recovered(self, threshold=9.99):
        '''
        indicies of recovered stars in both filters.
        threshold of recovery [9.99]
        '''
        rec1, = np.nonzero(self.mag1diff > threshold)
        rec2, = np.nonzero(self.mag2diff > threshold)
        self.rec = list(set(rec1) & set(rec2))
        return rec

    def load_fake(self, filename):
        '''
        reads matchfake file and assigns each column to its own attribute
        see artificial_star_tests.__doc__
        '''
        names = ['mag1', 'mag2', 'mag1diff', 'mag2diff']
        self.data = np.genfromtxt(filename, names = names)
        # unpack into attribues
        for name in names:
            self.__setattr__(name, self.data[name])
    
    def bin_asts(self, binsize=0.2, bins=None):
        '''
        bins the artificial star tests in bins of *binsize* or with *bins*
        assigns attributes am1_inds and am2_inds, the indices of the bins to
        which each value in mag1 and mag2 belong (see np.digitize).
        '''
        if bins is None:
            ast_max = np.max(np.concatenate((self.mag1, self.mag2)))
            ast_min = np.min(np.concatenate((self.mag1, self.mag2)))
            self.ast_bins = np.arange(ast_min, ast_max, binsize)
        else:
            self.ast_bins = bins
        
        self.am1_inds = np.digitize(self.mag1, self.ast_bins)
        self.am2_inds = np.digitize(self.mag2, self.ast_bins)

    def _random_select(self, arr, nselections):
        '''
        randomly sample *arr* *nselections* times, used in ast_correction
    
        input
        arr: array or list to sample
        nselections: int times to sample
        returns
        rands: list of selections, length nselections
        '''
        import random
        rands = [random.choice(arr) for i in range(nselections)]
        return rands
    
    def ast_correction(self, obs_mag1, obs_mag2, **kwargs):
        '''
        apply ast correction to input mags.
        This is done by going through obs_mag1 in bins of *bin_asts* (which
        will be called if not already) and randomly selecting magdiff values
        in that ast_bin. obs_mag2 simply follows along since it is tied to
        obs_mag1.

        Random selection was chosen because of the spatial nature of artificial
        star tests. If there are 400 asts in one mag bin, and 30 are not
        recovered, random selection should match that distribution (if there
        are many obs stars).
        input: 
        obs_mag1, obs_mag2: N, 1 arrays
        kwargs:
        binsize, bins: for bin_asts if not already run. 

        TODO:
        possibly return magXdiff rather than magX + magXdiff?
        reason not to: using AST results from one filter to another isn't
        kosher. At least not glatt kosher.
        '''
        nstars = obs_mag1.size
        if obs_mag1.size != obs_mag2.size:
            print 'mag arrays of different lengths'
            return -1

        # corrected mags are filled with nan.
        cor_mag1 = np.empty(nstars)
        cor_mag1.fill(np.nan)
        cor_mag2 = np.empty(nstars)
        cor_mag2.fill(np.nan)

        # need asts to be binned for this method.
        if not hasattr(self, 'ast_bins'):
            binsize = kwargs.get('binsize', 0.2)
            bins = kwargs.get('bins')
            self.bin_asts(binsize=binsize, bins=bins)


        om1_inds = np.digitize(obs_mag1, self.ast_bins)

        for i in range(len(self.ast_bins)):
            obsbin, = np.nonzero(om1_inds == i)
            astbin, = np.nonzero(self.am1_inds == i)
            nobs = len(obsbin)
            nast = len(astbin)
            if nobs == 0:
                # no stars in this mag bin to correct
                continue
            if nast == 0:
                # no asts in this bin, probably means the simulation
                # is too deep
                continue

            # randomly select the appropriate ast correction for obs stars
            # in this bin
            cor1 = artificial_star_tests._random_select(self, 
                                                        self.mag1diff[astbin], 
                                                        nobs)
            cor2 = artificial_star_tests._random_select(self, 
                                                        self.mag2diff[astbin], 
                                                        nobs)

            # apply corrections
            cor_mag1[obsbin] = obs_mag1[obsbin] + cor1
            cor_mag2[obsbin] = obs_mag2[obsbin] + cor2

            # finite values only: not implemented because trilegal array should
            # maintain the same size.
            #fin1, = np.nonzero(np.isfinite(cor_mag1))
            #fin2, = np.nonzero(np.isfinite(cor_mag2))
            #fin = list(set(fin1) & set(fin2))
        return cor_mag1, cor_mag2

