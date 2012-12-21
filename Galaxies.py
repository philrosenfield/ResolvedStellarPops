import ResolvedStellarPops as rsp
import ResolvedStellarPops.graphics.GraphicsUtils as rspgraph
import matplotlib.nxutils as nxutils
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from TrilegalUtils import get_stage_label, get_label_stage
from scatter_contour import scatter_contour
import os
import sys
import numpy as np
import brewer2mpl
import itertools
import logging
logger = logging.getLogger()

angst_data = rsp.angst_tables.AngstTables()
        
class galaxies(object):
    '''
    wrapper for lists of galaxy objects, each method returns lists, unless they
    are setting attributes.
    '''
    def __init__(self, galaxy_objects):
        self.galaxies = galaxy_objects

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
        if not hasattr(self, 'zs'):
            return
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


class star_pop(object):
    def __init__(self):
        pass

    def plot_cmd(self, color, mag, fig=None, ax=None, xlim=None, ylim=None, yfilter=None,
                 contour_args={}, scatter_args={}, plot_args={}, scatter_off=False):
        if fig is None:
            fig = plt.figure(figsize=(8, 8))
        if ax is None:
            ax = plt.axes()
        if xlim is None:
            ax.set_xlim(color.min(), color.max())
        if ylim is None:
            ax.set_ylim(mag.max(), mag.min())
        if yfilter is None:
            yfilter = self.filter2
        if scatter_off is False:
            if len(contour_args) == 0:
                contour_args = {'cmap': cm.gray_r, 'zorder': 100}
            if len(scatter_args) == 0:
                scatter_args = {'marker': '.', 'color': 'black', 'alpha': 0.2,
                                'edgecolors': None, 'zorder': 1}
    
            ncolbin = int(np.diff((np.min(color), np.max(color))) / 0.05)
            nmagbin = int(np.diff((np.min(mag), np.max(mag))) / 0.05)

            plt_pts, cs = scatter_contour(color, mag,
                                          threshold=10, levels=20,
                                          hist_bins=[ncolbin, nmagbin],
                                          contour_args=contour_args,
                                          scatter_args=scatter_args,
                                          ax=ax)
            self.plt_pts = plt_pts
            self.cs = cs
        else:
            ax.plot(color, mag, '.', **plot_args)

        ax.set_xlabel('$%s-%s$' % (self.filter1, self.filter2), fontsize=20)
        ax.set_ylabel('$%s$' % yfilter, fontsize=20)

        self.ax = ax
        self.fig = fig
        return

    def all_stages(self, *stages):
        '''
        adds the indices of some stage as an attribute.
        '''
        for stage in stages:
            i = stage_inds(self.stage, stage)
            self.__setattr__('i%s' % stage.lower(), i)
        return

    def stage_inds(self, stage_name):
        '''
        not so useful on its own, use all_stages to add the inds as attribues.
        '''
        if not hasattr(self, 'stage'):
            logger.warning('no stages marked in this file')
            return
        else:
            inds, = np.nonzero(self.stage == get_stage_label(stage_name))
        return inds
    
    def make_hess(self, binsize, absmag=False, hess_kw = {}):
        '''
        adds a hess diagram of color, mag2 or Color, Mag2. See astronomy_utils
        doc for more information.
        '''
        if absmag:
            if not hasattr(self, 'Color'):
                self.Color = self.Mag1 - self.Mag2
            hess = rsp.astronomy_utils.hess(self.Color, self.Mag2, binsize,
                                            **hess_kw)
            self.Hess = hess
        else:
            if not hasattr(self, 'color'):
                self.color = self.mag1 - self.mag2
            hess = rsp.astronomy_utils.hess(self.color, self.mag2, binsize,
                                            **hess_kw)
            self.hess = hess
        return hess

    def get_header(self):
        '''
        utility for writing data files, sets header attribute and returns
        header string.
        '''
        key_dict = self.data.key_dict
        names = [k[0] for k in sorted(key_dict.items(),
                                      key=lambda (k, v): (v, k))]
        self.header = '# %s' % ' '.join(names)
        return self.header

    def delete_data(self, data_names=None):
        '''
        for wrapper functions, I don't want gigs of data stored when they
        are no longer needed.
        '''
        if data_names is None:
            data_names = ['data', 'mag1', 'mag2', 'color', 'stage', 'ast_mag1',
                          'ast_mag2', 'ast_color', 'rec']
        for data_name in data_names:
            if hasattr(data_name):
                self.__delattr__(data_name)
            if hasattr(data_name.title()):
                self.__delattr__(data_name.title())

    def convert_mag(self, dmod=0., Av=0., target=None, shift_distance=False):
        '''
        convert from mag to Mag or from Mag to mag or just shift distance.

        pass dmod, Av, or use AngstTables to look it up from target.
        shift_distance: for the possibility of doing dmod, Av fitting of model
        to data the key here is that we re-read the mag from the original data
        array.

        Without shift_distance: Just for common usage. If trilegal was given a
        dmod, it will swap it back to Mag, if it was done at dmod=10., will
        shift to given dmod. mag or Mag attributes are set in __init__.
        '''
        if target is not None or hasattr(self, 'target'):
            logger.info('converting distance and Av to match %s' % target)
            self.target = target
            filters = ','.join((self.filter1, self.filter2))
            tad = angst_data.get_tab5_trgb_av_dmod(self.target, filters)
            __, self.Av, self.dmod = tad
        else:
            self.dmod = dmod
            self.Av = Av
        mag_covert_kw = {'Av': self.Av, 'dmod': self.dmod}

        if shift_distance is True:
            m1 = self.data.get_col(self.filter1)
            m2 = self.data.get_col(self.filter2)
            self.mag1 = rsp.astronomy_utils.Mag2mag(m1,
                                                    self.filter1,
                                                    self.photsys,
                                                    **mag_covert_kw)
            self.mag2 = rsp.astronomy_utils.Mag2mag(m2,
                                                    self.filter2,
                                                    self.photsys,
                                                    **mag_covert_kw)
            self.color = self.mag1 - self.mag2
        else:
            if hasattr(self, 'mag1'):
                self.Mag1 = rsp.astronomy_utils.mag2Mag(self.mag1,
                                                        self.filter1,
                                                        self.photsys,
                                                        **mag_covert_kw)
                self.Mag2 = rsp.astronomy_utils.mag2Mag(self.mag2, 
                                                        self.filter2,
                                                        self.photsys,
                                                        **mag_covert_kw)
                self.Color = self.Mag1 - self.Mag2
                if hasattr(self, 'trgb'):
                    self.Trgb = rsp.astronomy_utils.mag2Mag(self.trgb,
                                                            self.filter2,
                                                            self.photsys,
                                                            **mag_covert_kw)

            if hasattr(self, 'Mag1'):
                self.mag1 = rsp.astronomy_utils.Mag2mag(self.Mag1, 
                                                        self.filter1,
                                                        self.photsys,
                                                        **mag_covert_kw)
                self.mag2 = rsp.astronomy_utils.Mag2mag(self.Mag2,
                                                        self.filter2,
                                                        self.photsys,
                                                        **mag_covert_kw)
                self.color = self.mag1 - self.mag2

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


class galaxy(star_pop):
    '''
    angst and angrrr galaxy object. data is a ascii tagged file with stages.
    '''
    def __init__(self, fname, filetype='tagged_phot', **kwargs):
        self.base, self.name = os.path.split(fname)
        star_pop.__init__(self)
        hla = kwargs.get('hla', True)
        angst = kwargs.get('angst', True)
        band = kwargs.get('band')
        # name spaces
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
                logger.warning('assuming this is wfc3 data')
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
        elif filetype == 'match_phot':
            self.data = np.genfromtxt(fname, names=['mag1', 'mag2'])
            self.mag1 = self.data['mag1']
            self.mag2 = self.data['mag2']
        else:
            logger.error('filetype must be fitstable, tagged_phot or match_phot use simgalaxy for trilegal')
            sys.exit(2)

        self.color = self.mag1 - self.mag2
        # angst table loads
        if angst is True:
            self.comp50mag1 = angst_data.get_50compmag(self.target, self.filter1)
            self.comp50mag2 = angst_data.get_50compmag(self.target, self.filter2)
            self.trgb, self.Av, self.dmod = galaxy.trgb_av_dmod(self)
            # Abs mag
            self.convert_mag()
            self.z = galaxy_metallicity(self, self.target, **kwargs)

    def trgb_av_dmod(self):
        '''
        returns trgb, av, dmod from angst table
        '''
        filters = ','.join((self.filter1, self.filter2))
        return angst_data.get_tab5_trgb_av_dmod(self.target, filters)
    
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

class simgalaxy(star_pop):
    '''
    reads a trilegal output catalog
    there is an issue with mags being abs mag or app mag. If dmod == 0, mags
    are assumed to be abs mag and get title() attributes.
    '''
    def __init__(self, trilegal_out, filter1, filter2, photsys=None,
                 count_offset=0.0, table_data=False):
        star_pop.__init__(self)
        if table_data is True:
            self.data = trilegal_out
            self.base = trilegal_out.base
            self.name = trilegal_out.name
        else:
            self.data = rsp.fileIO.read_table(trilegal_out)
            self.base, self.name = os.path.split(trilegal_out)
        self.filter1 = filter1
        self.filter2 = filter2
        self.count_offset = count_offset
        if photsys is None:
            # assume it's the last _item before extension and it's HST baby!!
            self.photsys = self.name.split('_')[-1].split('.')[0]
            if self.photsys != 'wfpc2':
                self.photsys = 'acs_wfc'
        else:
            self.photsys = photsys

        absmag = False
        if self.data.get_col('m-M0')[0] == 0.:
            absmag = True
        if absmag is True:
            self.Mag1 = self.data.get_col(self.filter1)
            self.Mag2 = self.data.get_col(self.filter2)
            self.Color = self.Mag1 - self.Mag2
        else:
            self.mag1 = self.data.get_col(self.filter1)
            self.mag2 = self.data.get_col(self.filter2)
            self.color = self.mag1 - self.mag2
        self.stage = self.data.get_col('stage')

        do_slice = simgalaxy.load_ast_corrections(self)
        if do_slice:
            data_to_slice = ['mag1', 'mag2', 'stage', 'ast_mag1', 'ast_mag2']
            slice_inds = self.rec
            simgalaxy.slice_data(self, data_to_slice, slice_inds)
            self.ast_color = self.ast_mag1 - self.ast_mag2

        #simgalaxy.load_ic_mstar(self)

    def burst_duration(self):
        '''
        for calculating ages of bursts
        '''
        lage = self.data.get_col('logAge')
        self.burst_length, = np.diff((10 ** np.min(lage), 10 ** np.max(lage)))


    def get_fits(self):
        match_out_dir = os.path.join(os.path.split(self.base)[0], 'match',
                                     'output')
        fit_file_name = '%s_%s_%s.fit'%(self.ID, self.mix, self.model_name)
        try:
            fit_file, = rsp.fileIO.get_files(match_out_dir, fit_file_name)
            self.chi2, self.fit = MatchUtils.get_fit(fit_file)
        except ValueError:
            logger.warning('no match output for %s.' % fit_file_name)
        return

    def load_ast_corrections(self):
        try:
            diff1 = self.data.get_col('%s_cor' % self.filter1)
            diff2 = self.data.get_col('%s_cor' % self.filter2)
        except KeyError:
            # there may not be AST corrections... everything is recovered
            logger.warning('no ast corrections.')
            # makes self.rec all inds.
            self.rec = range(len(self.data.get_col('m-M0')))
            return 0
        recovered1, = np.nonzero(abs(diff1) < 90.)
        recovered2, = np.nonzero(abs(diff2) < 90.)
        self.rec = list(set(recovered1) & set(recovered2))
        if hasattr(self, 'mag1'):
            self.ast_mag1 = diff1
            self.ast_mag2 = diff2
            self.ast_color = self.ast_mag1 - self.ast_mag2
        return 1

    def mix_modelname(self,model):
        '''
        give a model, will split into CAF09, modelname
        '''
        self.mix, self.model_name = get_mix_modelname(model)

    def load_ic_mstar(self):
        '''
        separate C and M stars, sets their indicies as attributes: icstar and
        imstar, will include artificial star tests (if there are any).
        M star: C/O <= 1, LogL >= 3.3 Mdot <=-5, and TPAGB flag
        C star: C/O >= 1, Mdot <=-5, and TPAGB flag
        '''
        try:
            co = self.data.get_col('C/O')[self.rec]
        except KeyError:
            logger.warning('no agb stars... trilegal ran w/o -a flag?')
            return

        lage = self.data.get_col('logAge')[self.rec]
        mdot = self.data.get_col('logML')[self.rec]
        logl = self.data.get_col('logL')[self.rec]

        self.imstar, = np.nonzero((co <= 1) &
                                  (logl >= 3.3) &
                                  (mdot <=- 5) &
                                  (self.stage == get_stage_label('TPAGB')))

        self.icstar, = np.nonzero((co >= 1) &
                                  (mdot <= -5) &
                                  (self.stage == get_stage_label('TPAGB')))

    def normalize(self, stage_lab, mag2=None, stage=None, by_stage=True,
                  magcut=999., useasts=False, sinds_cut=None, verts=None,
                  ndata_stars=None):
        '''
        randomly sample stars from a trilegal simulation maintaining their 
        distribution. Sample selection is set either by specific indices of 
        data in some stage or by a CMD polygon based on data.
        
        assumes mag2 either self.mag2 or if useasts self.ast_mag2
        
        INPUT
        stage_lab: the label of the stage, probably 'ms' or 'rgb' it will be
        used to set attribute names

        by_stage:
           mag2, stage: data arrays of filter2 and the tagged stage inds
           if using galaxy object, mag2 = gal.mag2, stage = gal.stage.

        else, by verts: requires verts and ndata_stars
           don't normalize by data stage (e.g., not a tagged file)
           in this case verts should be a cmd polygon and ndata_stars should be
           the number of data stars in that cmd polygon.
        
        magcut: flat mag2 cut to apply (return brighter than)
        useasts: use artificial star test corrections.
        sinds_cut: use a slice of simgalaxy mag2 instead of all (currently 
                   overridden if useasts=True)
        
        RETURNS
        normalization: N_i/N_j -- the lower the fraction the better. Consider
        a larger simulation if normalization is less than 0.75.
            N_i = number of stars in stage or verts
            N_j = number of simulated stars in stage or verts

        inds: random sample of simulated stars < normalization (to slice
        simgalaxy array)
        
        creates attributes: [stage]_norm and [stage]_norm_inds

        Note:
        if there are no simulated stars in the normalization region, will
        return inds = [-1] and norm = 999.
        '''
        if useasts:
            smag2 = self.data.get_col('%s_cor' % self.filter2)
            smag1 = self.data.get_col('%s_cor' % self.filter1)
            scolor = smag1 - smag2
            # ast corrections keep nans and infs to stay the same length as
            # data
            sinds_cut, = np.nonzero(np.isfinite(smag2) & np.isfinite(smag1))
        else:
            smag2 = self.mag2
            scolor = self.mag1 - self.mag2

        new_attr = '%s_norm' % stage_lab.lower()

        self.all_stages(stage_lab)
        ibright, = np.nonzero(smag2 < magcut)
        sinds = list(set(self.__dict__['i%s' % stage_lab]) & set(ibright))
    
        if sinds_cut is not None:
        # inf could be less than magcut, so better keep only finite vals.
            sinds = list(set(sinds) & set(sinds_cut))
        
        if by_stage is False:
            points = np.column_stack((scolor[sinds], smag2[sinds]))
            sinds, = np.nonzero(nxutils.points_inside_poly(points, verts))
        
        nsim_stars = float(len(sinds))    

        if len(sinds) == 0:
            logger.warning('no stars with %s < %.2f' % (new_attr, magcut))
            self.__setattr__('%s_inds' % new_attr, [-1])
            self.__setattr__('%s' % new_attr, 999.)
            return [-1], 999.
    
        if by_stage is True:
            dsinds, = np.nonzero((stage == get_stage_label(stage_lab)) &
                                 (mag2 < magcut))
            assert dsinds.size > 0, 'no data stars in stage %s' % stage_lab
            ndata_stars = float(len(dsinds))

        normalization = ndata_stars / nsim_stars

        # random sample the data distribution
        rands = np.random.random(len(smag2))
        ind, = np.nonzero(rands < normalization)
        self.__setattr__('%s_inds' % new_attr, ind)
        self.__setattr__('%s' % new_attr, normalization)
        return ind, normalization
        
    def diagnostic_cmd(self, trgb=None, figname=None, inds=None, **kwargs):
        '''
        make a cmd of both simulated cmd and ast corrected cmd by stage.
        input
        sgal with ast corrections (will die without TODO!)

        optional input
        trgb 
            plot a line at the trgb mag2 with the number of stars brighter
            than trgb.
        
        figname
            save the plot as figname and figname_spread
        
        inds
            only use some subset of indices
            
        kwargs:
        xlim tuple of plot xlimits
        ylim tuple of plot ylimits
        '''
        if not hasattr(self, 'ast_mag2'):
            logger.error('currently need ast corrections for diagnostic plot...')
            return -1
        if inds is not None:
            ustage = np.unique(self.stage[inds])
        else:
            ustage = np.unique(self.stage)
    
        nplots = ustage.size + 1.
        bcols = brewer2mpl.get_map('Paired', 'qualitative', len(ustage))
        cols = bcols.mpl_colors
        subplots_kwargs = {'sharex': 1, 'sharey': 1, 'figsize': (12, 8)}
        j = 0
        # loop for both simulation and spread simulation
        for color, mag2 in zip((self.color, self.ast_color),
                               (self.mag2, self.ast_mag2)):
            if inds is not None:
                stage = self.stage[inds]
            else:
                stage = self.stage

            fig, (axs) = rspgraph.setup_multiplot(nplots, **subplots_kwargs)

            for ax in axs.ravel():
                xlim = kwargs.get('xlim', (np.min(self.ast_color[self.rec]),
                                           np.max(self.ast_color[self.rec])))
                ylim = kwargs.get('ylim', (np.max(self.mag2),
                                           np.min(self.mag2)))
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

            # first plot is all stars.
            ax0, cols = rspgraph.colorplot_by_stage(axs.ravel()[0],
                                                    color, mag2, '.', stage,
                                                    cols=cols)
            i = 0
            # go through each existing evolutionary phase and plot those stars.
            for ax, st in zip(axs.ravel()[1:], ustage): 
                label = get_label_stage(int(st))
                ind = self.stage_inds(label)
                if inds is not None:
                    ind = list(set(ind) & set(inds))
                if len(ind) == 0:
                    continue
                ax.plot(color[ind], mag2[ind], '.', color=cols[i], mew=0,
                        label='N=%i' % len(ind))
                kwargs['color'] = 'black'
                ax.set_title(label, **{'color': cols[i]})
                i += 1
                ax.legend(loc=1, numpoints=1, frameon=False)
                # add another line and set of numbers brighter than trgb
                if trgb is not None:
                    rspgraph.plot_lines([ax], ax.get_xlim(), trgb)
                    text_offset = 0.02
                    xpos = ax.get_xlim()[0] + 2 * text_offset
                    ypos = trgb - text_offset
                    num = rsp.math_utils.brighter(mag2, trgb - self.count_offset,
                                                  inds=ind).size
                    rspgraph.plot_numbs(ax, num, xpos, ypos, **kwargs)

            if figname:
                import matplotlib.pyplot as plt
                if j == 0: 
                    extra = ''
                else:
                    extra = '_spread'
                plt.savefig(figname.replace('.png', '%s.png' % extra))
                logger.info('wrote %s' % figname.replace('.png', '%s.png' % extra))
                plt.close()
            else:
                plt.show()
            j+=1
        return figname.replace('.png', '%s.png' % extra)


def get_mix_modelname(model):
    '''
    separate the mix and model name
    eg cmd_input_CAF09_COV0.5_ENV0.5.dat => CAF09, COV0.5_ENV0.5
    '''
    mix = model.split('.')[0].split('_')[2]
    model_name = '_'.join(model.split('.')[0].split('_')[3:])
    return mix, model_name


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
    got = 0
    met_table = read_galtable(**kwargs)
    for i, t in enumerate(met_table['Target']):
        if t.lower() in target:
            z = met_table['Z'][i]
            if met_table['Target'][i] != t:
                print 'fuck!!!'
            got = 1
    if got == 0:
        logger.error(target, 'not found')
        z = np.nan
    gal.z = z
    return z


def get_fake(target, fake_loc='.'):
    return rsp.fileIO.get_files(fake_loc, '*%s*.matchfake' % target.upper())[0]


def ast_correct_trilegal_sim(sgal, fake_file, outfile=None, overwrite=False, 
                             spread_too=False, spread_outfile=None, 
                             savefile=False):
    '''
    correct trilegal simulation with artificial star tests.
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
        sgal.fake_file = fake_file
    else:
        asts = fake_file

    if sgal.filter1 != asts.filter1 or sgal.filter2 != asts.filter2:
        logger.error('bad filter match between sim gal and ast.')
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
            logger.warning('%s exists, send overwrite=True arg to overwrite' % outfile)
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
    PID_TARGET_FILTER1_FILTER2_... or TARGET_FILTER1_FILTER2_
    this is how attributes target, filter1, and filter2 are assigned.
    
    '''
    def __init__(self, filename):
        self.base, self.name = os.path.split(filename)
        try:
            __, self.target, self.filter1, self.filter2, __ = self.name.split('_')
        except:
            self.target, self.filter1, filter2 = self.name.split('_')
            self.filter2 = filter2.split('.')[0]
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
            logger.error('mag arrays of different lengths')
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

