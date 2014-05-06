from __future__ import print_function
import os
import copy
import numpy as np
import pylab as plt
from matplotlib.ticker import NullFormatter, MaxNLocator, MultipleLocator

from .. import fileio
from .. import match
from .. import trilegal
from .. import graphics
from .. import math
from .starpop import StarPop

__all__ = ['SimGalaxy', 'SimAndGal', 'get_mix_modelname']


class SimGalaxy(StarPop):
    '''
    reads a trilegal output catalog there is an issue with mags being abs mag
    or app mag. If dmod == 0, mags are assumed to be abs mag and get title()
    attributes.
    '''
    def __init__(self, trilegal_out, filter1, filter2, photsys=None,
                 count_offset=0.0, table_data=False):

        StarPop.__init__(self)

        if table_data is True:
            self.data = trilegal_out
            self.base = trilegal_out.base
            self.name = trilegal_out.name
        else:
            self.data = io.read_table(trilegal_out)
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

        try:
            dmod = self.data.get_col('m-M0')[0]
        except KeyError:
            dmod = np.nan
        if dmod == 0.:
            absmag = True
        if absmag is True:
            self.Mag1 = self.data.get_col(self.filter1)
            self.Mag2 = self.data.get_col(self.filter2)
            self.Color = self.Mag1 - self.Mag2
        else:
            self.mag1 = self.data.get_col(self.filter1)
            self.mag2 = self.data.get_col(self.filter2)
            self.color = self.mag1 - self.mag2
        try:
            self.stage = self.data.get_col('stage')
        except KeyError:
            self.stage = np.zeros(len(self.data.get_col('Gc'))) - 1.

        # what do these in the init?!?
        #do_slice = simgalaxy.load_ast_corrections(self)
        #if do_slice:
        #    data_to_slice = ['mag1', 'mag2', 'stage', 'ast_mag1', 'ast_mag2']
        #    slice_inds = self.rec
        #    simgalaxy.slice_data(self, data_to_slice, slice_inds)
        #    self.ast_color = self.ast_mag1 - self.ast_mag2
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
        fit_file_name = '%s_%s_%s.fit' % (self.ID, self.mix, self.model_name)
        fit_file, =  io.get_files(match_out_dir, fit_file_name)
        self.chi2, self.fit = match.get_fit(fit_file)

    def load_ast_corrections(self):
        try:
            diff1 = self.data.get_col('%s_cor' % self.filter1)
            diff2 = self.data.get_col('%s_cor' % self.filter2)

            if self.get_header().count('cor') > 3:
                self.filter3, self.filter4 = self.get_header().replace('_cor', '').split()[-2:]
                diff3 = self.data.get_col('%s_cor' % self.filter3)
                diff4 = self.data.get_col('%s_cor' % self.filter4)

        except KeyError:
            # there may not be AST corrections... everything is recovered
            print('no ast corrections. Everything recovered?')
            # makes self.rec all inds.
            self.rec = range(len(self.data.get_col('m-M0')))
            return 0
        recovered1, = np.nonzero(abs(diff1) < 90.)
        recovered2, = np.nonzero(abs(diff2) < 90.)
        if hasattr(self, 'filter3'):
            recovered3, = np.nonzero(abs(diff3) < 90.)
            recovered4, = np.nonzero(abs(diff4) < 90.)
            self.ir_rec = list(set(recovered3) & set(recovered4))
            self.ast_mag3 = diff3
            self.ast_mag4 = diff4
            self.rec3 = recovered3
            self.rec4 = recovered4

        self.rec = list(set(recovered1) & set(recovered2))
        if hasattr(self, 'mag1'):
            self.ast_mag1 = diff1
            self.ast_mag2 = diff2
            self.ast_color = self.ast_mag1 - self.ast_mag2

        self.rec1 = recovered1
        self.rec2 = recovered2
        return 1

    def mix_modelname(self, model):
        '''
        give a model, will split into CAF09, modelname
        '''
        self.mix, self.model_name = get_mix_modelname(model)

    def load_ic_mstar(self):
        '''
        separate C and M stars, sets their indicies as attributes: icstar and
        imstar, will include artificial star tests (if there are any).

        Trilegal simulation must have been done with -l and -a flags.

        This is done using self.rec meaning use should be e.g.:
        self.ast_mag2[self.rec][self.icstar]

        Hard coded:
        M star: C/O <= 1, LogL >= 3.3 Mdot <=-5, and TPAGB flag
        C star: C/O >= 1, Mdot <=-5, and TPAGB flag
        '''
        if not hasattr(self, 'rec'):
            self.rec = np.arange(len(self.data.get_col('C/O')))
        try:
            co = self.data.get_col('C/O')[self.rec]
        except KeyError as e:
            print('no agb stars... trilegal ran w/o -a flag?')
            raise e

        mdot = self.data.get_col('logML')[self.rec]
        logl = self.data.get_col('logL')[self.rec]
        self.imstar, = np.nonzero((co <= 1) & (logl >= 3.3) & (mdot <= -5) &
                                  (self.stage[self.rec] == trilegal.get_stage_label('TPAGB')))

        self.icstar, = np.nonzero((co >= 1) & (mdot <= -5) &
                                  (self.stage[self.rec] == trilegal.get_stage_label('TPAGB')))

    def normalize(self, stage_lab, filt1, filt2, mag=None, stage=None, by_stage=True,
                  magcut=999., useasts=False, verts=None, ndata_stars=-np.inf,
                  sinds_cut=None):
        '''
        randomly sample stars from a trilegal simulation maintaining their
        distribution. Sample selection is set either by specific indices of
        data in some stage or by a CMD polygon based on data.

        assumes mag2 either self.mag2 or if useasts self.ast_mag2

        INPUT
        stage_lab: the label of the stage, probably 'ms' or 'rgb' it will be
        used to set attribute names
        by_stage: DO NOT USE THIS
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

        new_attr = '%s_norm' % stage_lab.lower()
        if hasattr(self, 'filter3'):
            filters = [self.filter1, self.filter2, self.filter3, self.filter4]
        else:
            filters = [self.filter1, self.filter2]

        f1 = filters.index(filt1) + 1
        f2 = filters.index(filt2) + 1
        if useasts is True:
            if not hasattr(self, 'ast_mag2'):
                self.load_ast_corrections()
            smag1 = self.__getattribute__('ast_mag%i' % f1)
            rec1 = self.__getattribute__('rec%i' % f1)
            smag = self.__getattribute__('ast_mag%i' % f2)
            rec2 = self.__getattribute__('rec%i' % f2)
            rec = list(set(rec1) & set(rec2))
            scolor = smag1[rec] - smag[rec]
            smag = smag[rec]
            self.norm_rec = rec
        else:
            smag1 = self.__getattribute__('mag%i' % f1)
            smag = self.__getattribute__('mag%i' % f2)
            scolor = smag1 - smag
            rec = np.arange(smag.size)

        # initialize slices, they will just be the enitre array if not
        # changed below
        ibright = np.arange(smag.size)
        st_inds = np.arange(smag.size)
        reg_inds = np.arange(smag.size)
        if sinds_cut is None:
            sinds_cut = np.arange(smag.size)

        # mag cut
        ibright, = np.nonzero(smag < magcut)

        # stage cut
        if by_stage is True:
            istage = 'i%s' % stage_lab
            if not hasattr('self', istage):
                self.all_stages(stage_lab)
            st_inds = self.__dict__[istage]

        # cmd space cut
        if verts is not None:
            points = np.column_stack((scolor, smag))
            reg_inds, = np.nonzero(math.points_inside_poly(points, verts))

        # combine all inds.
        sinds = list(set(ibright) & set(st_inds) & set(reg_inds) &
                     set(sinds_cut))

        nsim_stars = float(len(sinds))

        if len(sinds) == 0:
            print('no stars with %s < %.2f' % (new_attr, magcut))
            self.__setattr__('%s_inds' % new_attr, [-1])
            self.__setattr__('%s' % new_attr, 999.)
            return [-1], 999.

        # find inds for normalization
        if by_stage is True:
            assert ndata_stars is -np.inf, \
                'error! with by_stage=True, ndata_stars will be derived'
            dsinds, = np.nonzero((stage == trilegal.get_stage_label(stage_lab)) & (mag < magcut))

            if (dsinds.size > 0):
                raise ValueError('no data stars in stage %s' % stage_lab)

            ndata_stars = float(len(dsinds))

        normalization = ndata_stars / nsim_stars
        # random sample the data distribution
        rands = np.random.random(len(smag))
        ind, = np.nonzero(rands < normalization)
        if hasattr(self, 'filter3'):
            # insert the band in the attribute names so they are not
            # overwritten.
            if '814' in [filt1, filt2]:
                extra = 'opt'
            if '160' in [filt1, filt2]:
                extra = 'nir'
            self.__setattr__('%s_nsim_stars' % extra, nsim_stars)
            self.__setattr__('%s_%s_inds' % (extra, new_attr), np.array(rec)[ind])
            self.__setattr__('%s_%s' % (extra, new_attr), normalization)
        else:
            self.nsim_stars = nsim_stars
            self.__setattr__('%s_inds' % new_attr, np.array(rec)[ind])
            self.__setattr__('%s' % new_attr, normalization)
        return ind, normalization

    def diagnostic_cmd(self, trgb=None, figname=None, inds=None, **kwargs):
        '''
        make a cmd of both simulated cmd and ast corrected cmd by stage.
        input

        will make diagnostic plots depending on attributes available.
        ast_mag2, mag2, Mag2. See cmd_by_stage

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
        xlim = kwargs.get('xlim')
        ylim = kwargs.get('ylim')
        if hasattr(self, 'ast_mag2'):
            extra = '_spread'
            color = self.ast_color
            mag2 = self.ast_mag2
            if xlim is None:
                xlim = (np.min(color[self.rec]), np.max(color[self.rec]))
            if ylim is None:
                ylim = (np.max(self.mag2), np.min(self.mag2))
            self.cmd_by_stage(color, mag2, inds=inds, extra=extra, xlim=xlim,
                              ylim=ylim, trgb=trgb)
        if hasattr(self, 'mag2'):
            extra = ''
            color = self.color
            mag2 = self.mag2
            if xlim is None:
                xlim = (np.min(color), np.max(color))
            if ylim is None:
                ylim = (np.max(self.mag2), np.min(self.mag2))
            self.cmd_by_stage(color, mag2, inds=inds, extra=extra, xlim=xlim,
                              ylim=ylim, trgb=trgb)
        if hasattr(self, 'Mag2'):
            extra = '_abs'
            color = self.Color
            mag2 = self.Mag2
            if xlim is None:
                xlim = (np.min(color), np.max(color))
            if ylim is None:
                ylim = (np.max(mag2), np.min(mag2))
            self.cmd_by_stage(color, mag2, inds=inds, extra=extra, xlim=xlim,
                              ylim=ylim, trgb=trgb)
        return

    def cmd_by_stage(self, color, mag2, inds=None, xlim=None, ylim=None,
                     extra='', figname=None, trgb=None, cmap=None, **kwargs):
        '''
        made to be called from diagnostic plots. Make a panel of plots for
        each stage (get_label_stage)
        '''
        stage = self.stage
        if inds is not None:
            stage = self.stage[inds]
        ustage = np.unique(stage)

        fig, (axs) = graphics.setup_plot_by_stage(self.stage, inds=inds)
        # first plot is summary.

        ax0, cols = graphics.colorplot_by_stage(axs.ravel()[0], color, mag2,
                                                '.', stage, cmap=cmap)
        # go through each existing evolutionary phase and plot those stars.
        for i, (ax, st) in enumerate(zip(axs.ravel()[1:], ustage)):
            label = trilegal.get_label_stage(int(st))
            ind = self.stage_inds(label)
            if inds is not None:
                ind = list(set(ind) & set(inds))
            if len(ind) == 0:
                continue
            ax.plot(color[ind], mag2[ind], '.', color=cols[i], mew=0,
                    label='N=%i' % len(ind))
            ax.set_xlabel('$%s-%s$' % (self.filter1, self.filter2))
            ax.set_ylabel('$%s$' % (self.filter2))
            ax.set_title(label, **{'color': cols[i]})
            ax.legend(loc=1, numpoints=1, frameon=False)
            # add another line and set of numbers brighter than trgb
            kwargs['color'] = 'black'
            if trgb is not None:
                xlim = ax.get_xlim()
                graphics.plot_hlines(trgb, xlim[0], xlim[1], ax=ax)
                text_offset = 0.02
                xpos = xlim[0] + 2 * text_offset
                ypos = trgb - text_offset
                num = math.brighter(mag2, trgb - self.count_offset, inds=ind).size
                graphics.plot_numbs(ax, num, xpos, ypos, **kwargs)

        if xlim is not None and ylim is not None:
            for ax in axs.ravel():
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
        if figname is None:
            figname = io.replace_ext(self.name, '.png')

        figname = figname.replace('.png', '%s.png' % extra)
        plt.savefig(figname)

    def hist_by_attr(self, attr, bins=10, stage=None, slice_inds=None):
        '''
        builds a histogram of some attribute (mass, logl, etc)
        slice_inds will cut the full array, and stage will limit to only that stage.
        '''
        data = self.data.get_col(attr)
        if stage is not None:
            istage_s = 'i%s' % stage.lower()
            if not hasattr(self, istage_s):
                self.all_stages(stage.lower())
            istage = self.__dict__[istage_s]
        else:
            istage = np.arange(data.size)

        if slice_inds is None:
            slice_inds = np.arange(data.size)

        inds = list(set(istage) & set(slice_inds))

        hist, bins = np.histogram(data[inds], bins=bins)

        return hist, bins

    def all_stages(self, *stages):
        '''
        adds the indices of some stage as an attribute.
        '''
        if stages is ():
            stages = ['PMS', 'MS', 'SUBGIANT', 'RGB', 'HEB', 'RHEB', 'BHEB',
                      'EAGB', 'TPAGB', 'POSTAGB', 'WD']
        for stage in stages:
            i = self.stage_inds(stage)
            self.__setattr__('i%s' % stage.lower(), i)
        return

    def stage_inds(self, stage_name):
        '''
        not so useful on its own, use all_stages to add the inds as attribues.
        '''
        assert hasattr(self, 'stage'), 'no stages marked in this file'
        inds, = np.nonzero(self.stage == trilegal.utils.get_stage_label(stage_name))
        return inds

class SimAndGal(object):
    def __init__(self, galaxy, simgalaxy):
        self.gal = galaxy
        self.sgal = simgalaxy
        if hasattr(self.gal, 'maglims'):
            self.maglims = self.gal.maglims
        else:
            self.maglims = [90., 90.]

        if not hasattr(self.sgal, 'norm_inds'):
            self.sgal.norm_inds = np.arange(len(self.sgal.data.data_array))

    def make_mini_hess(self, color, mag, scolor, smag, ax=None, **kwargs):

        hess_kw = {'binsize': 0.1, 'cbinsize': 0.05}
        hess_kw.update(**kwargs)

        self.gal_hess = graphics.make_hess(color, mag, **hess_kw)
        hess_kw['cbin'] = self.gal_hess[0]
        hess_kw['mbin'] = self.gal_hess[1]
        self.sgal_hess = graphics.make_hess(scolor, smag, **hess_kw)
        comp_hess = copy.deepcopy(self.gal_hess)
        comp_hess = comp_hess[:-1] + ((self.gal_hess[-1] - self.sgal_hess[-1]),)
        self.comp_hess = comp_hess

    def nrgb_nagb(self, band=None, agb_verts=None):
        '''
        this is the real deal, man.
        '''
        self.sgal.nbrighter = []
        self.gal.nbrighter = []

        if band is None and hasattr(self.gal, 'mag4'):
            '''
            four filter catalogs.
            '''
            mag = self.gal.mag4
            smag = self.sgal.ast_mag4
        else:
            smag = self.sgal.ast_mag2[self.sgal.norm_inds]
            scolor = self.sgal.ast_color[self.sgal.norm_inds]
            mag = self.gal.mag2
            color = self.gal.color

        spoints = np.column_stack((scolor, smag))
        if agb_verts is not None:
            points = np.column_stack((color, mag))
            ginds, = np.nonzero(math.points_inside_poly(points, agb_verts))
            sinds, = np.nonzero(math.points_inside_poly(spoints, agb_verts))
        else:
            sinds = None
            ginds = None

        # the number of rgb stars used for normalization
        # this is not set in this file! WTH PHIL? W T H?
        try:
            self.gal.nbrighter.append(len(self.gal.rgb_norm_inds))
        except TypeError:
            self.gal.nbrighter.append(self.gal.rgb_norm_inds)

        # the number of data stars in the agb_verts polygon
        self.gal.nbrighter.append(len(ginds))

        # the number of sim stars in the rgb box set by data verts
        srgb_norm, = np.nonzero(math.points_inside_poly(spoints, self.gal.norm_verts))

        self.sgal.nbrighter.append(len(srgb_norm))

        # the number of sim stars in the agb_verts polygon
        self.sgal.nbrighter.append(len(sinds))

        nrgb_nagb_data = float(self.gal.nbrighter[1]) / float(self.gal.nbrighter[0])
        nrgb_nagb_sim = float(self.sgal.nbrighter[1]) / float(self.sgal.nbrighter[0])
        self.agb_verts = agb_verts
        return nrgb_nagb_data, nrgb_nagb_sim

    def _make_LF(self, filt1, filt2, res=0.1, plt_dir=None, plot_LF_kw={},
                 comp50=False, add_boxes=True, color_hist=False,
                 plot_tpagb=False, figname=None):
        itpagb = None
        f1 = self.gal.filters.index(filt1) + 1
        f2 = self.gal.filters.index(filt2) + 1
        mag1 = self.gal.__getattribute__('mag%i' % f1)
        mag = self.gal.__getattribute__('mag%i' % f2)
        color = mag1 - mag
        if comp50 is True:
            self.gal.rec, = np.nonzero((mag1 < self.gal.__getattribute__('comp50mag%i' % f1)) &
                                       (mag < self.gal.__getattribute__('comp50mag%i' % f2)))
        else:
            self.gal.rec = np.arange(len(mag1))

        self.nbins = (mag[self.gal.rec].max() - mag[self.gal.rec].min()) / res
        self.nbins = np.int(np.sqrt(len(self.gal.rec)))
        self.gal_hist, self.bins = np.histogram(mag[self.gal.rec], self.nbins)

        # using all ast recovered stars for the histogram and normalizing
        # by a multiplicative factor.
        #self.sgal_hist, _ = np.histogram(smag[self.sgal.norm_rec],
        #                                 bins=self.bins)
        #self.sgal_hist *= self.sgal.rgb_norm
        # hist of what's plotted (cmd)
        try:
            smag1 = self.sgal.__getattribute__('ast_mag%i' % f1)
            smag = self.sgal.__getattribute__('ast_mag%i' % f2)
        except AttributeError:
            print('not using ASTs!!')
            smag1 = self.sgal.__getattribute__('mag%i' % f1)
            smag = self.sgal.__getattribute__('mag%i' % f2)

            if self.maglims >= 90.:
                inds = list(set(np.nonzero(smag1 < self.maglims[0])[0]) &
                            set(np.nonzero(smag < self.maglims[1])[0]))
                smag1 = smag1[inds]
                smag = smag[inds]

        scolor = smag1 - smag

        if plot_tpagb is True:
            self.sgal.all_stages('TPAGB')
            itpagb = np.intersect1d(self.sgal.itpagb, self.sgal.norm_inds)
            assert np.unique(self.sgal.stage[itpagb]).size == 1, 'Indexing Error'

        if len(self.sgal.norm_inds) < len(smag):
            self.sgal_hist, _ = np.histogram(smag[self.sgal.rec], bins=self.bins)
            self.sgal_hist *= self.sgal.rgb_norm
        else:
            # lmc, eg, doesn't need normalization.
            self.sgal_hist, _ = np.histogram(smag, bins=self.bins)
            if plot_tpagb is True:
                self.sgal_tpagb_hist, _ = np.histogram(smag[self.sgal.itpagb], bins=self.bins)

        if color_hist is True:
            # this is a colored histogram of cmd plotted. (otherwise use self.sgal.norm_rec)
            maglim = self.maglims[1]
            iabove, = np.nonzero(mag < maglim)
            siabove, = np.nonzero(smag < maglim)

            nbins = (color.max() - color.min()) / 0.01

            self.gal_color_hist, self.color_bins = np.histogram(color[iabove],
                                                                bins=nbins)
            norm_siabove = list(set(siabove) & set(self.sgal.norm_inds))
            scolor_above = scolor[norm_siabove]
            smag_above = smag[norm_siabove]

            self.sgal_color_hist, _ = np.histogram(scolor_above, bins=self.color_bins)

            # add the tpagb star color hist if option chosen
            if itpagb is not None:
                self.sgal_tpagb_color_hist, _ = np.histogram(scolor[itpagb], bins=self.color_bins)

            self.make_mini_hess(color[iabove], mag[iabove], scolor_above,
                                smag_above)

            mass_bins = np.insert(np.arange(1, 6.5, 0.5), 0, 0.8)
            mass_hist, _ = self.sgal.hist_by_attr('m_ini', bins=mass_bins,
                                                  slice_inds=norm_siabove)

            tp_mass_hist, _ = self.sgal.hist_by_attr('m_ini', bins=mass_bins,
                                                     slice_inds=itpagb)
            self.mass_hist = mass_hist
            self.tp_mass_hist = tp_mass_hist
            self.mass_bins = mass_bins

        assert hasattr(self.sgal, 'norm_inds'), 'need norm_inds!'

        _plot_LF_kw = {'model_plt_color': 'red', 'data_plt_color': 'black',
                       'color_hist': color_hist}
        _plot_LF_kw.update( **plot_LF_kw)

        #itpagb, = np.nonzero(self.sgal.stage[self.sgal.norm_inds] == 8)
        fig, axs, top_axs = self._plot_LF(color, mag,
                                          scolor[self.sgal.norm_inds],
                                          smag[self.sgal.norm_inds], filt1,
                                          filt2, itpagb=itpagb,
                                          gal_hist=self.gal_hist,
                                          bins=self.bins,
                                          sgal_hist=self.sgal_hist,
                                          **_plot_LF_kw)
        # not working not sure why, just hacking...
        #if self.maglims < 99.:
        fig, axs = self.add_lines_LF(fig, axs)

        if add_boxes is True:
            if hasattr(self, 'agb_verts'):
                axs[1].plot(self.agb_verts[:, 0], self.agb_verts[:, 1],
                            color='black', lw=1)
                axs[0].plot(self.agb_verts[:, 0], self.agb_verts[:, 1],
                            color='red', lw=1)
            axs[1].plot(self.gal.norm_verts[:, 0], self.gal.norm_verts[:, 1],
                        color='black', lw=1)
            axs[0].plot(self.gal.norm_verts[:, 0], self.gal.norm_verts[:, 1],
                        color='red', lw=1)
        if figname is None:
            figname = '_'.join((self.gal.target, self.sgal.mix,
                                self.sgal.model_name, filt1,
                                filt2, 'LF.png'))

        if plt_dir is not None:
            figname = os.path.join(plt_dir, figname)

        plt.savefig(figname, dpi=300, bbox_to_inches='tight')
        print('wrote %s' % figname)
        return fig, axs, top_axs

    def add_lines_LF(self, fig, axs):
        '''
        must have attributes sgal, gal nbrighter
        '''
        if not 'nbrighter' in self.gal.__dict__.keys():
            self.gal.nbrighter = self.nbrighter

        for i, maglim in enumerate(self.maglims):
            # lines and numbers on plots

            line_on_it_kw = {'annotate': 0, 'ls': '-'}
            for ax, col, g in zip(axs[:2],
                                  (self.gal.data_plt_color, self.gal.model_plt_color),
                                  (self.gal, self.sgal)):
                    self.gal.put_a_line_on_it(ax, maglim, color=col,
                                              **line_on_it_kw)
                    self.gal.annotate_cmd(ax, maglim, '$%i$' % g.nbrighter[i],
                                          text_kw={'color': col})
        return fig, axs

    def _plot_LF(self, color, mag, scolor, smag, filt1, filt2,
                 model_plt_color='red', data_plt_color='black', ylim=None,
                 xlim=None, xlim2=None, model_title='Model', title=False,
                 band='opt', color_hist=False, itpagb=None, gal_hist=None,
                 bins=None, sgal_hist=None, sbins=None):

        def make_title(self, fig, band='opt'):

            if band == 'opt':
                trgb = self.trgb
            elif band == 'ir':
                trgb = self.ir_trgb

            text_kwargs = {'ha': 'center', 'va': 'top', 'size': 20}
            title = '$m_{TRGB}=%.3f$' % trgb

            if np.isfinite(self.z):
                title += ' $Z=%.4f$' % (self.z)

            fig.text(0.5, 0.96, title, **text_kwargs)

        def setup_lfplot(self, filt1, filt2, model_title='Model',
                         color_hist=False, lab_kw={}):

            fig = plt.figure(figsize=(9, 9))
            # plot limits determined by hand
            if color_hist is False:
                bottom, height = 0.1, 0.8
            else:
                bottom, height = 0.1, 0.6

            widths = [0.29, 0.28, 0.22]
            lefts = [0.1, 0.42, 0.73]

            axs = [plt.axes([lefts[i], bottom, widths[i], height])
                   for i in range(3)]

            if color_hist is False:
                top_axs = []
            else:
                bottom, height = 0.7, 0.2
                top_axs = [plt.axes([lefts[i], bottom, widths[i], height])
                           for i in range(3)]

            lab_kw = dict({'fontsize': 20}.items() + lab_kw.items())

            # titles
            axs[0].set_title('$%s$' % self.target,
                             color=self.data_plt_color, **lab_kw)

            axs[1].set_title('$%s$' % model_title, color=self.model_plt_color,
                             **lab_kw)

            axs[0].set_xlabel('$%s-%s$' % (filt1, filt2),
                              **lab_kw)

            axs[0].set_ylabel('$%s$' % filt2, **lab_kw)
            axs[1].set_xlabel(axs[0].get_xlabel(), **lab_kw)
            axs[2].set_xlabel('$\#$', **lab_kw)

            for ax in axs:
                ax.tick_params(labelsize=20)

            return fig, axs, top_axs

        def fix_plot(axs, xlim=None, xlim2=None, ylim=None, top_axs=[]):
            # fix axes
            if xlim is not None:
                axs[0].set_xlim(xlim)
            if ylim is not None:
                axs[0].set_ylim(ylim)
            if xlim2 is not None:
                axs[2].set_xlim(xlim2)

            for ax in axs[:2]:
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.xaxis.set_minor_locator(MultipleLocator(0.2))

            for ax in axs:
                ax.set_ylim(axs[0].get_ylim())
                ax.yaxis.set_major_locator(MultipleLocator(2))
                ax.yaxis.set_minor_locator(MultipleLocator(0.5))

            axs[1].set_xlim(axs[0].get_xlim())

            # no formatters on mid and right plots
            [ax.yaxis.set_major_formatter(NullFormatter()) for ax in axs[1:]]

            if len(top_axs) > 0:
                top_axs[-1].set_xlim(axs[0].get_xlim())
                top_axs[0].xaxis.set_major_formatter(NullFormatter())
                top_axs[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
                top_axs[-1].xaxis.set_minor_locator(MultipleLocator(0.2))
                top_axs[-1].set_ylim(axs[0].get_ylim())
                top_axs[-1].yaxis.set_major_locator(MultipleLocator(2))
                top_axs[-1].yaxis.set_minor_locator(MultipleLocator(0.5))
                top_axs[-1].set_ylim(self.comp_hess[1].max(), self.comp_hess[1].min())
                top_axs[-1].set_xlim(self.comp_hess[0].min(), self.comp_hess[0].max())

        if gal_hist is None:
            gal_hist = self.gal_hist
        if bins is None:
            bins = self.bins
        if sgal_hist is None:
            sgal_hist = self.sgal_hist
        if sbins is None:
            sbins = bins
        self.data_plt_color = data_plt_color
        self.model_plt_color = model_plt_color

        fig, axs, top_axs = setup_lfplot(self, filt1, filt2,
                                         model_title=model_title,
                                         color_hist=color_hist)

        if title is True:
            make_title(self, fig, band=band)

        plt_kw = {'threshold': 25, 'levels': 3, 'scatter_off': True,
                  'filter1': filt1, 'filter2': filt2,
                  'plot_args': {'alpha': 1, 'color': self.data_plt_color,
                                'mew': 0, 'mec': self.data_plt_color}}

        # plot data
        self.plot_cmd(color, mag, ax=axs[0], **plt_kw)

        # plot simulation
        plt_kw['plot_args']['color'] = self.model_plt_color
        plt_kw['plot_args']['mec'] = self.model_plt_color

        self.plot_cmd(scolor, smag, ax=axs[1], **plt_kw)

        if itpagb is not None:
            print('doing itpagb')
            plt_kw['plot_args']['color'] = 'royalblue'
            plt_kw['plot_args']['mec'] = 'royalblue'
            self.plot_cmd(scolor[itpagb], smag[itpagb], ax=axs[1], **plt_kw)
        axs[1].set_ylabel('')

        # plot histogram
        hist_kw = {'drawstyle': 'steps', 'color': self.data_plt_color, 'lw': 2}

        axs[2].semilogx(gal_hist, bins[1:], **hist_kw)

        if color_hist is True:
            top_axs[0].plot(self.color_bins[1:], self.gal_color_hist, **hist_kw)

        hist_kw['color'] = self.model_plt_color
        axs[2].semilogx(sgal_hist, sbins[1:], **hist_kw)

        if itpagb is not None:
            hist_kw['color'] = 'royalblue'
            if hasattr(self, 'sgal_tpagb_hist'):
                axs[2].semilogx(self.sgal_tpagb_hist, self.bins[1:], **hist_kw)

        if color_hist is True:
            hist_kw['color'] = 'royalblue'
            top_axs[0].plot(self.color_bins[1:], self.sgal_tpagb_color_hist,
                            **hist_kw)
            top_axs[0].set_ylabel('$\#$', fontsize=16)
            top_axs[1].semilogy(self.mass_bins[1:], self.tp_mass_hist, **hist_kw)
            top_axs[1].set_xlabel('$M_\odot$', fontsize=16)
            black2red = graphics.stitch_cmap(plt.cm.Reds_r, plt.cm.Greys,
                                             stitch_frac=0.555, dfrac=0.05)

            graphics.hess_plot(self.comp_hess, ax=top_axs[2], imshow=True,
                               cmap=black2red, interpolation='nearest',
                               aspect='equal', norm=None,
                               vmax=np.abs(self.comp_hess[-1]).max(),
                               vmin=-np.abs(self.comp_hess[-1]).max())

        fix_plot(axs, xlim=xlim, xlim2=xlim2, ylim=ylim, top_axs=top_axs)

        return fig, axs, top_axs


def get_mix_modelname(model):
    '''
    separate the mix and model name
    eg cmd_input_CAF09_COV0.5_ENV0.5.dat => CAF09, COV0.5_ENV0.5
    '''
    mix = model.split('.')[0].split('_')[2]
    model_name = '_'.join(model.split('.')[0].split('_')[3:])
    return mix, model_name
