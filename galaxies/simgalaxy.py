from __future__ import print_function
import os
import numpy as np
import pylab as plt

from ..fileio import fileIO
from .. import match
from .. import trilegal
from .. import graphics
from .. import utils
from .starpop import StarPop

__all__ = ['SimGalaxy', 'get_mix_modelname']


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
        pass

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
        fit_file, =  fileIO.get_files(match_out_dir, fit_file_name)
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

        logl = self.data.get_col('logL')[self.rec]
        self.imstar, = np.nonzero((co <= 1) & (logl >= 3.3) &
                                  (self.stage[self.rec] == trilegal.get_stage_label('TPAGB')))

        self.icstar, = np.nonzero((co >= 1) &
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
            reg_inds, = np.nonzero(utils.points_inside_poly(points, verts))

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
                num = utils.brighter(mag2, trgb - self.count_offset, inds=ind).size
                graphics.plot_numbs(ax, num, xpos, ypos, **kwargs)

        if xlim is not None and ylim is not None:
            for ax in axs.ravel():
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
        if figname is None:
            figname = fileio.replace_ext(self.name, '.png')

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

    def make_lf(self, mag, bins=None, stages=None, inds=None, bin_width=0.1,
                hist_it_up=False, stage_inds=None):
        '''
        make a lf out of mag

        ARGS:
        mag: array to hist
        bins: bins for the histogram (or give bin_width)
        stages: evolutionary stages (will add individually)
        inds: indices of array include
        bin_width: width of mag bin for histogram
        stage_inds: indices to slice the stages
        RETURNS
        if stages, will name the attributes
        self.i[stage]_lfhist
        self.i[stage]_lfbins
        otherwise,
        self.lfhist
        self.lfbins
        '''
        # will use the variable again... silly.
        original_bins = bins
        if inds is None:
            inds = np.arange(len(mag))
        # will cycle through stages, so if none are passed, will just make
        # one LF.
        if stages == 'all':
            stages = ['PMS', 'MS', 'SUBGIANT', 'RGB', 'HEB', 'RHEB', 'BHEB',
                      'EAGB', 'TPAGB', 'POSTAGB', 'WD']
        if stages is None:
            sindss = [np.arange(len(mag))]
            extra = ['']
        elif type(stages) is str:
            stages = [stages]

        if type(stages) is list:
            self.all_stages(*stages)
            stage_names = ['i%s' % s.lower() for s in stages]
            sindss = [self.__getattribute__(s) for s in stage_names]
            extra = [s + '_' for s in stage_names]

        for i, sinds in enumerate(sindss):
            if stage_inds is not None:
                s_inds = stage_inds
            s_inds = np.intersect1d(inds, sinds)
            imag = mag[s_inds]

            if len(imag) < 2:
                print('no stars found with stage %s' % stages[i])
                hist = np.zeros(len(bins) - 1)

            if original_bins is None:
                bins = (np.max(imag) - np.min(imag)) / bin_width

            if hist_it_up is True:
                hist, bins = utils.hist_it_up(imag, threash=5)
            else:
                if type(bins) == np.float64 and bins < 1:
                    continue
                hist, _ = np.histogram(imag, bins=bins)
            self.__setattr__('%slfhist' % extra[i], hist)
            self.__setattr__('%slfbins' % extra[i], bins)
        return hist, bins

    def lognormalAv(self, disk_frac, mu, sigma, fg=0, df_young=0, df_old=8,
                    age_sep=3):
        '''
        Alexia ran calcsfh on PHAT data with:
        -dAvy=0 -diskav=0.20,0.385,-0.95,0.65,0,8,3
        MATCH README states:
          -diskAv=N1,N2,N3,N4 sets differential extinction law, which is treated as
            a foreground component (flat distribution from zero to N1), plus
            a disk component (lognormal with mu=N3 and sigma=N4) affecting a
            fraction of stars equal to N2.  The ratio of the star scale
            height to gas scale height is specified per time bin in the
            parameter file.  For small values (0 to 1), the effect is
            simple differential extinction.  For larger values, one will see
            some fraction of the stars (1-N2) effectively unreddened (those
            in front of the disk) and the remainder affected by the lognormal.
            N1 should be non-negative, N2 should fall between zero and 1, and
            N4 should be positive.
         -diskAv=N1,N2,N3,N4,N5,N6,N7 is identical to the previous selection,
            except that the ratio of star to disk scale height is N5 for
            recent star formation, N6 for ancient stars, and transitions
            with a timescale of N7 Gyr.  N5 and N6 should be non-negative, and
            N7 should be positive.
        -dAvy=0.5 sets max additional differential extinction for young stars only.
            For stars under 40 Myr, up to this full value may be added; there
            is a linear ramp-down with age until 100 Myr, at which point no
            differential extinction is added.  It is possible that the
            best value to use here could be a function of metallicity.  Note
            that if both -dAv and -dAvy are used, the extra extinction applied
            to young stars is applied to the first of the two flat
            distributions.


        '''
        #  N1 Flat distribution from zero to N1 [0.2]
        #  N2 disk fraction of stars with lognormal [0.385]
        #  N3 mu lognormal [-0.95]
        #  N4 sigma lognormal [0.65]
        #  N5 like N2 but for recent SFR [0]
        #  N6 like N2 but for ancient SFR  [8]
        #  N7 transition between recent and ancient (Gyr) [3]
        #  dAvy was run at 0, not implemented yet.
        from scipy.stats import lognorm
        N1 + lognorm(mu=N3, sigma=N4)

def get_mix_modelname(model):
    '''
    separate the mix and model name
    eg cmd_input_CAF09_COV0.5_ENV0.5.dat => CAF09, COV0.5_ENV0.5
    '''
    mix = model.split('.')[0].split('_')[2]
    model_name = '_'.join(model.split('.')[0].split('_')[3:])
    return mix, model_name
