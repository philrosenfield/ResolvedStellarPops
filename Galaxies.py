# Strangers wrote
import os
import sys
import numpy as np
import matplotlib.nxutils as nxutils
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import FancyArrow
from subprocess import PIPE, Popen
from matplotlib.ticker import NullFormatter, MaxNLocator, MultipleLocator
import pyfits
import itertools
import copy
from scipy.interpolate import interp1d
import logging
logger = logging.getLogger()

# Friends wrote
from scatter_contour import scatter_contour
import brewer2mpl

# I wrote
import astronomy_utils
import TrilegalUtils  # import get_stage_label, get_label_stage
import graphics.GraphicsUtils as rspgraph
import math_utils
import match_utils
import fileIO
import angst_tables
angst_data = angst_tables.AngstTables()


class star_pop(object):
    def __init__(self):
        pass

    def plot_cmd(self, color, mag, fig=None, ax=None, xlim=None, ylim=None,
                 yfilter=None, contour_args={}, scatter_args={}, plot_args={},
                 scatter_off=False, levels=5, threshold=75, contour_lw={},
                 color_by_arg_kw={}, filter1=None, filter2=None, slice_inds=None,
                 hist_bin_res=0.05, make_labels=True, log_counts=False):
        '''
        plot the galaxy cmd

        '''
        set_fig, set_ax = 0, 0
        if fig is None and ax is None:
            fig = plt.figure(figsize=(8, 8))
            set_fig, set_ax = 1, 1
            ax = plt.axes()

        if filter2 is None:
            if hasattr(self, 'filter2'):
                filter2 = self.filter2

        if filter1 is None:
            if hasattr(self, 'filter1'):
                filter1 = self.filter1

        if yfilter is None:
            yfilter = filter2

        if slice_inds is not None:
            color = color[slice_inds]
            mag = mag[slice_inds]

        if len(color_by_arg_kw) != 0:
            scatter_off = True
            self.color_by_arg(ax=ax, fig=fig, **color_by_arg_kw)

        if scatter_off is False:
            contour_args = dict({'cmap': cm.gray_r, 'zorder': 100}.items() +
                                contour_args.items())

            scatter_args = dict({'marker': '.', 'color': 'black', 'alpha': 0.2,
                                'edgecolors': 'none', 'zorder': 1}.items() +
                                scatter_args.items())

            contour_lw = dict({'linewidths': 2, 'colors': 'white',
                               'zorder': 200}.items() + contour_lw.items())
            if type(hist_bin_res) is list:
                hist_bin_res_c, hist_bin_res_m = hist_bin_res
            else:
                hist_bin_res_c = hist_bin_res
                hist_bin_res_m = hist_bin_res               
            ncolbin = int(np.diff((np.nanmin(color), np.nanmax(color))) / hist_bin_res_c)
            nmagbin = int(np.diff((np.nanmin(mag), np.nanmax(mag))) / hist_bin_res_m)
            plt_pts, cs = scatter_contour(color, mag,
                                          threshold=threshold, levels=levels,
                                          hist_bins=[ncolbin, nmagbin],
                                          contour_args=contour_args,
                                          scatter_args=scatter_args,
                                          contour_lw=contour_lw,
                                          ax=ax, log_counts=log_counts)
            self.plt_pts = plt_pts
            self.cs = cs
        else:
            plot_args = dict({'marker': '.', 'color': 'black', 'mew': 0.,
                              'lw': 0}.items() +
                             plot_args.items())
            ax.plot(color, mag, **plot_args)
        if xlim is None:
            xlim = (color.min(), color.max())
        if ylim is None:
            ylim = (mag.max(), mag.min())
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if make_labels is True:
            ax.set_xlabel('$%s-%s$' % (filter1, filter2), fontsize=20)
            ax.set_ylabel('$%s$' % yfilter, fontsize=20)
        ax.tick_params(labelsize=16)
        if set_ax == 1:
            self.ax = ax
        if set_fig == 1:
            self.fig = fig
        return fig, ax

    def color_by_arg(self, xcol, ycol, colorcol, bins=None, cmap=None, ax=None,
                     fig=None, labelfmt='$%.3f$', xdata=None, ydata=None,
                     coldata=None, xlim=None, ylim=None, slice_inds=None,
                     legend=True):
        if fig is None:
            if not hasattr(self, 'fig'):
                fig = plt.figure()
            else:
                fig = self.fig
        if ax is None:
            if not hasattr(self, 'ax'):
                ax = plt.axes()
            else:
                ax = self.ax
        if bins is None:
            bins = 10
        if xdata is None:
            xdata = self.data.get_col(xcol)
        if ydata is None:
            ydata = self.data.get_col(ycol)
        if colorcol is None:
            coldata = self.data.get_col(colorcol)
        if slice_inds is not None:
            xdata = xdata[slice_inds]
            ydata = ydata[slice_inds]
            coldata = coldata[slice_inds]
        # need the bins to be an array to use digitize.
        if type(bins) == int:
            hist, bins = np.histogram(coldata, bins=bins)
        inds = np.digitize(coldata, bins)
        uinds = np.unique(inds)
        # digitize sticks all points that aren't in bins in the final bin
        # cut that bin, or plot will be meaningless..
        if uinds[-1] == len(bins):
            uinds = uinds[:-1]
        if cmap is None:
            if 3 <= len(uinds) <= 11:
                #bmap = brewer2mpl.get_map('Spectral', 'Diverging', len(uinds))
                bmap = brewer2mpl.get_map('Paired', 'Qualitative', len(uinds))
                cols = bmap.mpl_colors
            else:
                cols = rspgraph.discrete_colors(len(uinds), colormap='RdYlGn')
        else:
            cols = rspgraph.discrete_colors(len(uinds), colormap=cmap)
        sub_inds = np.array([])
        nc = len(cols[0])
        colors = np.ndarray(shape=(len(xdata), nc), dtype=float)
        labs = []
        for j, i in enumerate(uinds):
            sinds, = np.nonzero(inds == i)
            N = len(sinds)
            if N == 0:
                continue
            if labelfmt != '':
                labs.append(labelfmt % bins[i])  # bins are left bin edges.
            #sinds_colors = np.repeat(cols[j], [N,N,N], axis=0).reshape(3, N).T
            colors[sinds] = cols[j]
            #fig1 = plt.figure()
            #ax1 = plt.axes()
            #ax1.plot(xdata[sinds], ydata[sinds], 'o', color=cols[j],
            #    label=labelfmt % bins[i])
            #ax1.legend()
            #ax1.set_xlim(xlim)
            #ax1.set_ylim(ylim)
            sub_inds = np.append(sub_inds, sinds)
        #colors = colors.reshape(len(colors)/3, 3)
        inds = map(int, sub_inds[:])
        np.random.shuffle(inds)
        # fake out the legend...
        if labelfmt != '':
            [ax.plot(999, 999, 'o', color=cols[i], mec=cols[i], label=labs[i])
             for i in range(len(labs))]

        ax.scatter(xdata[inds], ydata[inds], marker='o', s=15,
                   edgecolors='none', color=colors[inds])

        if xlim is None:
            xlim = (xdata.min(), xdata.max())
        if ylim is None:
            ylim = (ydata.max(), ydata.min())
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if hasattr(self, 'filter1') and hasattr(self, 'filter2') and legend is True:
            ax.set_xlabel('$%s-%s$' % (self.filter1, self.filter2), fontsize=20)
            ax.set_ylabel('$%s$' % self.filter2, fontsize=20)
            ax.tick_params(labelsize=16)
        if legend is True:
            ax.legend(loc=0, numpoints=1, frameon=False)
        return ax

    def decorate_cmd(self, mag1_err=None, mag2_err=None, trgb=False, ax=None,
                     reddening=True, dmag=0.5, text_extra=None, errors=True,
                     cmd_errors_kw={}, filter1=None):
        self.redding_vector(dmag=dmag, ax=ax)
        if errors is True:
            cmd_errors_kw['ax'] = ax
            self.cmd_errors(**cmd_errors_kw)
        self.text_on_cmd(extra=text_extra, ax=ax)
        if trgb is True:
            if filter1 is None:
                self.put_a_line_on_it(ax, self.trgb)
            else:
                self.put_a_line_on_it(ax, self.trgb, filter1=filter1,
                                      consty=False)

    def put_a_line_on_it(self, ax, val, consty=True, color='black',
                         ls='--', lw=2, annotate=True, filter1=None,
                         annotate_fmt='$TRGB=%.2f$'):
        """
        if consty is True: plots a constant y value across ax.xlims().
        if consty is False: plots a constant x on a plot of y vs x-y
        """
        (xmin, xmax) = ax.get_xlim()
        (ymin, ymax) = ax.get_ylim()
        xarr = np.linspace(xmin, xmax, 20)
        # y axis is magnitude...
        yarr = np.linspace(ymin, ymax, 20)
        if consty is True:
            # just a contsant y value over the plot range of x.
            ax.hlines(val, xmin, xmax, color=color, lw=lw)
            new_xarr = xarr
        if consty is False:
            # a plot of y vs x-y and we want to mark
            # where a constant value of x is
            # e.g, f814w vs f555-f814; val is f555
            new_xarr = val - yarr
            # e.g, f555w vs f555-f814; val is f814
            if filter1 is not None:
                yarr = xarr + val
                new_xarr = xarr
            ax.plot(new_xarr, yarr, ls, color=color, lw=lw)
        if annotate is True:
            ax.annotate(annotate_fmt % val, xy=(new_xarr[-1]-0.1,
                        yarr[-1]-0.2), ha='right', fontsize=16,
                        **rspgraph.load_ann_kwargs())

    def redding_vector(self, dmag=1., ax=None):
        if ax == None:
            ax = self.ax
        Afilt1 = astronomy_utils.parse_mag_tab(self.photsys, self.filter1)
        Afilt2 = astronomy_utils.parse_mag_tab(self.photsys, self.filter2)
        Rslope = Afilt2 / (Afilt1 - Afilt2)
        dcol = dmag / Rslope
        pstart = np.array([0., 0.])
        pend = pstart + np.array([dcol, dmag])
        points = np.array([pstart, pend])
        data_to_display = ax.transData.transform
        display_to_axes = ax.transAxes.inverted().transform
        ax_coords = display_to_axes(data_to_display(points))
        dy_ax_coords = ax_coords[1, 1] - ax_coords[0, 1]
        dx_ax_coords = ax_coords[1, 0] - ax_coords[0, 0]
        arr = FancyArrow(0.05, 0.95, dx_ax_coords, (1.)*dy_ax_coords,
                         transform=ax.transAxes, color='black', ec="none",
                         width=.005, length_includes_head=1, head_width=0.02)
        ax.add_patch(arr)

    def cmd_errors(self, binsize=0.1, errclr=-1.5, absmag=False, ax=None):
        if ax == None:
            ax = self.ax
        if type(self.data) == pyfits.fitsrec.FITS_rec:
            mag1err = self.data.MAG1_ERR
            mag2err = self.data.MAG2_ERR
        if absmag is False:
            mag1 = self.mag1
            mag2 = self.mag2
        else:
            mag1 = self.Mag1
            mag2 = self.Mag2
        color = mag1 - mag2

        nbins = (np.max(mag2) - np.min(mag2)) / binsize
        nbars = int(nbins / 5) - 1
        errmag = np.zeros(nbars)
        errcol = np.zeros(nbars)
        errmagerr = np.zeros(nbars)
        errcolerr = np.zeros(nbars)
        for q in range(len(errmag) - 1):
            test = mag2.min() + 5. * (q + 2) * binsize + 2.5 * binsize
            test2, = np.nonzero((mag2 > test - 2.5 * binsize) &
                               (mag2 <= test + 2.5 * binsize) &
                               (mag1 - mag2 > -0.5) &
                               (mag1 - mag2 < 2.5))
            if len(test2) < 5:
                continue
            errmag[q] = mag2.min() + 5. * (q + 2) * binsize \
                + 2.5 * binsize
            errcol[q] = errclr
            m2inds, = np.nonzero((mag2 > errmag[q] - 2.5 * binsize) &
                                (mag2 < errmag[q] + 2.5 * binsize))
            cinds, = np.nonzero((color > -0.5) & (color < 2.5))
            cinds = list(set(m2inds) & set(cinds))
            errmagerr[q] = np.mean(mag2err[m2inds])
            errcolerr[q] = np.sqrt(np.mean(mag1err[cinds] ** 2 +
                                           mag2err[cinds] ** 2))
        ax.errorbar(errcol, errmag, xerr=errcolerr, yerr=errmagerr,
                         ecolor='white', lw=3, capsize=0, fmt=None)
        ax.errorbar(errcol, errmag, xerr=errcolerr, yerr=errmagerr,
                         ecolor='black', lw=2, capsize=0, fmt=None)

    def text_on_cmd(self, extra=None, ax=None):
        an_kw = rspgraph.load_ann_kwargs()
        if ax is None:
            ax = self.ax
        strings = '$%s$ $\mu=%.3f$ $A_v=%.2f$' % (self.target.upper(), self.dmod,
                                                  self.Av)
        offset = .17
        if extra is not None:
            strings += ' %s' % extra
            offset = 0.2
        for string in strings.split():
            offset -= 0.04
            ax.text(0.95, offset, string, transform=ax.transAxes,
                         ha='right', fontsize=16, color='black', **an_kw)

    def annotate_cmd(self, ax, yval, string, offset=0.1, text_kw={}):
        text_kw = dict({'fontsize': 20}.items() + text_kw.items() + 
                         rspgraph.load_ann_kwargs().items())
        ax.text(ax.get_xlim()[0] + offset, yval - offset, string, **text_kw)

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
        if not hasattr(self, 'stage'):
            logger.warning('no stages marked in this file')
            return
        else:
            inds, = np.nonzero(self.stage ==
                               TrilegalUtils.get_stage_label(stage_name))
        return inds

    def make_hess(self, binsize, absmag=False, useasts=False, slice_inds=None,
                  hess_kw={}):
        '''
        adds a hess diagram of color, mag2 or Color, Mag2 (if absmag is True).
        if useasts is true will use ast_mags.

        slice_inds will slice the arrays and only bin those stars, there
        is no evidence besides the hess tuple itself that hess is not of the
        full cmd.

        See astronomy_utils doc for more information.
        '''
        if absmag is True:
            col = self.Color
            mag = self.Mag2
        elif useasts is True:
            col = self.ast_color[self.rec]
            mag = self.ast_mag2[self.rec]
        else:
            col = self.color
            mag = self.mag2
        if slice_inds is not None:
            col = col[slice_inds]
            mag = mag[slice_inds]
        self.hess = astronomy_utils.hess(col, mag, binsize, **hess_kw)
        return

    def hess_plot(self, fig=None, ax=None, colorbar=False, imshow_kw={}):
        '''
        Plots a hess diagram with imshow.
        default kwargs passed to imshow:
        default_kw = {'norm': LogNorm(vmin=None, vmax=self.[2].max()),
                      'cmap': cm.gray,
                      'interpolation': 'nearest',
                      'extent': [self.hess[0][0], self.hess[0][-1],
                                 self.hess[1][-1], self.hess[1][0]]}
        '''
        assert hasattr(self, 'hess'), 'run self.make_hess before plotting'
        if hasattr(self, 'filter2') and hasattr(self, 'filter1'):
            filter1 = self.filter1
            filter2 = self.filter2
        else:
            filter1 = None
            filter2 = None
        ax = astronomy_utils.hess_plot(self.hess, fig=fig, ax=ax,
                                       filter1=filter1, filter2=filter2,
                                       colorbar=colorbar,
                                       imshow_kw=imshow_kw)
        return ax

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

    def convert_mag(self, dmod=0., Av=0., target=None, shift_distance=False,
                    useasts=False):
        '''
        convert from mag to Mag or from Mag to mag or just shift distance.
        pass dmod, Av, or use AngstTables to look it up from target.
        shift_distance: for the possibility of doing dmod, Av fitting of model
        to data the key here is that we re-read the mag from the original data
        array.

        useasts only work with shift_distance is true.
        It will calculate the original dmod and av from self, and then shift
        that to the new dmod av. there may be a faster way, but this is just
        multiplicative effects.
        Without shift_distance: Just for common usage. If trilegal was given a
        dmod, it will swap it back to Mag, if it was done at dmod=10., will
        shift to given dmod. mag or Mag attributes are set in __init__.

        '''
        check = [(dmod + Av == 0.), (target is None)]
        #assert False in check, 'either supply dmod and Av or target'
        if check[0] is True:
            filters = ','.join((self.filter1, self.filter2))
            if target is not None:
                logger.info('converting mags with angst table using %s' %
                            target)
                self.target = target
            elif hasattr(self, 'target'):
                logger.info('converting mags with angst table using initialized %s'
                            % self.target)
            tad = angst_data.get_tab5_trgb_av_dmod(self.target, filters)
            __, self.Av, self.dmod = tad
        else:
            self.dmod = dmod
            self.Av = Av

        mag_covert_kw = {'Av': self.Av, 'dmod': self.dmod}
        if shift_distance is True:
            if useasts is True:
                am1 = self.ast_mag1
                am2 = self.ast_mag2
                old_dmod, old_Av = astronomy_utils.get_dmodAv(self)
                old_mag_covert_kw = {'Av': old_Av, 'dmod': old_dmod}
                M1 = astronomy_utils.mag2Mag(am1, self.filter1,
                                             self.photsys,
                                             **old_mag_covert_kw)
                M2 = astronomy_utils.mag2Mag(am2, self.filter2,
                                             self.photsys,
                                             **old_mag_covert_kw)
            else:
                M1 = self.data.get_col(self.filter1)
                M2 = self.data.get_col(self.filter2)
            self.mag1 = astronomy_utils.Mag2mag(M1, self.filter1, self.photsys,
                                                **mag_covert_kw)
            self.mag2 = astronomy_utils.Mag2mag(M2, self.filter2, self.photsys,
                                                **mag_covert_kw)
            self.color = self.mag1 - self.mag2
        else:
            if hasattr(self, 'mag1'):
                self.Mag1 = astronomy_utils.mag2Mag(self.mag1, self.filter1,
                                                    self.photsys,
                                                    **mag_covert_kw)
                self.Mag2 = astronomy_utils.mag2Mag(self.mag2, self.filter2,
                                                    self.photsys,
                                                    **mag_covert_kw)
                self.Color = self.Mag1 - self.Mag2
                if hasattr(self, 'trgb'):
                    self.Trgb = astronomy_utils.mag2Mag(self.trgb,
                                                        self.filter2,
                                                        self.photsys,
                                                        **mag_covert_kw)
            if hasattr(self, 'Mag1'):
                self.mag1 = astronomy_utils.Mag2mag(self.Mag1, self.filter1,
                                                    self.photsys,
                                                    **mag_covert_kw)
                self.mag2 = astronomy_utils.Mag2mag(self.Mag2,
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

        # new arrays must be equal length as the data
        len_test = np.array([len(v) == nrows
                            for v in new_cols.values()]).prod()
        if not len_test:
            'array lengths are not the same.'
            return -1
        header = self.get_header()
        # add new columns to the data and their names to the header.
        for k, v in new_cols.items():
            header += ' %s' % k
            data = np.column_stack((data, v))
        # update self.data
        self.data.data_array = data
        col_keys = header.replace('#', '').split()
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

    def double_gaussian_contamination(self, all_verts, dcol=0.05, Color=None,
                                      Mag2=None, color_sep=None, diag_plot=False,
                                      absmag=False, thresh=5):
        '''
        This function fits a double gaussian to a color histogram of stars
        within the <maglimits> and <colorlimits> (tuples).

        It then finds the intersection of the two gaussians, and the fraction
        of each integrated gaussian that crosses over the intersection color
        line.
        '''
        try:
            mpfit
        except NameError:
            from mpfit import mpfit
        
        try:
            integrate
        except NameError:
            from scipy import integrate
        # the indices of the stars within the MS/BHeB regions
    
        # poisson noise to compare with contamination
        if Color is None:
            if absmag is True:
                Color = self.Color
                Mag2 = self.Mag2
            else:
                Color = self.color
                Mag2 = self.mag2
        points = np.column_stack((Color, Mag2))
        all_inds, = np.nonzero(nxutils.points_inside_poly(points, all_verts))
        if len(all_inds) <= thresh:
            print 'not enough points found within verts'
            return np.nan, np.nan, np.nan, np.nan, np.nan
        poission_noise = np.sqrt(float(len(all_inds)))

        # make a color histogram
        #dcol = 0.05
        color = Color[all_inds]
        col_bins = np.arange(color.min(), color.max() + dcol, dcol)
        #nbins = np.max([len(col_bins), int(poission_noise)])
        hist = np.histogram(color, bins=col_bins)[0]
    
        # uniform errors
        err = np.zeros(len(col_bins[:1])) + 1.

        # set up inputs
        hist_in = {'x': col_bins[1:], 'y': hist, 'err': err}
    
        # set up initial parameters:
        # norm = max(hist),
        # mean set to be half mean, and 3/2 mean,
        # sigma set to be same as dcol spacing...
        p0 = [np.nanmax(hist)/2., np.mean(col_bins[1:]) - np.mean(col_bins[1:])/2, dcol,
              np.nanmax(hist)/2., np.mean(col_bins[1:]) + np.mean(col_bins[1:])/2, dcol]

        mp_dg = mpfit(math_utils.mp_double_gauss, p0, functkw=hist_in, quiet=True)
        if mp_dg.covar is None:
            print 'not double guassian'
            return 0., 0., poission_noise, float(len(all_inds)), color_sep
        else:
            perc_err = (np.array(mp_dg.perror) - np.array(mp_dg.params)) / \
                        np.array(mp_dg.params)
            if np.sum([p**2 for p in perc_err]) > 10.:
                print 'not double guassian, errors too large'
                return 0., 0., poission_noise, float(len(all_inds)), color_sep
        # take fit params and apply to guassians on an arb color scale
        color_array = np.linspace(col_bins[0], col_bins[-1], 1000)
        g_p1 = mp_dg.params[0: 3]
        g_p2 = mp_dg.params[3:]
        gauss1 = math_utils.gaussian(color_array, g_p1)
        gauss2 = math_utils.gaussian(color_array, g_p2)
        print g_p1[1], g_p2[1]
        # color separatrion is the intersection of the two gaussians..
        double_gauss = gauss1 + gauss2
        #between_peaks = np.arange(
        min_locs = math_utils.find_peaks(gauss1 + gauss2)['minima_locations']
        g1, g2 = np.sort([g_p1[1], g_p2[2]])
        ginds, = np.nonzero( (color_array > g1) & (color_array < g2))
        #ginds2, = np.nonzero(gauss2)
        #ginds = list(set(ginds1) & set(ginds2))
        min_locs = np.argmin(np.abs(gauss1[ginds]-gauss2[ginds]))
        print min_locs
        auto_color_sep = color_array[ginds][min_locs]
        print auto_color_sep
        if auto_color_sep == 0:
            auto_color_sep = np.mean(col_bins[1:])
            print 'using mean as color_sep'
        if color_sep is None:
            color_sep = auto_color_sep
        else:
            print 'you want color_sep to be %.4f, I found it at %.4f' % (color_sep,
                                                                         auto_color_sep)

        # find contamination past the color sep...
        g12_Integral = integrate.quad(math_utils.double_gaussian, -np.inf, np.inf,
                                      mp_dg.params)
        try:
            norm =  float(len(all_inds)) / g12_Integral[0] 
        except ZeroDivisionError:
            norm = 0.
        g1_Integral = integrate.quad(math_utils.gaussian, -np.inf, np.inf, g_p1)
        g2_Integral = integrate.quad(math_utils.gaussian, -np.inf, np.inf, g_p2)

        g1_Int_colsep = integrate.quad(math_utils.gaussian, -np.inf, color_sep, g_p1)
        g2_Int_colsep = integrate.quad(math_utils.gaussian, color_sep, np.inf, g_p2)

        left_in_right = (g1_Integral[0] - g1_Int_colsep[0]) * norm
        right_in_left = (g2_Integral[0] - g2_Int_colsep[0]) * norm
        '''
    
        try:
            left_in_right = g1_Int_colsep[0] / g1_Integral[0]
    
            left_in_right = 0.

        try:
            right_in_left = g2_Int_colsep[0] / g2_Integral[0]
        except ZeroDivisionError:
            right_in_left = 0.
        '''
        # diagnostic
        #print color_sep
        if diag_plot is True:
            fig1, ax1 = plt.subplots()
            ax1.plot(col_bins[1:], hist, ls='steps', lw=2)
            ax1.plot(col_bins[1:], hist, 'o')
            ax1.plot(color_array,
                    math_utils.double_gaussian(color_array, mp_dg.params))
            ax1.plot(color_array, gauss1)
            ax1.plot(color_array, gauss2)
            #ax1.set_ylim((0, 100))
            ax1.set_xlim(color.min(), color.max())
            ax1.set_xlabel('$%s-%s$' % (self.filter1, self.filter2), fontsize=20)
            ax1.set_ylabel('$\#$', fontsize=20)
            ax1.set_title('%s Mean Mag2: %.2f, Nbins: %i' % (self.target,
                                                             np.mean(np.array(all_verts)[:, 1]),
                                                             len(col_bins)))
            ax1.vlines(color_sep, *ax1.get_ylim())
            ax1.text(0.1, 0.95, 'left in right: %i' % left_in_right,
                     transform=ax1.transAxes)
            ax1.text(0.1, 0.90, 'right in left: %i' % right_in_left,
                     transform = ax1.transAxes)
            fig1.savefig('heb_contamination_%s_%s_%s_mag2_%.2f.png' % (self.filter1,
                                                                       self.filter2,
                                                                       self.target,
                                                                       np.mean(np.array(all_verts)[:, 1])))
            print 'wrote heb_contamination_%s_%s_%s_mag2_%.2f.png' % (self.filter1,
                                                                      self.filter2,
                                                                      self.target,
                                                                      np.mean(np.array(all_verts)[:, 1]))
            #plt.close()
        return left_in_right, right_in_left, poission_noise, float(len(all_inds)), color_sep

    def stars_in_region(self, mag2, mag_dim, mag_bright, mag1=None,
                        verts=None, col_min=None, col_max=None):
        '''
        counts stars in a region. Give mag2 and the mag2 limits.
        If col_min, col_max, and verts are none, will just give all stars
        between those mag limits (if no color info is used mag2 can actually
        be mag1.)
        If verts are given (Nx2) array, will use those, otherwise will build
        a polygon from col_* and mag_* limits.

        Returns indices inside.
        '''
        if verts is None:
            if col_min is None:
                inds = math_utils.between(mag2, mag_dim, mag_bright)
            else:
                verts = np.array([[col_min, mag_dim],
                                  [col_min, mag_bright],
                                  [col_max, mag_bright],
                                  [col_max, mag_dim],
                                  [col_min, mag_dim]])

                points = np.column_stack((mag1 - mag2, mag2))
                inds, = np.nonzero(nxutils.points_inside_poly(points, verts))
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
                print 'no stars found with stage %s' % stages[i]
                hist = np.zeros(len(bins)-1)
            if original_bins is None:
                bins = (np.max(imag) - np.min(imag)) / bin_width
            if hist_it_up is True:
                hist, bins = math_utils.hist_it_up(imag, threash=5)
            else:
                if type(bins) == np.float64 and bins < 1:
                    continue
                hist, _ = np.histogram(imag, bins=bins)
            self.__setattr__('%slfhist' % extra[i], hist)
            self.__setattr__('%slfbins' % extra[i], bins)
        return hist, bins

    def interp_errs(self, mag1err=None, mag2err=None, binsize=0.1):
        if type(self.data) == pyfits.fitsrec.FITS_rec:
            mag1err = self.data.MAG1_ERR
            mag2err = self.data.MAG2_ERR
        if absmag is True:
            mag1 = self.Mag1
            mag2 = self.Mag2
        else:
            mag1 = self.mag1
            mag2 = self.mag2

        interp_arr = np.linspace(0, 1, len(mag1))
        mag1e_hist = np.array(np.histogram(mag1err, bins=mag_bins)[0], dtype=float)

        mag2e_hist = np.array(np.histogram(mag2err, bins=mag_bins)[0], dtype=float)

        self.fmag1err = interp1d(mag_bins, mag1e_hist, bounds_error=False)
        self.fmag2err = interp1d(mag_bins, mag2e_hist, bounds_error=False)
        return


class galaxies(star_pop):
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
            d[key] = galaxies.select_on_key(self, 'z', z)
        d['no z'] = [g for g in self.galaxies if np.isnan(g.z)]
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


class galaxy(star_pop):
    '''
    angst and angrrr galaxy object. data is a ascii tagged file with stages.
    '''
    def __init__(self, fname, filetype=None, hla=True, angst=True,
                 band=None, photsys=None, trgb=np.nan, z=-99, Av=None, dmod=None,
                 filter1=None, filter2=None):
        '''
        I hate this init.
        TODO:
        make a file_type reader that the init calls.
        add IR stuff to angst_tables not down here to be read at each call.
        '''
        self.base, self.name = os.path.split(fname)
        star_pop.__init__(self)
        # name spaces
        self.trgb = trgb
        self.z = z
        self.Av = Av
        self.dmod = dmod
        self.load_data(fname, filetype=filetype, hla=hla, angst=angst,
                       band=band, photsys=photsys, filter1=filter1,
                       filter2=filter2)

        # angst table loads
        if angst is True:
            self.comp50mag1 = angst_data.get_50compmag(self.target,
                                                       self.filter1)
            self.comp50mag2 = angst_data.get_50compmag(self.target,
                                                       self.filter2)
            if hasattr(self, 'filter3'):
                self.comp50mag3 = angst_data.get_50compmag(self.target,
                                                           self.filter3)
                self.comp50mag4 = angst_data.get_50compmag(self.target,
                                                           self.filter4)

            self.trgb_av_dmod()
            # Abs mag
            self.convert_mag(dmod=self.dmod, Av=self.Av, target=self.target)
            #self.z = galaxy_metallicity(self, self.target)

    def load_data(self, fname, filetype=None, hla=True, angst=True,
                  band=None, photsys=None, filter1=None, filter2=None):

        if hla is True:
            self.survey, self.propid, self.target, filts, psys = \
                hla_galaxy_info(fname)
            # photometry
            self.filter1, self.filter2 = filts.upper().split('-')
            self.photsys = psys.replace('-', '_')
        else:
            self.survey = ' '
            self.photsys = photsys
            if None in [filter1, filter2]:
                self.propid, self.target, self.filter1, self.filter2 =  \
                    bens_fmt_galaxy_info(fname)
            else:
                self.propid = ''
                self.target = fname
                self.filter1 = filter1
                self.filter2 = filter2

        if filetype is None:
            self.data = fileIO.readfile(fname)
            if not None in [self.filter1, self.filter2]:
                self.mag1 = self.data[self.filter1]
                self.mag2 = self.data[self.filter2]
            else:
                self.mag1 = np.nan
                self.mag2 = np.nan
            self.data = self.data.view(np.recarray)

        elif 'fits' in filetype:
            hdu = pyfits.open(fname)
            #self.data =  fileIO.read_fits(fname)
            if photsys is not None:
                ext = self.photsys.upper().split('_')[0]
            else:
                cam = hdu[0].header['CAMERA']
                if cam == 'ACS':
                    self.photsys = 'acs_wfc'
                elif cam == 'WFPC2':
                    self.photsys = 'wfpc2'
                else:
                    logger.error('I do not know the photsys.')
            self.data = hdu[1].data
            self.ra = self.data['ra']
            self.dec = self.data['dec']

            if filetype == 'fitstable':
                self.header = hdu[0].header
                ext = self.header['CAMERA']
                if '-' in ext:
                    if 'ACS' in ext:
                        ext = 'ACS'
                    else:
                        ext = ext.split('-')[-1]
                if band.upper() == 'IR':
                    ext = band.upper()
                self.mag1 = self.data['mag1_%s' % ext]
                self.mag2 = self.data['mag2_%s' % ext]
                self.filters = [self.filter1, self.filter2]
            if filetype == 'fitsimage':
                # made to read holtmann data...
                # this wont work on ir filters.
                filts = [f for f in self.data.columns.names
                         if f.endswith('w') and f.startswith('f')]
                order = np.argsort([float(f.replace('f', '').replace('w', ''))
                                    for f in filts])
                self.filter1 = filts[order[0]].upper()
                self.filter2 = filts[order[1]].upper()
                self.mag1 = self.data[self.filter1]
                self.mag2 = self.data[self.filter2]
        elif filetype == 'tagged_phot':
            self.data = fileIO.read_tagged_phot(fname)
            self.mag1 = self.data['mag1']
            self.mag2 = self.data['mag2']
            self.stage = self.data['stage']

        elif filetype == 'match_phot':
            self.data = np.genfromtxt(fname, names=['mag1', 'mag2'])
            self.mag1 = self.data['mag1']
            self.mag2 = self.data['mag2']

        elif filetype == 'm31brick':
            assert band is not None, 'Must supply band, uv, ir, acs'
            hdu = pyfits.open(fname)
            self.data = hdu[1].data
            self.header = hdu[0].header
            ext = band
            mag1 = self.data['%s_mag1' % ext]
            mag2 = self.data['%s_mag2' % ext]
            inds = list(set(np.nonzero(np.isfinite(mag1))[0]) &
                        set(np.nonzero(np.isfinite(mag2))[0]))
            self.mag1 = mag1  # [inds]
            self.mag2 = mag2  # [inds]
            self.rec = inds
            self.ra = self.data['ra']
            self.dec = self.data['dec']
        elif filetype == 'agbsnap':
            data = pyfits.getdata(fname)
            self.propid, self.target, _, self.filter1, self.filter2, \
                self.filter3, self.filter4 = self.name.split('.')[0].split('_')

            self.filters = [self.filter1, self.filter2, self.filter3,
                            self.filter4]

            if hasattr(data, 'MAG1_ACS'):
                self.mag1 = data.MAG1_ACS
                self.mag2 = data.MAG2_ACS
            else:
                print 'not sure what to do here, boss'
                sys.exit(2)
            self.mag3 = data.MAG3_IR
            self.mag4 = data.MAG4_IR
            self.ra = data.RA
            self.dec = data.DEC
            self.data = data
        else:
            logger.error('bad filetype')
            sys.exit(2)
        self.color = self.mag1 - self.mag2

    def trgb_av_dmod(self):
        '''
        returns trgb, av, dmod from angst table
        '''
        filters = ','.join((self.filter1, self.filter2))
        (self.trgb, self.Av, self.dmod) = \
            angst_data.get_tab5_trgb_av_dmod(self.target, filters)
        if hasattr(self, 'filter3'):
            filters = ','.join((self.filter3, self.filter4))
            (self.ir_trgb, self.Av, self.dmod) = \
                angst_data.get_tab5_trgb_av_dmod(self.target, filters)

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
            self.data = fileIO.read_table(trilegal_out)
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
        try:
            fit_file, =  fileIO.get_files(match_out_dir, fit_file_name)
            self.chi2, self.fit = match_utils.get_fit(fit_file)
        except ValueError:
            logger.warning('no match output for %s.' % fit_file_name)
        return

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
            logger.warning('no ast corrections.')
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
        try:
            co = self.data.get_col('C/O')[self.rec]
        except KeyError:
            logger.warning('no agb stars... trilegal ran w/o -a flag?')
            return

        mdot = self.data.get_col('logML')[self.rec]
        logl = self.data.get_col('logL')[self.rec]
        self.imstar, = np.nonzero((co <= 1) &
                                  (logl >= 3.3) &
                                  (mdot <= -5) &
                                  (self.stage[self.rec] ==
                                   TrilegalUtils.get_stage_label('TPAGB')))

        self.icstar, = np.nonzero((co >= 1) &
                                  (mdot <= -5) &
                                  (self.stage[self.rec] ==
                                   TrilegalUtils.get_stage_label('TPAGB')))

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
            reg_inds, = np.nonzero(nxutils.points_inside_poly(points, verts))

        # combine all inds.
        sinds = list(set(ibright) & set(st_inds) & set(reg_inds) &
                     set(sinds_cut))

        nsim_stars = float(len(sinds))
        
        if len(sinds) == 0:
            logger.warning('no stars with %s < %.2f' % (new_attr, magcut))
            self.__setattr__('%s_inds' % new_attr, [-1])
            self.__setattr__('%s' % new_attr, 999.)
            return [-1], 999.

        # find inds for normalization
        if by_stage is True:
            assert ndata_stars is -np.inf, \
                'error! with by_stage=True, ndata_stars will be derived'
            dsinds, = np.nonzero((stage ==
                                  TrilegalUtils.get_stage_label(stage_lab)) &
                                 (mag < magcut))
            assert dsinds.size > 0, 'no data stars in stage %s' % stage_lab
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

    def setup_plot_by_stage(self, inds=None):
        stage = self.stage
        if inds is not None:
            stage = self.stage[inds]
        ustage = np.unique(stage)
        nplots = ustage.size + 1.
        bcols = brewer2mpl.get_map('Paired', 'qualitative', len(ustage))
        cols = bcols.mpl_colors
        subplots_kwargs = {'sharex': True, 'sharey': True, 'figsize': (12, 8)}
        fig, (axs) = rspgraph.setup_multiplot(nplots, **subplots_kwargs)
        return fig, (axs), cols

    def cmd_by_stage(self, color, mag2, inds=None, xlim=None, ylim=None,
                     extra='', figname=None, trgb=None, **kwargs):
        '''
        made to be called from diagnostic plots. Make a panel of plots for
        each stage (get_label_stage)
        '''
        stage = self.stage
        if inds is not None:
            stage = self.stage[inds]
        ustage = np.unique(stage)
        fig, (axs), cols = self.setup_plot_by_stage(inds=inds)
        # first plot is summary.

        ax0, cols = rspgraph.colorplot_by_stage(axs.ravel()[0],
                                                color, mag2, '.', stage,
                                                cols=cols)
        # go through each existing evolutionary phase and plot those stars.
        for i, (ax, st) in enumerate(zip(axs.ravel()[1:], ustage)):
            label = TrilegalUtils.get_label_stage(int(st))
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
                rspgraph.plot_lines([ax], ax.get_xlim(), trgb)
                text_offset = 0.02
                xpos = ax.get_xlim()[0] + 2 * text_offset
                ypos = trgb - text_offset
                num = math_utils.brighter(mag2, trgb-self.count_offset,
                                          inds=ind).size
                rspgraph.plot_numbs(ax, num, xpos, ypos, **kwargs)

        if xlim is not None and ylim is not None:
            for ax in axs.ravel():
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
        if figname is None:
            figname = fileIO.replace_ext(self.name, '.png')

        figname = figname.replace('.png', '%s.png' % extra)
        plt.savefig(figname)
        logger.info('wrote %s' % figname)
        return

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
        
        hist, bins = np.histogram(data[inds], bins)
        
        return hist, bins

  
class sim_and_gal(object):
    def __init__(self, galaxy, simgalaxy):
        self.gal = galaxy
        self.sgal = simgalaxy
        if hasattr(self.gal, 'maglims'): 
            self.maglims = self.gal.maglims
        else:
            self.maglims = [90., 90.]
        
        if not hasattr(self.sgal, 'norm_inds'):
            self.sgal.norm_inds = np.arange(len(self.sgal.data.data_array))


    def make_mini_hess(self, color, mag, scolor, smag, ax=None, hess_kw={}):
        
        hess_kw = dict({'binsize': 0.1, 'cbinsize': 0.05}.items() + hess_kw.items())
        self.gal_hess = astronomy_utils.hess(color, mag, **hess_kw)
        hess_kw['cbin'] = self.gal_hess[0]
        hess_kw['mbin'] = self.gal_hess[1]
        self.sgal_hess = astronomy_utils.hess(scolor, smag, **hess_kw)
        comp_hess = copy.deepcopy(self.gal_hess)
        comp_hess = comp_hess[:-1] + ((self.gal_hess[-1] - self.sgal_hess[-1]),)
        self.comp_hess = comp_hess
        #self.comp_hist = sgal_hist

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
            ginds, = np.nonzero(nxutils.points_inside_poly(points, agb_verts))
            sinds, = np.nonzero(nxutils.points_inside_poly(spoints, agb_verts))
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
        srgb_norm, = np.nonzero(nxutils.points_inside_poly(spoints,
                                                           self.gal.norm_verts))

        self.sgal.nbrighter.append(len(srgb_norm))

        # the number of sim stars in the agb_verts polygon
        self.sgal.nbrighter.append(len(sinds))

        nrgb_nagb_data = float(self.gal.nbrighter[1])/float(self.gal.nbrighter[0])
        nrgb_nagb_sim = float(self.sgal.nbrighter[1])/float(self.sgal.nbrighter[0])
        self.agb_verts = agb_verts
        return nrgb_nagb_data, nrgb_nagb_sim

    def _make_LF(self, filt1, filt2, res=0.1, plt_dir=None, plot_LF_kw={},
                comp50=False, add_boxes=True, color_hist=False, plot_tpagb=False,
                figname=None):
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
            print 'not using ASTs!!'
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

        plot_LF_kw = dict({'model_plt_color': 'red',
                           'data_plt_color': 'black',
                           'color_hist': color_hist}.items() +
                           plot_LF_kw.items())
        
        #itpagb, = np.nonzero(self.sgal.stage[self.sgal.norm_inds] == 8)
        fig, axs, top_axs = self._plot_LF(color, mag,
                                             scolor[self.sgal.norm_inds],
                                             smag[self.sgal.norm_inds],
                                             filt1, filt2, itpagb=itpagb,
                                             gal_hist=self.gal_hist, bins=self.bins,
                                             sgal_hist=self.sgal_hist,
                                             **plot_LF_kw)
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
        print 'wrote %s' % figname
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
                # Set the top histograms ylims to the ylim that is bigger
                #ymax1 = top_axs[0].get_ylim()[1]
                #ymax2 = top_axs[1].get_ylim()[1]
                #ylim, = [ymax1 if ymax1 > ymax2 else ymax2]
                #[ax.set_ylim(top_axs[0].get_ylim()[0], ylim) for ax in top_axs[:-1]]
                #top_axs[1].xaxis.set_major_formatter(NullFormatter())
                #[ax.xaxis.set_major_formatter(NullFormatter()) for ax in top_axs[:-1]]
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
            print 'doing itpagb'
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
            #top_axs[0].plot(self.color_bins[1:], self.sgal_color_hist,
            #               **hist_kw)
            #top_axs[1].semilogy(self.mass_bins[1:], self.mass_hist, **hist_kw)

            hist_kw['color'] = 'royalblue'
            top_axs[0].plot(self.color_bins[1:], self.sgal_tpagb_color_hist,
                            **hist_kw)
            top_axs[0].set_ylabel('$\#$', fontsize=16)
            top_axs[1].semilogy(self.mass_bins[1:], self.tp_mass_hist, **hist_kw)
            top_axs[1].set_xlabel('$M_\odot$', fontsize=16)
            black2red = rspgraph.stitch_cmap(plt.cm.Reds_r, plt.cm.Greys,
                                             stitch_frac=0.555, dfrac=0.05)

            astronomy_utils.hess_plot(self.comp_hess, ax=top_axs[2],
                                      imshow_kw={'cmap': black2red,
                                                 'interpolation': 'nearest',
                                                 'aspect': 'equal',
                                                 'norm': None,
                                                 'vmax': np.abs(self.comp_hess[-1]).max(),
                                                 'vmin': -np.abs(self.comp_hess[-1]).max()},
                                      imshow=True)

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
    print 'galaxy_metallicity is gonna break shit.'
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
    return fileIO.get_files(fake_loc, '*%s*.matchfake' % target.upper())[0]


def ast_correct_trilegal_sim(sgal, fake_file=None, outfile=None,
                             overwrite=False, spread_outfile=None,
                             leo_method=False, spread_outfile2=None,
                             asts_obj=None):
    '''
    correct trilegal simulation with artificial star tests.
    options to write the a copy of the trilegal simulation with the corrected
    mag columns, and also just the color, mag into a file (spread_too).

    ARGS:
    sgal: simgalaxy instance with convert_mag method already called.
    fake_file: string - matchfake file(s), or an artifical_star_tests instance(s).
    outfile: string - ast_file to write
    overwrite: overwrite output files, is also passed to write_spread
    spread_too: call write_spread flag
    spread_outfile: outfile for write_spread

    RETURNS:
    adds corrected mags to sgal.data.data_array and updates sgal.data.key_dict

    if fake_file is an a list of opt and ir, will do the corrections twice.
    fake_file as artificial_star_instance won't work if leo_method=True.
    '''
    ir_too = False

    def leo_ast_correction(leo_code, fake_file, sim_file, spread_outfile):
        # <hit enter>
        EOF = os.path.join(os.environ['PYTHONPATH'], 'EOF')

        cmd = '%s << %s\n' % (leo_code, EOF)
        cmd += '\n'.join((fake_file, sim_file, spread_outfile, EOF))

        logger.debug("%s + %s -> %s" % (sim_file, fake_file, spread_outfile))
        logger.debug('Running %s...' % os.path.split(leo_code)[1])
        p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE,
                  close_fds=True)

        stdout, stderr = (p.stdout, p.stderr)
        p.wait()
        return

    def add_cols_leo_method(sgal, spread_outfile):
        spout = fileIO.readfile(spread_outfile)
        ast_filts = [s for s in spout.dtype.names if '_cor' in s]
        new_cols = {}
        for ast_filt in ast_filts:
            new_cols[ast_filt] = spout[ast_filt]
            print len(spout[ast_filt])
        print len(sgal.mag2)
        print sgal.add_data(**new_cols)

    #assert fake_file is not None and asts_obj is not None, \
    #    'ast_correct_trilegal_sim: fake_file now needs to be passed'

    if leo_method is True:
        print 'ARE YOU SURE YOU WANT TO USE THIS METHOD!?'
        # this method tosses model stars where there are no ast corrections
        # which is fine for completeness < .50 but not for completeness > .50!
        # so don't use it!! It's also super fucking slow.
        assert spread_outfile is not None, \
            'need spread_outfile set for Leo AST method'

        logger.info("completeness using Leo's method")

        sim_file = os.path.join(sgal.base, sgal.name)

        # special version of spread_angst made for wfc3snap photsys!
        # THIS IS BAD CODING HARD LINK?!
        leo_code = '/home/rosenfield/research/TP-AGBcalib/SNAP/models/spread_angst'

        if type(fake_file) == list:
            assert spread_outfile2 is not None, \
                'need another string for spread file name with four filters'
            fake_file_ir = fake_file[1]
            fake_file = fake_file[0]
            ir_too = True

        leo_ast_correction(leo_code, fake_file, sim_file, spread_outfile)
        add_cols_leo_method(sgal, spread_outfile)
        if ir_too is True:
            # run the out ast corrections through a version of spread_angst to
            # correct the ir.
            leo_code += '_ir'
            leo_ast_correction(leo_code, fake_file_ir, spread_outfile, spread_outfile2)
            add_cols_leo_method(sgal, spread_outfile2)
            # no need to keep the spread_outfile now that we have a new one.
            os.remove(spread_outfile)
    else:
        if type(fake_file) is str:
            fake_files = [fake_file]
        else:
            fake_files = fake_file

        sgal.fake_files = fake_files

        if asts_obj is None:
            asts_obj = [artificial_star_tests(fake_file)
                        for fake_file in fake_files]
        elif type(asts_obj) is str:
            asts_obj = [asts_obj]

        for asts in asts_obj:
            # sgal.filter1 or sgal.mag1 doesn't have to match the asts.filter1
            # in fact, it won't if this is done on both opt and nir data.
            mag1 = sgal.data.get_col(asts.filter1)
            mag2 = sgal.data.get_col(asts.filter2)

            cor_mag1, cor_mag2 = asts.ast_correction(mag1, mag2,
                                                     **{'binsize': 0.2})
            new_cols = {'%s_cor' % asts.filter1: cor_mag1,
                        '%s_cor' % asts.filter2: cor_mag2}
            sgal.add_data(**new_cols)

        if outfile is not None:
            if overwrite is True or not os.path.isfile(outfile):
                TrilegalUtils.write_trilegal_sim(sgal, outfile)
            else:
                logger.warning('%s exists, not overwriting' % outfile)

        if spread_outfile is not None:
            TrilegalUtils.write_spread(sgal, outfile=spread_outfile,
                                       overwrite=overwrite)
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
            __, self.target, self.filter1, self.filter2, _ = \
                self.name.split('_')
        except:
            try:
                self.target, self.filter1, filter2 = self.name.split('_')
                self.filter2 = filter2.split('.')[0]
            except:
                try:
                    __, self.target, __, self.filter1, self.filter2, _ = \
                        self.name.split('_')
                except:
                    pass
        artificial_star_tests.load_fake(self, filename)

    def recovered(self, threshold=9.99):
        '''
        find indicies of stars with magdiff < threshold

        ARGS:
        threshold: [9.99] magin - magout threshold for recovery

        RETURNS:
        self.rec: recovered stars in both filters
        rec1, rec2: recovered stars in filter1, filter2
        '''
        rec1, = np.nonzero(self.mag1diff < threshold)
        rec2, = np.nonzero(self.mag2diff < threshold)
        self.rec = list(set(rec1) & set(rec2))
        return rec1, rec2

    def load_fake(self, filename):
        '''read MATCH fake file into attributes'''
        names = ['mag1', 'mag2', 'mag1diff', 'mag2diff']
        self.data = np.genfromtxt(filename, names=names)
        # unpack into attribues
        for name in names:
            self.__setattr__(name, self.data[name])

    def bin_asts(self, binsize=0.2, bins=None):
        '''
        bin the artificial star tests

        ARGS:
        bins: bins for the asts
        binsize: width of bins for the asts

        RETURNS:
        self.am1_inds, self.am2_inds: the indices of the bins to
            which each value in mag1 and mag2 belong (see np.digitize).
        self.ast_bins: bins used for the asts.
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
        randomly sample *arr* *nselections* times

        ARGS:
        arr: array or list to sample
        nselections: int times to sample

        RETURNS:
        rands: list of selections, length nselections
        '''
        import random
        rands = [random.choice(arr) for i in range(nselections)]
        return rands

    def ast_correction(self, obs_mag1, obs_mag2, binsize=0.2, bins=None,
                       not_rec_val=np.nan):
        '''
        apply ast correction to input mags.

        ARGS:
        obs_mag1, obs_mag2: N, 1 arrays

        KWARGS:
        binsize, bins: for bin_asts if not already run.

        RETURNS:
        cor_mag1, cor_mag2: ast corrected magnitudes

        RAISES:
        returns -1 if obs_mag1 and obs_mag2 are different sizes

        Corrections are made by going through obs_mag1 in bins of
        bin_asts and randomly selecting magdiff values in that ast_bin.
        obs_mag2 simply follows along since it is tied to obs_mag1.

        Random selection was chosen because of the spatial nature of
        artificial star tests. If there are 400 asts in one mag bin,
        and 30 are not recovered, random selection should match the
        distribution (if there are many obs stars).

        If there are obs stars in a mag bin where there are no asts,
        will throw the star out unless the completeness in that mag bin
        is more than 50%.

        TODO:
        possibly return magXdiff rather than magX + magXdiff?
        reason not to: using AST results from one filter to another isn't
        kosher. At least not glatt kosher.
        '''
        self.completeness(combined_filters=True, interpolate=True)

        nstars = obs_mag1.size
        if obs_mag1.size != obs_mag2.size:
            logger.error('mag arrays of different lengths')
            return -1

        # corrected mags are filled with nan.
        cor_mag1 = np.empty(nstars)
        cor_mag1.fill(not_rec_val)
        cor_mag2 = np.empty(nstars)
        cor_mag2.fill(not_rec_val)

        # need asts to be binned for this method.
        if not hasattr(self, 'ast_bins'):
            self.bin_asts(binsize=binsize, bins=bins)
        om1_inds = np.digitize(obs_mag1, self.ast_bins)

        for i in range(len(self.ast_bins)):
            # the obs and artificial stars in each bin
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
                if self.fcomp2(self.ast_bins[i]) < 0.5:
                    continue
                else:
                    # model is producing stars where there was no data.
                    # assign no corrections.
                    cor1 = 0.
                    cor2 = 0.
            else:
                # randomly select the appropriate ast correction for obs stars
                # in this bin
                cor1 = self._random_select(self.mag1diff[astbin], nobs)
                cor2 = self._random_select(self.mag2diff[astbin], nobs)

            # apply corrections
            cor_mag1[obsbin] = obs_mag1[obsbin] + cor1
            cor_mag2[obsbin] = obs_mag2[obsbin] + cor2
            # finite values only: not implemented because trilegal array should
            # maintain the same size.
            #fin1, = np.nonzero(np.isfinite(cor_mag1))
            #fin2, = np.nonzero(np.isfinite(cor_mag2))
            #fin = list(set(fin1) & set(fin2))
        return cor_mag1, cor_mag2

    def completeness(self, combined_filters=False, interpolate=False):
        '''
        calculate the completeness of the data in each filter
        ARGS:
        combined_filters: Use individual or combined ast recovery
        interpolate: add a 1d spline the completeness function to self
        RETURNS:
        self.comp1 the completeness in filter1 binned with self.ast_bins
        self.comp2 same as above but filter2
        '''
        # calculate stars recovered, could pass theshold here.
        rec1, rec2 = self.recovered()

        # make sure ast_bins are good to go
        if not hasattr(self, 'ast_bins'):
            self.bin_asts()

        # gst uses both filters for recovery.
        if combined_filters is True:
            rec1 = rec2 = self.rec

        # historgram of all artificial stars
        qhist1 = np.array(np.histogram(self.mag1, bins=self.ast_bins)[0],
                          dtype=float)

        # histogram of recovered artificial stars
        rhist1 = np.array(np.histogram(self.mag1[rec1], bins=self.ast_bins)[0],
                          dtype=float)

        # completeness histogram
        self.comp1 = rhist1 / qhist1

        qhist2 = np.array(np.histogram(self.mag2, bins=self.ast_bins)[0],
                          dtype=float)
        rhist2 = np.array(np.histogram(self.mag2[rec2], bins=self.ast_bins)[0],
                          dtype=float)
        self.comp2 = rhist2 / qhist2

        if interpolate is True:
            # sometimes the histogram isn't as useful as the a spline
            # function... add the interp1d function to self.
            self.fcomp1 = interp1d(self.ast_bins[1:], self.comp1,
                                   bounds_error=False)
            self.fcomp2 = interp1d(self.ast_bins[1:], self.comp2,
                                   bounds_error=False)
        return

    def get_completeness_fraction(self, frac, dmag=0.01):
        assert hasattr(self, 'fcomp1'), \
            'need to run completeness with interpolate=True'

        # set up array to evaluate interpolation
        arr_min = 16
        arr_max = 31
        search_arr = np.arange(arr_min, arr_max, dmag)

        # completeness in each filter, and the finite vals
        # (frac - nan = frac)
        cfrac1 = self.fcomp1(search_arr)
        ifin1 = np.isfinite(cfrac1)

        cfrac2 = self.fcomp2(search_arr)
        ifin2 = np.isfinite(cfrac2)

        # closest completeness fraction to passed fraction
        icomp1 = np.argmin(np.abs(frac - cfrac1[ifin1]))
        icomp2 = np.argmin(np.abs(frac - cfrac2[ifin2]))

        # mag associated with completeness
        comp1 = search_arr[ifin1][icomp1]
        comp2 = search_arr[ifin2][icomp2]

        # sanity check... sometimes with few asts at bright mags the curve
        # starts near zero, not 1, get a bright mag limit. This makes sure
        # the completeness limit is past the half way point ... a bit of a
        # hack.
        if icomp1 < len(search_arr)/2.:
            print 'filter1 AST completeness is too bright, sanity checking.'
            cut_ind1 = np.argmax(cfrac1[ifin1])
            icomp1 = np.argmin(np.abs(frac - cfrac1[ifin1][cut_ind1:]))
            comp1 = search_arr[ifin1][cut_ind1:][icomp1]

        if icomp2 < len(search_arr)/2.:
            print 'filter2 AST completeness is too bright, sanity checking.'
            cut_ind2 = np.argmax(cfrac2[ifin2])
            icomp2 = np.argmin(np.abs(frac - cfrac2[ifin2][cut_ind2:]))
            comp2 = search_arr[ifin2][cut_ind2:][icomp2]
        
        return comp1, comp2

def stellar_prob(obs, model, normalize=False):
    '''
    FROM MATCH README
    The quality of the fit is calculated using a Poisson maximum likelihood
    statistic, based on the Poisson equivalent of chi^2.
      2 m                                if (n=0)
      2 [ 0.001 + n * ln(n/0.001) - n ]  if (m<0.001)
      2 [ m + n * ln(n/m) - n ]          otherwise
    m=number of model points; n=number of observed points

    This statistic is based on the Poisson probability function:
       P =  (e ** -m) (m ** n) / (n!),
    Recalling that chi^2 is defined as -2lnP for a Gaussian distribution and
    equals zero where m=n, we treat the Poisson probability in the same
    manner to get the above formula.

    '''
    n = obs
    m = model

    if normalize is True:
        n /= np.sum(n)
        m /= np.sum(m)

    d = 2. * (m + n * np.log(n / m) - n)

    smalln = np.abs(n) < 1e-10
    d[smalln] = 2. * m[smalln]

    smallm = (m < 0.001) & (n != 0)
    d[smallm] = 2. * (0.001 + n[smallm] * np.log(n[smallm]/0.001) - n[smallm])

    sig = np.sqrt(d) * np.sign(n - m)
    pct_dif = (m - n) / n
    prob = np.sum(d)/float(len(n)-1)
    return prob, pct_dif, sig
