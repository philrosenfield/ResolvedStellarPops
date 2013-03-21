# Strangers wrote
import os
import sys
import numpy as np
import matplotlib.nxutils as nxutils
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import FancyArrow
from subprocess import PIPE, Popen

import itertools
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
                 scatter_off=False, levels=20, threshold=10, contour_lw={},
                 color_by_arg_kw={}, filter1=None, filter2=None, slice_inds=None):
        set_fig = 0
        set_ax = 0
        if fig is None:
            fig = plt.figure(figsize=(8, 8))
            set_fig = 1
        if ax is None:
            ax = plt.axes()
            set_ax = 1
        if hasattr(self, 'filter2'):
            filter2 = self.filter2
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
                                'edgecolors': None, 'zorder': 1}.items() +
                                scatter_args.items())
            contour_lw = dict({'linewidths': 2, 'colors': 'white',
                               'zorder': 200}.items() + contour_lw.items())
            ncolbin = int(np.diff((np.nanmin(color), np.nanmax(color))) / 0.05)
            nmagbin = int(np.diff((np.nanmin(mag), np.nanmax(mag))) / 0.05)
            plt_pts, cs = scatter_contour(color, mag,
                                          threshold=threshold, levels=levels,
                                          hist_bins=[ncolbin, nmagbin],
                                          contour_args=contour_args,
                                          scatter_args=scatter_args,
                                          contour_lw=contour_lw,
                                          ax=ax)
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

    def decorate_cmd(self, mag1_err=None, mag2_err=None, trgb=False, ax=None):
        self.redding_vector()
        self.cmd_errors()
        self.text_on_cmd()
        if trgb is True:
            self.put_a_line_on_it(ax, self.trgb)

    def put_a_line_on_it(self, ax, val, consty=True, color='black',
                         ls='--', lw=2, annotate=True,
                         annotate_fmt='$TRGB=%.2f$'):
        """
        if consty is True: plots a constant y value across ax.xlims().
        if consty is False: plots a constant x on a plot of y vs x-y
        """
        (xmin, xmax) = ax.get_xlim()
        (ymin, ymax) = ax.get_ylim()
        xarr = np.linspace(xmin, xmax, 20)
        # y axis is magnitude...
        yarr = np.linspace(ymax, ymin, 20)
        if consty is True:
            # just a contsant y value over the plot range of x.
            new_yarr = np.repeat(val, len(xarr))
            new_xarr = xarr
        if consty is False:
            # a plot of y vs x-y and we want to mark
            # where a constant value of x is
            new_yarr = yarr
            # if you make the ability to make the yaxis filter1...
            #if filter1 == None:
            #    new_xarr = val - yarr
            new_xarr = val - yarr
        ax.plot(new_xarr, new_yarr, ls, color=color, lw=lw)
        if annotate is True:
            ax.annotate(annotate_fmt % val, xy=(new_xarr[-1]-0.1,
                        new_yarr[-1]-0.2), ha='right', fontsize=16,
                        **rspgraph.load_ann_kwargs())

    def redding_vector(self):
        Afilt1 = astronomy_utils.parse_mag_tab(self.photsys, self.filter1)
        Afilt2 = astronomy_utils.parse_mag_tab(self.photsys, self.filter2)
        Rslope = Afilt2 / (Afilt1 - Afilt2)
        dmag = 1.
        dcol = dmag / Rslope
        pstart = np.array([0., 0.])
        pend = pstart + np.array([dcol, dmag])
        points = np.array([pstart, pend])
        data_to_display = self.ax.transData.transform
        display_to_axes = self.ax.transAxes.inverted().transform
        ax_coords = display_to_axes(data_to_display(points))
        dy_ax_coords = ax_coords[1, 1] - ax_coords[0, 1]
        dx_ax_coords = ax_coords[1, 0] - ax_coords[0, 0]
        arr = FancyArrow(0.05, 0.95, dx_ax_coords, (1.)*dy_ax_coords,
                         transform=self.ax.transAxes, color='black', ec="none",
                         width=.005, length_includes_head=1, head_width=0.02)
        self.ax.add_patch(arr)

    def cmd_errors(self, binsize=0.1, errclr=-1.5):
        import pyfits
        if type(self.data) == pyfits.fitsrec.FITS_rec:
            self.mag1err = self.data.MAG1_ERR
            self.mag2err = self.data.MAG2_ERR
        nbins = (np.max(self.mag2) - np.min(self.mag2)) / binsize
        errmag = np.arange(int(nbins / 5) - 1) * 1.
        errcol = np.arange(int(nbins / 5) - 1) * 1.
        errmagerr = np.arange(int(nbins / 5) - 1) * 1.
        errcolerr = np.arange(int(nbins / 5) - 1) * 1.
        for q in range(len(errmag) - 1):
            test = self.mag2.min() + 5. * (q + 2) * binsize + 2.5 * binsize
            test2, = np.nonzero((self.mag2 > test - 2.5 * binsize) &
                               (self.mag2 <= test + 2.5 * binsize) &
                               (self.mag1 - self.mag2 > -0.5) &
                               (self.mag1 - self.mag2 < 2.5))
            if len(test2) < 5:
                continue
            errmag[q] = self.mag2.min() + 5. * (q + 2) * binsize \
                + 2.5 * binsize
            errcol[q] = errclr
            m2inds, = np.nonzero((self.mag2 > errmag[q] - 2.5 * binsize) &
                                (self.mag2 < errmag[q] + 2.5 * binsize))
            cinds, = np.nonzero((self.color > -0.5) & (self.color < 2.5))
            cinds = list(set(m2inds) & set(cinds))
            errmagerr[q] = np.mean(self.mag2err[m2inds])
            errcolerr[q] = np.sqrt(np.mean(self.mag1err[cinds] ** 2 +
                                           self.mag2err[cinds] ** 2))
        self.ax.errorbar(errcol, errmag, xerr=errcolerr, yerr=errmagerr,
                         ecolor='white', lw=3, capsize=0, fmt=None)
        self.ax.errorbar(errcol, errmag, xerr=errcolerr, yerr=errmagerr,
                         ecolor='black', lw=2, capsize=0, fmt=None)

    def text_on_cmd(self):
        #an_kw = rspgraph.load_ann_kwargs()
        strings = '$%s$ $\mu=%.3f$ $A_v=%.2f$' % (self.target, self.dmod,
                                                  self.Av)
        offset = 0.15
        for string in strings.split():
            offset -= 0.04
            self.ax.text(0.95, offset, string, transform=self.ax.transAxes,
                         ha='right', fontsize=16, color='black')

    def all_stages(self, *stages):
        '''
        adds the indices of some stage as an attribute.
        '''
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

        if check[0]:
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


class galaxies(star_pop):
    '''
    wrapper for lists of galaxy objects, each method returns lists, unless they
    are setting attributes.
    '''
    def __init__(self, galaxy_objects):
        self.galaxies = galaxy_objects
        # this will break if more than one filter1 or filter2... is that
        # how I want it?
        self.filter1, = np.unique([g.filter1 for g in galaxy_objects])
        self.filter2, = np.unique([g.filter2 for g in galaxy_objects])

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
    def __init__(self, fname, filetype='tagged_phot', hla=True, angst=True,
                 band=None, photsys=None, trgb=np.nan, z=-99, Av=None, dmod=None):
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
        if hla is True:
            self.survey, self.propid, self.target, filts, psys = \
                hla_galaxy_info(fname)
            # photometry
            self.filter1, self.filter2 = filts.upper().split('-')
            self.photsys = psys.replace('-', '_')
        else:
            self.survey = ' '
            self.photsys = photsys
            if self.photsys is None:
                self.photsys = 'wfc3'
                logger.warning('assuming this is wfc3 data')
            self.propid, self.target, self.filter1, self.filter2 =  \
                bens_fmt_galaxy_info(fname)
        if 'fits' in filetype:
            import pyfits
            hdu = pyfits.open(fname)
            #self.data =  fileIO.read_fits(fname)
            ext = self.photsys.upper().split('_')[0]
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
                if band is not None:
                    ext = band.upper()
                self.mag1 = self.data['mag1_%s' % ext]
                self.mag2 = self.data['mag2_%s' % ext]
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
            import pyfits
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
        else:
            logger.error('filetype must be fitstable, tagged_phot or match_phot use simgalaxy for trilegal')
            sys.exit(2)
        self.color = self.mag1 - self.mag2
        # angst table loads
        if angst is True:
            if self.photsys == 'acs_wfc' or self.photsys == 'wfpc2':
                self.comp50mag1 = angst_data.get_50compmag(self.target,
                                                           self.filter1)
                self.comp50mag2 = angst_data.get_50compmag(self.target,
                                                           self.filter2)
            self.trgb_av_dmod(band=band)
            # Abs mag
            self.convert_mag(dmod=self.dmod, Av=self.Av, target=self.target)
            #self.z = galaxy_metallicity(self, self.target)

    def trgb_av_dmod(self, band=None):
        '''
        returns trgb, av, dmod from angst table
        TODO: move the wfc3 crap to angst_tables and make an IR table.
        '''
        if band == 'ir':
            table = '/Users/phil/research/TP-AGBcalib/SNAP/tables/table.dat'
            with open(table, 'r') as t:
                lines = t.readlines()
            for line in lines:
                line = line.strip().split()
                if line[0].strip() == self.target:
                    self.dmod, self.Av = map(float, line[1:3])

            table = '/Users/phil/research/TP-AGBcalib/SNAP/tables/IR_NAGBs.dat'
            with open(table, 'r') as t:
                lines = t.readlines()
            for line in lines:
                line = line.strip().split(', ')
                if line[0].split('_')[1] == self.target:
                    self.trgb = float(line[1])
        else:
            filters = ','.join((self.filter1, self.filter2))
            (self.trgb, self.Av, self.dmod) = \
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

    def normalize(self, stage_lab, mag2=None, stage=None, by_stage=True,
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

        new_attr = '%s_norm' % stage_lab.lower()

        if useasts is True:
            if not hasattr(self, 'ast_mag2'):
                self.load_ast_corrections()
            smag2 = self.ast_mag2[self.rec]
            scolor = self.ast_color[self.rec]
        else:
            smag2 = self.mag2
            scolor = self.mag1 - self.mag2

        # initialize slices
        ibright = np.arange(smag2.size)
        st_inds = np.arange(smag2.size)
        reg_inds = np.arange(smag2.size)
        if sinds_cut is None:
            sinds_cut = np.arange(smag2.size)

        # mag cut
        ibright, = np.nonzero(smag2 < magcut)

        # stage cut
        if by_stage is True:
            stage = 'i%s' % stage_lab
            if not hasattr('self', stage):
                self.all_stages(stage_lab)
            st_inds = self.__dict__[stage]

        # cmd space cut
        if verts is not None:
            points = np.column_stack((scolor, smag2))
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
        subplots_kwargs = {'sharex': 1, 'sharey': 1, 'figsize': (12, 8)}
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
                             leo_method=False):
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
    assert fake_file is not None, \
        'ast_correct_trilegal_sim: fake_file now needs to be passed'

    if leo_method is True:
        assert spread_outfile is not None, \
            'need spread_outfile set for Leo AST method'

        logger.info("completeness using Leo's method")

        sim_file = os.path.join(sgal.base, sgal.name)
        # <hit enter>
        EOF = os.path.join(os.environ['PYTHONPATH'], 'EOF')

        # special version of spread_angst made for wfc3snap photsys!
        leo_code = '/Users/phil/research/TP-AGBcalib/SNAP/models/spread_angst'

        cmd = '%s << %s\n' % (leo_code, EOF)
        cmd += '\n'.join((fake_file, sim_file, spread_outfile, EOF))

        logger.debug("%s + %s -> %s" % (sim_file, fake_file, spread_outfile))
        logger.debug('Running spread_angst...')

        p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE,
                  close_fds=True)

        stdout, stderr = (p.stdout, p.stderr)
        p.wait()
    else:
        if type(fake_file) is str:
            asts = artificial_star_tests(fake_file)
            sgal.fake_file = fake_file
        else:
            asts = fake_file
        if sgal.filter1 != asts.filter1 or sgal.filter2 != asts.filter2:
            logger.error('bad filter match between sim gal and ast.')
            return -1
        mag1 = sgal.mag1
        mag2 = sgal.mag2
        cor_mag1, cor_mag2 = asts.ast_correction(mag1, mag2, **{'binsize': 0.2})
        new_cols = {'%s_cor' % asts.filter1: cor_mag1,
                    '%s_cor' % asts.filter2: cor_mag2}
        sgal.add_data(**new_cols)
        if outfile is not None:
            if overwrite or not os.path.isfile(outfile):
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
                __, self.target, __, self.filter1, self.filter2, _ = \
                    self.name.split('_')
        artificial_star_tests.load_fake(self, filename)

    def recovered(self, threshold=9.99):
        '''
        indicies of recovered stars in both filters.
        threshold of recovery [9.99]
        '''
        rec1, = np.nonzero(self.mag1diff > threshold)
        rec2, = np.nonzero(self.mag2diff > threshold)
        self.rec = list(set(rec1) & set(rec2))
        return self.rec

    def load_fake(self, filename):
        '''
        reads matchfake file and assigns each column to its own attribute
        see artificial_star_tests.__doc__
        '''
        names = ['mag1', 'mag2', 'mag1diff', 'mag2diff']
        self.data = np.genfromtxt(filename, names=names)
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

    def ast_correction(self, obs_mag1, obs_mag2, binsize=0.2, bins=None,
                       not_rec_val=np.nan):
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
        cor_mag1.fill(not_rec_val)
        cor_mag2 = np.empty(nstars)
        cor_mag2.fill(not_rec_val)

        # need asts to be binned for this method.
        if not hasattr(self, 'ast_bins'):
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
