from __future__ import print_function
import numpy as np
import pylab as plt
from scipy import integrate
#import pyfits

from ..graphics import scatter_contour, colorify, make_hess, plot_hess, plot_cmd_redding_vector
from ..tools import get_dmodAv, mag2Mag, Mag2mag
from ..angst_tables import angst_data
from .. import utils
from .. import fileio

__all__ = ['StarPop', 'plot_cmd', 'color_by_arg' , 'stars_in_region']


class StarPop(object):
    def __init__(self):
        self.data = None

    def plot_cmd(self, *args, **kwargs):
        '''
        plot a stellar color-magnitude diagram
        see :func: plot_cmd
        '''
        return plot_cmd(self, *args, **kwargs)

    def color_by_arg(self, *args, **kwargs):
        """
        see :func: color_by_arg
        """
        return color_by_arg(self, *args, **kwargs)

    def redding_vector(self, dmag=1., **kwargs):
        """ Add an arrow to show the reddening vector """

        return plot_cmd_redding_vector(self.filter1, self.filter2,
                                       self.photsys, dmag=dmag, **kwargs)

    def decorate_cmd(self, mag1_err=None, mag2_err=None, trgb=False, ax=None,
                     reddening=True, dmag=0.5, text_kw={}, errors=True,
                     cmd_errors_kw={}, filter1=None, text=True):
        """ add annotations on the cmd, such as reddening vector, typical errors etc
        """

        self.redding_vector(dmag=dmag, ax=ax)

        if errors is True:
            cmd_errors_kw['ax'] = ax
            self.cmd_errors(**cmd_errors_kw)

        self.text_on_cmd(ax=ax, **text_kw)

        if trgb is True:
            if filter1 is None:
                self.put_a_line_on_it(ax, self.trgb)
            else:
                self.put_a_line_on_it(ax, self.trgb, filter1=filter1, consty=False)

    def put_a_line_on_it(self, ax, val, consty=True, color='black',
                         ls='--', lw=2, annotate=True, filter1=None,
                         annotate_fmt='$TRGB=%.2f$', **kwargs):
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
            xy = (new_xarr[-1] - 0.1, yarr[-1] - 0.2)
            ax.annotate(annotate_fmt % val, xy=xy, ha='right', fontsize=16, **kwargs)

    def text_on_cmd(self, extra=None, ax=None, distance_av=True, **kwargs):
        """ add a text on the cmd
        """
        ax = ax or plt.gca()

        if distance_av is True:
            strings = '$%s$ $\mu=%.3f$ $A_v=%.2f$' % (self.target.upper(), self.dmod, self.Av)
            offset = .17
        else:
            strings = '$%s$' % self.target.upper().replace('-DEEP', '').replace('-', '\!-\!')
            offset = .09
        if extra is not None:
            strings += ' %s' % extra
            offset = 0.2
        for string in strings.split():
            offset -= 0.04
            ax.text(0.95, offset, string, transform=ax.transAxes, ha='right',
                    fontsize=16, color='black', **kwargs)

    def annotate_cmd(self, yval, string, offset=0.1, text_kw={}, ax=None):
        ax = ax or plt.gca()
        ax.text(ax.get_xlim()[0] + offset, yval - offset, string, **text_kw)

    def make_hess(self, binsize, absmag=False, useasts=False, slice_inds=None,
                  **kwargs):
        '''
        adds a hess diagram of color, mag2 or Color, Mag2 (if absmag is True).
        if useasts is true will use ast_mags.

        slice_inds will slice the arrays and only bin those stars, there
        is no evidence besides the hess tuple itself that hess is not of the
        full cmd.

        See `:func: helpers.make_hess`
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
        self.hess = make_hess(col, mag, binsize, **kwargs)
        return

    def hess_plot(self, fig=None, ax=None, colorbar=False, **kwargs):
        '''
        Plots a hess diagram with imshow.

        See `:func: helpers.plot_hess`
        '''
        if not hasattr(self, 'hess'):
            raise AttributeError('run self.make_hess before plotting')

        if hasattr(self, 'filter2') and hasattr(self, 'filter1'):
            filter1 = self.filter1
            filter2 = self.filter2
        else:
            filter1 = None
            filter2 = None

        ax = plot_hess(self.hess, ax=ax, filter1=filter1, filter2=filter2,
                       colorbar=colorbar, **kwargs)
        return ax

    def get_header(self):
        '''
        utility for writing data files, sets header attribute and returns
        header string.
        '''
        names = [k for k, v in sorted(self.key_dict.items(), key=lambda (k,v): v)]
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
            if hasattr(self, data_name):
                self.__delattr__(data_name)
            if hasattr(self, data_name.title()):
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
                print('converting mags with angst table using %s' % target)
                self.target = target
            elif hasattr(self, 'target'):
                print('converting mags with angst table using initialized %s' % self.target)

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
                old_dmod, old_Av = get_dmodAv(self)
                old_mag_covert_kw = {'Av': old_Av, 'dmod': old_dmod}
                M1 = mag2Mag(am1, self.filter1, self.photsys, **old_mag_covert_kw)
                M2 = mag2Mag(am2, self.filter2, self.photsys, **old_mag_covert_kw)
            else:
                M1 = self.data.get_col(self.filter1)
                M2 = self.data.get_col(self.filter2)
            self.mag1 = Mag2mag(M1, self.filter1, self.photsys, **mag_covert_kw)
            self.mag2 = Mag2mag(M2, self.filter2, self.photsys, **mag_covert_kw)
            self.color = self.mag1 - self.mag2
        else:
            if hasattr(self, 'mag1'):
                self.Mag1 = mag2Mag(self.mag1, self.filter1, self.photsys, **mag_covert_kw)
                self.Mag2 = mag2Mag(self.mag2, self.filter2, self.photsys, **mag_covert_kw)
                self.Color = self.Mag1 - self.Mag2

                if hasattr(self, 'trgb'):
                    self.Trgb = mag2Mag(self.trgb, self.filter2, self.photsys, **mag_covert_kw)

            if hasattr(self, 'Mag1'):
                self.mag1 = Mag2mag(self.Mag1, self.filter1, self.photsys, **mag_covert_kw)
                self.mag2 = Mag2mag(self.Mag2, self.filter2, self.photsys, **mag_covert_kw)
                self.color = self.mag1 - self.mag2

    def add_data(self, names, data):
        '''
        add columns to self.data, update self.key_dict
        see numpy.lib.recfunctions.append_fields.__doc__

        Parameters
        ----------
        names : string, sequence
            String or sequence of strings corresponding to the names
            of the new fields.
        data : array or sequence of arrays
            Array or sequence of arrays storing the fields to add to the base.

        Returns
        -------
        header
        '''
        import numpy.lib.recfunctions as nlr
        data = nlr.append_fields(self.data, names, data).data
        self.data = data.view(np.recarray)

        # update key_dict
        header = self.get_header()
        header += ' ' + ' '.join(names)
        col_keys = header.replace('#', '').split()
        self.key_dict = dict(zip(col_keys, range(len(col_keys))))
        return header

    def slice_data(self, keys, inds):
        '''slice already set attributes by some index list.'''
        for d in keys:
            if hasattr(self, d):
                self.__setattr__(d, self.__dict__[d][inds])
            if hasattr(self, d.title()):
                d = d.title()
                self.__setattr__(d, self.__dict__[d][inds])

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
        all_inds, = np.nonzero(utils.points_inside_poly(points, all_verts))

        if len(all_inds) <= thresh:
            print('not enough points found within verts')
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
        p0 = [np.nanmax(hist) / 2., np.mean(col_bins[1:]) - np.mean(col_bins[1:]) / 2., dcol,
              np.nanmax(hist) / 2., np.mean(col_bins[1:]) + np.mean(col_bins[1:]) / 2., dcol]

        mp_dg = utils.mpfit(utils.mp_double_gauss, p0, functkw=hist_in, quiet=True)
        if mp_dg.covar is None:
            print('not double gaussian')
            return 0., 0., poission_noise, float(len(all_inds)), color_sep
        else:
            perc_err = (np.array(mp_dg.perror) - np.array(mp_dg.params)) / np.array(mp_dg.params)
            if np.sum([p ** 2 for p in perc_err]) > 10.:
                print('not double guassian, errors too large')
                return 0., 0., poission_noise, float(len(all_inds)), color_sep
        # take fit params and apply to guassians on an arb color scale
        color_array = np.linspace(col_bins[0], col_bins[-1], 1000)
        g_p1 = mp_dg.params[0: 3]
        g_p2 = mp_dg.params[3:]
        gauss1 = utils.gaussian(color_array, g_p1)
        gauss2 = utils.gaussian(color_array, g_p2)
        print(g_p1[1], g_p2[1])
        # color separatrion is the intersection of the two gaussians..
        #double_gauss = gauss1 + gauss2
        #between_peaks = np.arange(
        min_locs = utils.find_peaks(gauss1 + gauss2)['minima_locations']
        g1, g2 = np.sort([g_p1[1], g_p2[2]])
        ginds, = np.nonzero( (color_array > g1) & (color_array < g2))
        #ginds2, = np.nonzero(gauss2)
        #ginds = list(set(ginds1) & set(ginds2))
        min_locs = np.argmin(np.abs(gauss1[ginds] - gauss2[ginds]))
        print(min_locs)
        auto_color_sep = color_array[ginds][min_locs]
        print(auto_color_sep)
        if auto_color_sep == 0:
            auto_color_sep = np.mean(col_bins[1:])
            print('using mean as color_sep')
        if color_sep is None:
            color_sep = auto_color_sep
        else:
            print('you want color_sep to be %.4f, I found it at %.4f' % (color_sep, auto_color_sep))

        # find contamination past the color sep...
        g12_Integral = integrate.quad(utils.double_gaussian, -np.inf, np.inf, mp_dg.params)[0]

        try:
            norm = float(len(all_inds)) / g12_Integral
        except ZeroDivisionError:
            norm = 0.

        g1_Integral = integrate.quad(utils.gaussian, -np.inf, np.inf, g_p1)[0]
        g2_Integral = integrate.quad(utils.gaussian, -np.inf, np.inf, g_p2)[0]

        g1_Int_colsep = integrate.quad(utils.gaussian, -np.inf, color_sep, g_p1)[0]
        g2_Int_colsep = integrate.quad(utils.gaussian, color_sep, np.inf, g_p2)[0]

        left_in_right = (g1_Integral - g1_Int_colsep) * norm
        right_in_left = (g2_Integral - g2_Int_colsep) * norm
        # diagnostic
        #print color_sep
        if diag_plot is True:
            fig1, ax1 = plt.subplots()
            ax1.plot(col_bins[1:], hist, ls='steps', lw=2)
            ax1.plot(col_bins[1:], hist, 'o')
            ax1.plot(color_array, utils.double_gaussian(color_array, mp_dg.params))
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
            ax1.text(0.1, 0.95, 'left in right: %i' % left_in_right, transform=ax1.transAxes)
            ax1.text(0.1, 0.90, 'right in left: %i' % right_in_left, transform=ax1.transAxes)

            fig_fname = 'heb_contamination_%s_%s_%s_mag2_%.2f.png' % (self.filter1, self.filter2, self.target, np.mean(np.array(all_verts)[:, 1]))

            fig1.savefig(fig_fname)
            print('wrote {0:s}'.format(fig_fname))

        return left_in_right, right_in_left, poission_noise, float(len(all_inds)), color_sep

    def stars_in_region(self, *args, **kwargs):
        '''
        counts stars in a region.
        see :func: stars_in_region
        '''
        return stars_in_region(*args, **kwargs)

    def write_data(self, outfile, overwrite=False):
        '''call fileio.savetxt to write self.data'''
        fileio.savetxt(outfile, self.data, fmt='%5g', header=self.get_header(),
                       overwrite=overwrite)
        return

def stars_in_region(mag2, mag_dim, mag_bright, mag1=None, verts=None,
                    col_min=None, col_max=None):
    '''
    counts stars in a region of a CMD or LF

    Parameters
    ----------
    mag1, mag2 : array mag1 optional
        arrays of star mags. If mag1 is supplied stars are assumed to be in a
        CMD, not LF.

    mag_dim, mag_bright : float, float
        faint and bright limits of mag2.

    col_min, col_max : float, float optional
        color min and max of CMD box.

    verts : array
        array shape 2, of verticies of a CMD polygon to search within

    Returns
    -------
    inds : array
        indices of mag2 inside LF or
        indices of mag1 and mag2 inside CMD box or verts
    '''
    if verts is None:
        if col_min is None:
            return utils.between(mag2, mag_dim, mag_bright)
        else:
            verts = np.array([[col_min, mag_dim],
                              [col_min, mag_bright],
                              [col_max, mag_bright],
                              [col_max, mag_dim],
                              [col_min, mag_dim]])

    points = np.column_stack((mag1 - mag2, mag2))
    inds, = np.nonzero(utils.points_inside_poly(points, verts))
    return inds


def plot_cmd(starpop, color, mag, ax=None, xlim=None, ylim=None, xlabel=None,
             ylabel=None, contour_kwargs={}, scatter_kwargs={}, plot_kwargs={},
             scatter=True, levels=5, threshold=75, contour_lw={},
             color_by_arg_kw={}, slice_inds=None, hist_bin_res=0.05,
             log_counts=False):
    '''
    plot a stellar color-magnitude diagram

    Parameters
    ----------
    starpop: StarPop instance
        population to plot

    color: ndarray, dtype=float, ndim=1
        colors of the stars

    mag: ndarray, dtype=float, ndim=1
        magnitudes of the stars

    ax: plt.Axes instance, optional (default=plt.gca())
        axes in which it will make the plot

    xlim: optional
        if set, adjust the limits on the color axis

    ylim: optional
        if set, adjust the limits on the magnitude axis

    xlabel: str, optional
        label of the color axis

    ylabel: str, optional
        label of the magntiude axis

    contour_args: dict, optional
        keywords for contour plot

    scatter_args: dict, optional
        keywords for scatter

    plot_args: dict, optional
        kweywords for plot

    scatter: bool, optional (default=True)
        CMD will be generated with plt.scatter

    levels: optional (default=5)

    threshold: optional (default=75)

    contour_lw: optional

    color_by_arg_kw: optional

    slice_inds: slice instance, optional
        if set, cull the data and only consider this slice

    hist_bin_res: optional (default=0.05),

    log_counts: bool, optional (default=False)
    '''
    ax = ax or plt.gca()

    if slice_inds is not None:
        color = color[slice_inds]
        mag = mag[slice_inds]

    if xlim is None:
        xlim = (color.min(), color.max())

    if ylim is None:
        ylim = (mag.max(), mag.min())

    if len(color_by_arg_kw) != 0:
        scatter = False
        color_by_arg(starpop, ax=ax, **color_by_arg_kw)

    if scatter is True:
        _contour_kwargs = {'cmap': plt.cm.gray_r, 'zorder': 100}
        _contour_kwargs.update(**contour_kwargs)

        _scatter_kwargs = {'marker': '.', 'color': 'black', 'alpha': 0.2,
                           'edgecolors': 'none', 'zorder': 1}
        _scatter_kwargs.update(**scatter_kwargs)

        if type(hist_bin_res) is list:
            hist_bin_res_c, hist_bin_res_m = hist_bin_res
        else:
            hist_bin_res_c = hist_bin_res
            hist_bin_res_m = hist_bin_res

        ncolbin = int(np.diff((np.nanmin(color), np.nanmax(color))) / hist_bin_res_c)
        nmagbin = int(np.diff((np.nanmin(mag), np.nanmax(mag))) / hist_bin_res_m)
        #print(ncolbin, nmagbin)

        scatter_contour(color, mag, bins=[ncolbin, nmagbin], levels=levels,
                        threshold=threshold, log_counts=log_counts,
                        plot_args=_scatter_kwargs,
                        contour_args=_contour_kwargs, ax=ax)
    else:
        #simple plotting
        _plot_kwargs = {'marker': '.', 'color': 'black', 'mew': 0., 'lw': 0, 'rasterize': True}
        _plot_kwargs.update(**plot_kwargs)
        ax.plot(color, mag, **_plot_kwargs)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if plt.isinteractive():
        plt.draw()

    return ax

    def make_phot(self, fname='phot.dat'):
        '''
        makes phot.dat input file for match, a list of V and I mags.
        '''
        np.savetxt(fname, np.column_stack((self.mag1, self.mag2)), fmt='%.4f')


def make_match_param(gal, more_gal_kw=None):
    '''
    Make param.sfh input file for match
    see rsp.match_utils.match_param_fmt()

    takes calcsfh search limits to be the photometric limits of the stars in
    the cmd.
    gal is assumed to be angst galaxy, so make sure attr dmod, Av, comp50mag1,
    comp50mag2 are there.

    only set up for acs and wfpc, if other photsystems need to check syntax
    with match filters.

    All values passed to more_gal_kw overwrite defaults.
    '''

    more_gal_kw = more_gal_kw or {}

    # load parameters
    inp = fileio.input_parameters(default_dict=match_param_default_dict())

    # add parameteres
    cmin = gal.color.min()
    cmax = gal.color.max()
    vmin = gal.mag1.min()
    imin = gal.mag2.min()

    if 'acs' in gal.photsys:
        V = gal.filter1.replace('F', 'WFC')
        I = gal.filter2.replace('F', 'WFC')
    elif 'wfpc' in gal.photsys:
        V = gal.filter1.lower()
        I = gal.filter2.lower()
    else:
        print(gal.photsys, gal.name, gal.filter1, gal.filter2)

    # default doesn't move dmod or av.
    gal_kw = {'dmod1': gal.dmod, 'dmod2': gal.dmod, 'av1': gal.Av,
              'av2': gal.Av, 'V': V, 'I': I, 'Vmax': gal.comp50mag1,
              'Imax': gal.comp50mag2, 'V-Imin': cmin, 'V-Imax': cmax,
              'Vmin': vmin, 'Imin': imin}

    # combine sources of params
    phot_kw = dict(match_param_default_dict().items() \
                   + gal_kw.items() + more_gal_kw.items())

    inp.add_params(phot_kw)

    # write out
    inp.write_params('param.sfh', match_param_fmt())
    return inp


def color_by_arg(starpop, xcol, ycol, colorcol, bins=None, cmap=None, ax=None,
                 fig=None, labelfmt='$%.3f$', xdata=None, ydata=None,
                 coldata=None, xlim=None, ylim=None, slice_inds=None,
                 legend=True):
    """
    Parameters
    ----------
    starpop: StarPop instance
        population to plot

    xcol: ndarray

    ycol: ndarray

    colorcol: ndarray

    bins: optional

    cmap: optional
        colormap to use

    ax: plt.Axes instance, optional (default=plt.gca())

    labelfmt: str, optional (default='$%.3f$')
        colorbar label format

    xdata: optional

    ydata: optional

    coldata: optional

    xlim: optional
        if set, adjust the limits on the x-axis

    ylim: optional
        if set, adjust the limits on the y-axis

    slice_inds: slice instance, optional
        if set, cull the data and only consider this slice


    legend: bool, optional (default=True)
        if set, add a legend to the figure
    """
    ax = ax or plt.gca()

    if xlim is None:
        xlim = (xdata.min(), xdata.max())

    if ylim is None:
        ylim = (ydata.max(), ydata.min())

    if bins is None:
        bins = 10

    if xdata is None:
        xdata = starpop.data.get_col(xcol)

    if ydata is None:
        ydata = starpop.data.get_col(ycol)

    if colorcol is None:
        coldata = starpop.data.get_col(colorcol)

    if slice_inds is not None:
        xdata = xdata[slice_inds]
        ydata = ydata[slice_inds]
        coldata = coldata[slice_inds]

    # need the bins to be an array to use digitize.
    if type(bins) == int:
        hist, bins = np.histogram(coldata, bins=bins)
    inds = np.digitize(coldata, bins)
    colors, scalarmap = colorify(inds, cmap=cmap)
    lbls = [ labelfmt % bk for bk in bins ]  # bins are left bin edges.

    # fake out the legend...
    if labelfmt not in ['', 'None', None]:
        for color, label in zip(colors, lbls):
            ax.plot([999], [999], 'o', color=color, mec=color, label=label, visible=False)

    ax.scatter(xdata[inds], ydata[inds], marker='o', s=15, edgecolors='none',
               color=colors[inds])

    if legend is True:
        ax.legend(loc=0, numpoints=1, frameon=False)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return ax
