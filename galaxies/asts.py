""" All about Artificial star tests """
from __future__ import print_function
import os
import numpy as np
import matplotlib.pylab as plt
from scipy.interpolate import interp1d
import pyfits

from .. import astronomy_utils
__all__ = ['ast_correct_starpop', 'ASTs', 'parse_pipeline']

def parse_pipeline(filename, filter1=None, filter2=None):
    '''
    target, filter1, and filter2 are assigned:
    PID_TARGET_FILTER1_FILTER2_... or TARGET_FILTER1_FILTER2_
    '''
    name = os.path.split(filename)[1]
    if None in [filter1, filter2]:
        try:
            __, target, filter1, filter2, __ = name.split('_')
        except:
            try:
                target, filter1, filter2e = name.split('_')
                filter2 = filter2e.split('.')[0]
            except:
                try:
                    __, target, __, filter1, filter2, _ = name.split('_')
                except:
                    try:
                        __, target, filter1, filter2e = name.split('_')
                        filter2 = filter2e.split('.')[0]
                    except:
                        return None, None, None
    return target, filter1, filter2


def ast_correct_starpop(sgal, fake_file=None, outfile=None, overwrite=False,
                        asts_obj=None, correct_kw={}, diag_plot=False,
                        plt_kw={}, hdf5=True):
    '''
    correct mags with artificial star tests.

    Parameters
    ----------
    sgal : galaxies.SimGalaxy or StarPop instance
        must have apparent mags (corrected for dmod and Av)

    fake_file : string
         matchfake file

    outfile : string
        if sgal, a place to write the table with ast_corrections

    overwrite : bool
        if sgal and outfile, overwite if outfile exists

    asts_obj : AST instance
        if not loading from fake_file

    correct_kw : dict
        passed to ASTs.correct important to consider, dxy, xrange, yrange
        see AST.correct.__doc__

    diag_plot : bool
        make a mag vs mag diff plot

    plt_kw :
        kwargs to pass to pylab.plot

    Returns
    -------
    adds corrected mag1 and mag2

    If sgal, adds columns to sgal.data
    '''
    fmt = '%s_cor'
    sgal.fake_file = fake_file
    _, filter1, filter2 = parse_pipeline(fake_file)
    if hasattr(sgal.data, fmt % filter1) and hasattr(sgal.data,
                                                     fmt % filter2):
        errfmt = '%s, %s ast corrections already in file.'
        print(errfmt % (filter1, filter2))
        return sgal.data[fmt % filter1], sgal.data[fmt % filter2]

    if asts_obj is None:
        ast = ASTs(fake_file)
    else:
        ast = asts_obj

    mag1 = sgal.data[ast.filter1]
    mag2 = sgal.data[ast.filter2]

    correct_kw = dict({'dxy': (0.2, 0.15)}.items() + correct_kw.items())
    cor_mag1, cor_mag2 = ast.correct(mag1, mag2, **correct_kw)
    names = ['%s_cor' % ast.filter1, '%s_cor' % ast.filter2]
    data = [cor_mag1, cor_mag2]
    sgal.add_data(names, data)

    if outfile is not None:
        sgal.write_data(outfile, overwrite=overwrite, hdf5=hdf5)

    if diag_plot:
        from ..fileio.fileIO import replace_ext
        plt_kw = dict({'color': 'navy', 'alpha': 0.3, 'label': 'sim'}.items() \
                      + plt_kw.items())
        axs = ast.magdiff_plot()
        mag1diff = cor_mag1 - mag1
        mag2diff = cor_mag2 - mag2
        rec, = np.nonzero((np.abs(mag1diff) < 10) & (np.abs(mag2diff) < 10))
        axs[0].plot(mag1[rec], mag1diff[rec], '.', **plt_kw)
        axs[1].plot(mag2[rec], mag2diff[rec], '.', **plt_kw)
        if 'label' in plt_kw.keys():
            [ax.legend(loc=0, frameon=False) for ax in axs]
        plt.savefig(replace_ext(outfile, '_ast_correction.png'))
    return cor_mag1, cor_mag2

class ASTs(object):
    '''
    class for reading and using artificial stars.

    '''
    def __init__(self, filename, filter1=None, filter2=None, filt_extra=''):
        '''
        if filename has 'match' in it will assume this is a matchfake file.
        if filename has .fits extention will assume it's a binary fits table.
        '''
        self.base, self.name = os.path.split(filename)
        self.filter1 = filter1
        self.filter2 = filter2
        self.filt_extra = filt_extra

        self.target, self.filter1, self.filter2 = parse_pipeline(filename)
        self.read_file(filename)

    def recovered(self, threshold=9.99):
        '''
        find indicies of stars with magdiff < threshold

        ARGS:
        threshold: [9.99] magin - magout threshold for recovery

        RETURNS:
        self.rec: recovered stars in both filters
        rec1, rec2: recovered stars in filter1, filter2
        '''
        rec1, = np.nonzero(np.abs(self.mag1diff) < threshold)
        rec2, = np.nonzero(np.abs(self.mag2diff) < threshold)
        self.rec = list(set(rec1) & set(rec2))
        if len(self.rec) == len(self.mag1diff):
            print('warning: all stars recovered')
        return rec1, rec2

    def make_hess(self, binsize=0.1, yattr='mag2diff', hess_kw={}):
        '''
        make hess grid
        '''
        self.colordiff = self.mag1diff - self.mag2diff
        mag = self.__getattribute__(yattr)
        self.hess = astronomy_utils.hess(self.colordiff, mag, binsize,
                                         **hess_kw)

    def read_file(self, filename):
        '''
        read MATCH fake file into attributes
        mag1 is assumed to be mag1in
        mag2 is assumed to be mag2in
        mag1diff is assumed to be mag1in-mag1out
        mag2diff is assumed to be mag2in-mag2out
        '''
        if 'match' in filename:
            names = ['mag1', 'mag2', 'mag1diff', 'mag2diff']
            self.data = np.genfromtxt(filename, names=names)
            # unpack into attribues
            for name in names:
                self.__setattr__(name, self.data[name])
        elif filename.endswith('.fits'):
            assert not None in [self.filter1, self.filter2], \
                'Must specify filter strings'
            self.data = pyfits.getdata(filename)
            self.mag1 = self.data['%s_IN' % self.filter1]
            self.mag2 = self.data['%s_IN' % self.filter2]
            mag1out = self.data['%s%s' % (self.filter1, self.filt_extra)]
            mag2out = self.data['%s%s' % (self.filter2, self.filt_extra)]
            self.mag1diff = self.mag1 - mag1out
            self.mag2diff = self.mag2 - mag2out
        else:
            print(filename, 'not supported')

    def write_matchfake(self, newfile):
        '''write matchfake file'''
        dat = np.array([self.mag1, self.mag2, self.mag1diff, self.mag2diff]).T
        np.savetxt(newfile, dat, fmt='%.3f')

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
        randomly sample arr nselections times

        Parameters
        ----------
        arr : array or list
            input to sample
        nselections : int
            number of times to sample

        Returns
        -------
        rands : array
            len(nselections) of randomly selected from arr (duplicates included)
        '''
        rands = np.array([np.random.choice(arr) for i in range(nselections)])
        return rands

    def ast_correction(self, obs_mag1, obs_mag2, binsize=0.2, bins=None,
                       not_rec_val=np.nan, missing_data1=0., missing_data2=0.):
        '''
        apply ast correction to input mags.

        ARGS:
        obs_mag1, obs_mag2: N, 1 arrays

        KWARGS:
        binsize, bins: for bin_asts if not already run.
        not_rec_val: value for not recovered ast
        missing_data1: value for data outside ast limits (include=0)
        missing_data2: value for data outside ast limits (include=0)
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
            print('error: mag arrays of different lengths')
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
                    # assign correction for missing data
                    cor1 = missing_data1
                    cor2 = missing_data2
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

    def correct(self, obs_mag1, obs_mag2, bins=[100,200], xrange=[-0.5, 5.],
                yrange=[15., 27.], not_rec_val=0., dxy=None):
        """
        apply AST correction to obs_mag1 and obs_mag2

        Parameters
        ----------
        obs_mag1, obs_mag2 : arrays
            input mags to correct

        bins : [int, int]
            bins to pass to graphics.plotting.crazy_histogram2d

        xrange, yrange : shape 2, arrays
            limits of cmd space send to graphics.plotting.crazy_histogram2d
            since graphics.plotting.crazy_histogram2d is called twice it is
            important to have same bin sizes

        not_rec_val : float or nan
            value to fill output arrays where obs cmd does not overlap with
            ast cmd.

        dxy : array shape 2,
            color and mag step size to make graphics.plotting.crazy_histogram2d

        Returns
        -------
        cor_mag1, cor_mag2 : arrays len obs_mag1, obs_mag2
            corrections to obs_mag1 and obs_mag2
        """
        from ..graphics.plotting import crazy_histogram2d as chist

        nstars = obs_mag1.size
        if obs_mag1.size != obs_mag2.size:
            print('error: mag arrays of different lengths')
            return -1, -1

        # corrected mags are filled with nan.
        cor_mag1 = np.empty(nstars)
        cor_mag1.fill(not_rec_val)
        cor_mag2 = np.empty(nstars)
        cor_mag2.fill(not_rec_val)

        obs_color = obs_mag1 - obs_mag2
        ast_color = self.mag1 - self.mag2

        if dxy is not None:
            # approx number of bins.
            bins[0] = len(np.arange(*xrange, step=dxy[0]))
            bins[1] = len(np.arange(*yrange, step=dxy[1]))

        ckw = {'bins': bins, 'reverse_indices': True, 'xrange': xrange,
                    'yrange': yrange}
        SH, _, _, sixy, sinds = chist(ast_color, self.mag2, **ckw)
        H, _, _, ixy, inds = chist(obs_color, obs_mag2, **ckw)

        x, y = np.nonzero(SH * H > 0)
        # there is a way to do this with masking ...
        for i, j in zip(x, y):
            sind, = np.nonzero((sixy[:, 0] == i) & (sixy[:, 1] == j))
            hind, = np.nonzero((ixy[:, 0] == i) & (ixy[:, 1] == j))
            nobs = int(H[i, j])
            xinds = self._random_select(sinds[sind], nobs)
            cor_mag1[inds[hind]] = self.mag1diff[xinds]
            cor_mag2[inds[hind]] = self.mag2diff[xinds]

        return obs_mag1 + cor_mag1, obs_mag2 + cor_mag2

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

    def get_completeness_fraction(self, frac, dmag=0.01, guess=24):
        assert hasattr(self, 'fcomp1'), \
            'need to run completeness with interpolate=True'

        # set up array to evaluate interpolation
        arr_min = guess
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
        if icomp1 < len(search_arr) / 2.:
            print('filter1 AST completeness is too bright, sanity checking.')
            cut_ind1 = np.argmax(cfrac1[ifin1])
            icomp1 = np.argmin(np.abs(frac - cfrac1[ifin1][cut_ind1:]))
            comp1 = search_arr[ifin1][cut_ind1:][icomp1]

        if icomp2 < len(search_arr) / 2.:
            print('filter2 AST completeness is too bright, sanity checking.')
            cut_ind2 = np.argmax(cfrac2[ifin2])
            icomp2 = np.argmin(np.abs(frac - cfrac2[ifin2][cut_ind2:]))
            comp2 = search_arr[ifin2][cut_ind2:][icomp2]

        return comp1, comp2

    def magdiff_plot(self, axs=None):
        ''' not finished... plot some interesting stuff '''
        if not hasattr(self, 'rec'):
            self.completeness(combined_filters=True)
        fig, axs = plt.subplots(ncols=2, figsize=(12,8))
        axs[0].plot(self.mag1[self.rec], self.mag1diff[self.rec], '.', color='k')
        axs[1].plot(self.mag2[self.rec], self.mag2diff[self.rec], '.', color='k')
        xlab = r'${\rm Input}\ %s$'
        axs[0].set_xlabel(xlab % self.filter1, fontsize=20)
        axs[1].set_xlabel(xlab % self.filter2, fontsize=20)
        axs[0].set_ylabel(r'${\rm Input} - {\rm Ouput}$', fontsize=20)
        return axs

    def completeness_plot(self, axs=None):
        pass
