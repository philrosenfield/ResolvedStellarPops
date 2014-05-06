""" All about Artificial star tests """
from __future__ import print_function
import os
import numpy as np

import logging
logger = logging.getLogger()

from scipy.interpolate import interp1d
from subprocess import PIPE, Popen

from .. import trilegal
from .. import fileio
from .. import astronomy_utils
__all__ = ['ast_correct_trilegal_sim', 'ASTs']


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
        spout = io.readfile(spread_outfile)
        ast_filts = [s for s in spout.dtype.names if '_cor' in s]
        new_cols = {}
        for ast_filt in ast_filts:
            new_cols[ast_filt] = spout[ast_filt]
            print(len(spout[ast_filt]))
        print(len(sgal.mag2))
        print(sgal.add_data(**new_cols))

    #assert fake_file is not None and asts_obj is not None, \
    #    'ast_correct_trilegal_sim: fake_file now needs to be passed'

    if leo_method is True:
        print('ARE YOU SURE YOU WANT TO USE THIS METHOD!?')
        # this method tosses model stars where there are no ast corrections
        # which is fine for completeness < .50 but not for completeness > .50!
        # so don't use it!! It's also super fucking slow.
        if spread_outfile is None:
            raise ValueError('need spread_outfile set for Leo AST method')

        print("completeness using Leo's method")

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
            asts_obj = [ASTs(fake_file) for fake_file in fake_files]
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
                trilegal.write_trilegal_sim(sgal, outfile)
            else:
                logger.warning('%s exists, not overwriting' % outfile)

        if spread_outfile is not None:
            trilegal.write_spread(sgal, outfile=spread_outfile, overwrite=overwrite)


class ASTs(object):
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
        self.load_fake(filename)

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

    def make_hess(self, binsize=0.1, yattr='mag2diff', hess_kw={}):
        '''
        make hess grid
        '''
        self.colordiff = self.mag1diff - self.mag2diff
        mag = self.__getattribute__(yattr)
        self.hess = astronomy_utils.hess(self.colordiff, mag, binsize,
                                         **hess_kw)

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
