from __future__ import print_function
import numpy as np
import pprint
import utils
import os
import matplotlib.pylab as plt
from scipy.interpolate import splev, splprep
from critical_point import critical_point, inds_between_ptcris
#from ..mass_config import low_mass
low_mass = 1.25

class eep(object):
    '''
    a simple class to hold eep data. Gets added as an attribute to
    critical_point class.
    '''
    def __init__(self, eep_list=None, eep_lengths=None, eep_list_hb=None,
                 eep_lengths_hb=None):
        if eep_list is None:
            eep_list =  ['PMS_BEG', 'PMS_MIN',  'PMS_END', 'MS_BEG',
                         'MS_TMIN', 'MS_TO', 'SG_MAXL', 'RG_MINL',
                         'RG_BMP1', 'RG_BMP2', 'RG_TIP', 'HE_BEG',
                         'YCEN_0.550', 'YCEN_0.500', 'YCEN_0.400',
                         'YCEN_0.200', 'YCEN_0.100', 'YCEN_0.005',
                         'YCEN_0.000', 'TPAGB']
        if eep_lengths is None:
            eep_lengths = [60, 60, 60, 199, 100, 100, 70, 370, 30, 400,
                           10, 150, 100, 80, 100, 80, 80, 80, 100]
        if eep_list_hb is None:
            eep_list_hb = ['HB_BEG', 'YCEN_0.550', 'YCEN_0.500',
                           'YCEN_0.400', 'YCEN_0.200', 'YCEN_0.100',
                           'YCEN_0.005', 'YCEN_0.000', 'AGB_LY1',
                           'AGB_LY2']
        if eep_lengths_hb is None:
            eep_lengths_hb = [150, 100, 80, 100, 80, 80, 80, 100, 100]

        self.eep_list = eep_list
        self.nticks = eep_lengths
        self.eep_list_hb = eep_list_hb
        self.nticks_hb = eep_lengths_hb


class DefineEeps(object):
    '''
    Define the stages if not simply using Sandro's defaults.
    * denotes stages defined here, otherwise, taken from Sandro's defaults.
    PMS_BEG    Beginning of Pre Main Sequence
    PMS_MIN    Minimum of Pre Main Sequence
    PMS_END    End of Pre-Main  Sequence
    1 MS_BEG   Starting of the central H-burning phase
    2 MS_TMIN* First Minimum in Teff for high-mass
               Xc=0.30 for low-mass stars (BaSTi)
               For very low mass stars that do not reach Xc=0.30: AGE~=13.7 Gyr
    3 MS_TO*   Maximum in Teff along the Main Sequence - TURN OFF POINT (BaSTi)
               For very low mass stars that do not reach the MSTO in 100 Gyr,
               this is AGE~=100 Gyr
    4 SG_MAXL* Maximum in logL for high-mass or Xc=0.0 for low-mass stars
                  (BaSTi)
    5 RG_MINL* Minimum in logL for high-mass or Base of the RGB for
                  low-mass stars (BaSTi, but found with parametric
                  interpolation)
    6 RG_BMP1  The maximum luminosity during the RGB Bump
    7 RG_BMP2  The minimum luminosity during the RGB Bump
    8 RG_TIP   Tip of the RGB defined in 3 ways:
                  1) if the last track model still has a YCEN val > 0.1
                     the TRGB is either the min te or the last model, which
                     ever comes first. (low masses)
                  2) if there is no YCEN left in the core at the last track
                     model, TRGB is the min TE where YCEN > 1-Z-0.1.
                  3) if there is still XCEN in the core (very low mass), TRGB
                     is the final track model point.
    9 HE_BEG*  Start quiescent central He-burning phase
    10 YCEN_0.550* Central abundance of He equal to 0.55
    11 YCEN_0.500* Central abundance of He equal to 0.50
    12 YCEN_0.400* Central abundance of He equal to 0.40
    13 YCEN_0.200* Central abundance of He equal to 0.20
    14 YCEN_0.100* Central abundance of He equal to 0.10
    15 YCEN_0.000* Central abundance of He equal to 0.00
    16 TPAGB Starting of the central C-burning phase

    HB Tracks:
    AGB_LY1*     Helium (shell) fusion first overpowers hydrogen (shell) fusion
    AGB_LY2*     Hydrogen wins again (before TPAGB).
       **For low-mass HB (<0.485) the hydrogen fusion is VERY low (no atm!),
            and never surpasses helium, this is still a to be done!!

    Not yet implemented, no TPAGB tracks decided:
    x When the energy produced by the CNO cycle is larger than that
    provided by the He burning during the AGB (Lcno > L3alpha)
    x The maximum luminosity before the first Thermal Pulse
    x The AGB termination
    '''
    def __init__(self):
        pass

    def define_eep_stages(self, track, hb=False, plot_dir=None,
                          diag_plot=True, debug=False):

        if hb is True:
            print('\n\n       HB Current Mass: %.3f' % track.mass)
            self.add_hb_beg(track)
            self.add_cen_eeps(track, hb=hb)
            self.add_agb_eeps(track, diag_plot=diag_plot, plot_dir=plot_dir)
            return

        print('\n\n          Current Mass: %.3f' % track.mass)
        # set all to zero
        [self.add_eep(track, cp, 0) for cp in self.ptcri.please_define]
        nsandro_pts = len(np.nonzero(track.sptcri != 0)[0])
        
        self.add_ms_eeps(track)

        
        ims_to = track.iptcri[self.ptcri.get_ptcri_name('MS_TO', sandro=False)]
        if ims_to == 0:
            print('MS_TO and MS_TMIN found by AGE limits')
            self.add_eep_with_age(track, 'MS_TMIN', 13.7e9)
            self.add_eep_with_age(track, 'MS_TO', 100e9)

            # no found MSTO and track goes up to POINT_B (very low mass)
            if track.mass > 1.25:
                print('major problem with finding MS and ptcri file!')
        else:
            # go on defining eeps.
            imin_l = self.add_min_l_eep(track)
            if imin_l == -1:
                imax_l = self.add_max_l_eep(track, eep2='RG_BMP1')
            else:
                imax_l = self.add_max_l_eep(track)
            # high mass, low z, have hard to find base of rg, but easier to find
            # sg_maxl. This flips the order of finding. Also an issue is if
            # the L min after the MS_TO is much easier to find than the RG Base.
            # hence make sure the indexs are at least 10 apart.
            if imax_l == -1 or imax_l == 0:
                print('max_l near ms_to M=%.4f delta ind: %i' %
                             (track.mass, (imax_l - ims_to)))
                imax_l = self.add_max_l_eep(track, eep2='RG_BMP1')
                self.add_min_l_eep(track, eep1='SG_MAXL')

            ihe_beg = 0
            self.add_eep(track, 'HE_BEG', ihe_beg)  # initilizing
            self.add_cen_eeps(track)
            ycen1 = track.iptcri[self.ptcri.get_ptcri_name('YCEN_0.550',
                                                           sandro=False)]
            if ycen1 != 0:
                self.add_quiesscent_he_eep(track, 'YCEN_0.550')
                ihe_beg = track.iptcri[self.ptcri.get_ptcri_name('HE_BEG',
                                                                 sandro=False)]
            if ihe_beg == 0 or nsandro_pts <= 10:
                # No He EEPs
                # should now make sure all other eeps are 0.
                [self.add_eep(track, cp, 0) for cp in self.ptcri.please_define[5:]]

        if False in (np.diff(track.iptcri[np.nonzero(track.iptcri > 0)]) > 0):
            print('EEPs are not monotonically increasing. M=%.3f' % track.mass)
            print(pprint.pprint(track.iptcri))

    def remove_dupes(self, inds1, inds2, inds3, just_two=False):
        '''
        Duplicates will make the interpolation fail, and thus delay graduation
        dates. Here is where they die.
        '''
        inds = np.arange(len(inds1))
        if not just_two:
            mask, = np.nonzero(((np.diff(inds1) == 0) &
                                (np.diff(inds2) == 0) &
                                (np.diff(inds3) == 0)))
        else:
            mask, = np.nonzero(((np.diff(inds1) == 0) & (np.diff(inds2) == 0)))

        non_dupes = [i for i in inds if i not in mask]

        return non_dupes

    def add_quiesscent_he_eep(self, track, ycen1):
        '''
        He fusion starts after the RGB, but where? It was tempting to simply
        choose the min L on the HeB, but that could come after 1/2 the YCEN
        was burned for massive stars. I decided to find a place after the RGB
        where there was a bump in YCEN, a little spurt before it started
        burning He at a more consistent rate.

        The above method was not stable for all Z. I've instead moved to
        where there is a min after the TRGB in LY, that is it dips as the
        star contracts, and then ramps up.
        '''
        inds = inds_between_ptcris(track, 'RG_TIP', ycen1, sandro=False)
        eep_name = 'HE_BEG'

        if len(inds) == 0:
            print('no start HEB!!!! M=%.4f Z=%.4f' %
                         (track.mass, track.Z))
            self.add_eep(track, eep_name, 0)
            return 0

        min = np.argmin(track.data.LY[inds])
        # Sometimes there is a huge peak in LY before the min, find it...
        npts = inds[-1] - inds[0] + 1
        subset = npts / 3
        max = np.argmax(track.data.LY[inds[:subset]])
        # Peak isn't as important as the ratio between the start and end
        rat = track.data.LY[inds[max]]/track.data.LY[inds[0]]
        # If the min is at the point next to the TRGB, or the ratio is huge,
        # get the min after the peak.
        if min == 0 or rat > 10:
            amin = np.argmin(track.data.LY[inds[max + 1:]])
            min = max + 1 + amin
        he_beg = inds[min]
        self.add_eep(track, eep_name, he_beg)
        return he_beg

    def add_cen_eeps(self, track, hb=False, tol=0.01):
        '''
        Add YCEN_%.3f eeps, if YCEN=fraction not found to tol, will add 0 as
        the iptrcri, equivalent to not found.
        '''

        if hb is False:
            # not Horizontal branch, start before rgb tip
            irgbmp2 = self.ptcri.get_ptcri_name('RG_BMP2', sandro=False)
            istart = track.iptcri[irgbmp2]
            please_define = self.ptcri.please_define
        else:
            # Horizontal branch tracks, start at 0.
            istart = 0
            please_define = self.ptcri.please_define_hb

        inds = np.arange(istart, len(track.data.YCEN))

        # use undefined central values instead of given list.
        cens = [i for i in please_define if i.startswith('YCEN')]
        # e.g., YCEN_0.500
        cens = [float(cen.split('_')[-1]) for cen in cens]
        icens = []
        for cen in cens:
            ind, dif = utils.closest_match(cen, track.data.YCEN[inds])
            icen = inds[ind]
            # some tolerance for a good match.
            if dif > tol:
                icen = 0
            self.add_eep(track, 'YCEN_%.3f' % cen, icen, hb=hb)
            # for monotonic increase, even if there is another flare up in
            # He burning, this limits the matching indices to begin at this
            # new eep index.
            inds = np.arange(icen, len(track.data.YCEN))
            icens.append(icen)
        return icens

    def add_hb_beg(self, track):
        # this is just the first line of the track with age > 0.2 yr.
        # it could be placed in the load_track method. However, because
        # it's unphysical, I'm keeping it here to be clear.
        ainds, = np.nonzero(track.data['AGE'] > 0.2)
        hb_beg = ainds[0]
        eep_name = 'HB_BEG'
        self.add_eep(track, eep_name, hb_beg, hb=True)
        return hb_beg

    def add_agb_eeps(self, track, diag_plot=False, plot_dir=None):
        '''
        This is for HB tracks... not sure if it will work for tpagb.

        These EEPS will be when 1) helium (shell) fusion first overpowers
        hydrogen (shell) fusion and 2) when hydrogen wins again (before TPAGB).
        For low-mass HB (<0.485) the hydrogen fusion is VERY low (no atm!),
        and never surpasses helium, this is still a to be done!!
        '''
        if track.mass <= 0.480:
            print('HB AGB EEPS might not work for HPHB')

        ly = track.data.LY
        lx = track.data.LX
        norm_age = track.data.AGE/track.data.AGE[-1]

        ex_inds, = np.nonzero(track.data.YCEN == 0.00)

        diff_L = np.abs(ly[ex_inds] - lx[ex_inds])
        peak_dict = utils.find_peaks(diff_L)

        # there are probably thermal pulses in the track, taking the first 6
        # mins to try and avoid them. Yeah, I checked by hand, 6 usually works.
        mins = peak_dict['minima_locations'][:6]

        # the two deepest mins are the ly == lx match
        min_inds = np.asarray(mins)[np.argsort(diff_L[mins])[0:2]]
        (agb_ly1, agb_ly2) = np.sort(ex_inds[min_inds])

        # most of the time, the above works, a couple times the points
        # are instead in the thermal pulses.

        # if agb_ly1 is in a thermal pulse (and should be much younger)
        # take away some mins...
        i = 4
        if norm_age[ex_inds[mins[0]]] < 0.89 and norm_age[agb_ly1] > 0.98:
            while norm_age[agb_ly1] > 0.98:
                mins = mins[:i]
                min_inds = np.asarray(mins)[np.argsort(diff_L[mins])[0:2]]
                (agb_ly1, agb_ly2) = np.sort(ex_inds[min_inds])
                i -= 1

        # if the agb_ly2 is in a thermal pulse take away some mins...
        if norm_age[agb_ly2] > 0.999:
            while norm_age[agb_ly2] > 0.999:
                mins = mins[:i]
                min_inds = np.asarray(mins)[np.argsort(diff_L[mins])[0:2]]
                (agb_ly1, agb_ly2) = np.sort(ex_inds[min_inds])
                i -= 1

        self.add_eep(track, 'AGB_LY1', agb_ly1, hb=True)
        self.add_eep(track, 'AGB_LY2', agb_ly2, hb=True)

        if diag_plot is True:
            agb_ly1c = 'red'
            agb_ly2c = 'blue'
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
            ax1.plot(norm_age, ly, label='He', color='purple')
            ax1.plot(norm_age, lx, label='H', color='orange')
            ax1.plot(norm_age[ex_inds], diff_L, label='abs diff',
                     color='black')
            ax1.plot(norm_age[ex_inds[mins]], diff_L[mins], 'o',
                     color='black')
            ax1.plot(norm_age[agb_ly1], track.data.LX[agb_ly1], 'o',
                     color=agb_ly1c)
            ax1.plot(norm_age[agb_ly2], track.data.LX[agb_ly2], 'o',
                     color=agb_ly2c)

            ax1.set_title(track.mass)
            ax1.legend(loc=0)
            ax1.set_ylim(-0.05, 1.)
            ax1.set_xlim(norm_age[ex_inds[0]], 1)
            ax1.set_ylabel('Luminosity fraction from []')
            ax1.set_xlabel('Age/Total Age')

            ax2.plot(track.data.LOG_TE, track.data.LOG_L, color='green')
            ax2.plot(track.data.LOG_TE[agb_ly1], track.data.LOG_L[agb_ly1],
                     'o', color=agb_ly1c)
            ax2.plot(track.data.LOG_TE[agb_ly2], track.data.LOG_L[agb_ly2],
                     'o', color=agb_ly2c)
            ax2.set_xlim(ax2.get_xlim()[::-1])
            ax2.set_xlabel('$\log\ T_{eff}$')
            ax2.set_ylabel('$\log\ L$')

            figname = 'diag_agb_HB_eep_M%s.png' % track.mass
            if plot_dir is not None:
                figname = os.path.join(plot_dir, figname)
            plt.savefig(figname)
            # helpful in ipython:
            #if i == 4:
            #    plt.close()
            plt.close()
            print('wrote %s' % figname)
        return agb_ly1, agb_ly2

    def add_ms_eeps(self, track):
        '''
        Adds  MS_TMIN and MS_TO.

        MS_TMIN found in this function is either:
        a) np.argmin(LOG_TE) between Sandro's MS_BEG and POINT_C
        b) XCEN=0.3 if low mass (low mass value is hard coded and global)
        c) the log te min on the MS found by the second derivative of
            d^2 log_te / d model^2 where model is just the inds between
            MS_BEG and POINT_C (that is, model as in model number from the
            tracks)
        d) zero

        MS_TO is either:
        a) the max log te on the MS found by
           i) by the peak log Te; or
           ii) subtracting off a linear fit in log_te vs model number
            (see c. above)
        b) zero

        if there is an error, either MS_TO or MS_TMIN will -1

        more information: 
        '''
        def second_derivative(xdata, inds):
            '''
            The second derivative of d^2 xdata / d inds^2 
            
            why inds for interpolation, not log l?
            if not using something like model number instead of log l,
            the tmin will get hidden by data with t < tmin but different
            log l. This is only a problem for very low Z.
            If I find the arg min of teff to be very close to MS_BEG it
            probably means the MS_BEG is at a lower Teff than Tmin.
            '''
            tckp, _ = splprep([inds, xdata], s=0, k=3, nest=-1)
            arb_arr = np.arange(0, 1, 1e-2)
            xnew, ynew = splev(arb_arr, tckp)
            # second derivative, bitches.
            ddxnew, ddynew = splev(arb_arr, tckp, der=2)
            ddyddx = ddynew/ddxnew
            # not just argmin, but must be actual min...
            aind = [a for a in np.argsort(ddyddx) if ddyddx[a-1] > 0][0]
            tmin_ind, _ = utils.closest_match2d(aind, inds, xdata, xnew, ynew)
            return inds[tmin_ind]

        def delta_te_eeps(track, ms_to, ms_tmin):
            xdata = track.data.LOG_L
            return np.abs(np.diff((xdata[ms_to], xdata[ms_tmin])))
        
        inds = inds_between_ptcris(track, 'MS_BEG', 'POINT_C', sandro=True)

        if len(inds) == 0:
            ms_tmin = 0
        else:
            xdata = track.data.LOG_TE[inds]
            ms_tmin = inds[np.argmin(xdata)]
            delta_te = delta_te_eeps(track, ms_tmin, inds[0])

            if track.mass <= low_mass:
                # BaSTi uses XCEN == 0.3, could put this as a keyword
                xcen = 0.3
                dte = np.abs(track.data.XCEN[inds] - xcen)
                ms_tmin = inds[np.argmin(dte)]
                print('MS_TMIN found by XCEN=%.1f M=%.4f' % (xcen, track.mass))
            elif delta_te < .1:  # value to use interp instead
                # find the te min by interpolation.
                ms_tmin = second_derivative(xdata, inds)
                print('MS_TMIN found by interp M=%.4f' % track.mass)
            else:
                print('found MS_TMIN the easy way, np.argmin(LOG_TE)')

        self.add_eep(track, 'MS_TMIN', ms_tmin)

        if ms_tmin == 0:
            ms_to = 0
        else:
            inds = inds_between_ptcris(track, 'MS_TMIN', 'RG_BMP1',
                                       sandro=False)
            if len(inds) == 0:
                # No RGB_BMP1?
                inds = np.arange(ms_tmin, len(track.data.LOG_TE - 1))

            ms_to = inds[np.argmax(track.data.LOG_TE[inds])]

            delta_te = delta_te_eeps(track, ms_to, ms_tmin)
            
            if track.mass <= low_mass or delta_te < 0.01:
                # first try with parametric interpolation
                pf_kw = {'max': True, 'sandro': False,
                         'more_than_one': 'max of max',
                         'parametric_interp': False}

                ms_to = self.peak_finder(track, 'LOG_TE', 'MS_TMIN', 'RG_BMP1',
                                         **pf_kw)

                delta_te = delta_te_eeps(track, ms_to, ms_tmin)
                # if the points are too cluse try with less linear fit
                if ms_to == -1 or delta_te < 0.01:
                    pf_kw['less_linear_fit'] = True
                    ms_to = self.peak_finder(track, 'LOG_TE', 'MS_TMIN',
                                             'RG_BMP1', **pf_kw)
            if ms_to == -1:
                # do the same as for tmin... take second deriviative
                xdata = track.data.LOG_L[inds]
                ms_to = second_derivative(xdata, inds)

            if ms_to == -1:
                # tried four ways!?!!
                print('No MS_TO? M=%.4f Z=%.4f' % (track.mass, track.Z))
                ms_to = 0
        self.add_eep(track, 'MS_TO', ms_to)
        return ms_tmin, ms_to

    def add_min_l_eep(self, track, eep1='MS_TO'):
        '''
        The MIN L before the RGB for high mass or the base of the
        RGB for low mass.

        MIN L of RGB is found parametrically in log te, log l, and log age.
        If no min found that way, will use the base of RGB.
        this will be the deflection point in HRD space between
        MS_TO and RG_BMP1.
        '''

        pf_kw = {'sandro': False, 'more_than_one': 'last'}
        min_l = self.peak_finder(track, 'LOG_L', eep1, 'RG_BMP1', **pf_kw)

        if min_l == -1 or track.mass < low_mass:
            pf_kw = {'parametric_interp': False,
                     'more_than_one': 'min of min',
                     'sandro': False}

            pf_kw['less_linear_fit'] = True
            print('RG_MINL found with less linear fit %s %.3f %.4f' %
                         (eep1, track.mass, track.Z))
            min_l = self.peak_finder(track, 'LOG_L', eep1, 'RG_BMP1', **pf_kw)

        if min_l == -1:
            pf_kw['less_linear_fit'] = False
            print('RG_MINL without parametric')
            min_l = self.peak_finder(track, 'LOG_L', eep1, 'RG_BMP1', **pf_kw)

        if np.round(track.data.XCEN[min_l], 4) > 0 and min_l > 0:
            print('XCEN at RG_MINL should be zero if low mass (M=%.4f). %.4f' %
                         (track.mass, track.data.XCEN[min_l]))
        self.add_eep(track, 'RG_MINL', min_l)
        return min_l

    def add_max_l_eep(self, track, eep2='RG_MINL'):
        '''
        Adds SG_MAXL between MS_TO and RG_MINL.
        '''
        if track.Z < 0.001 and track.mass > 8:
            extreme = 'first'
            #if track.Z == 0.0005:
            #    if track.mass == 11.:
            #        print('%.4f doing it sg-maxl by hand bitches.' % track.mass)
            #        self.add_eep(track, 'SG_MAXL', 1540)
            #        return 1540
            #    elif track.mass == 12.:
            #        print('%.4f doing it sg-maxl by hand bitches.' % track.mass)
            #        self.add_eep(track, 'SG_MAXL', 1535)
            #        return 1515
        else:
            extreme = 'max of max'
        pf_kw = {'max': True, 'sandro': False, 'more_than_one': extreme,
                 'parametric_interp': False, 'less_linear_fit': True}

        if eep2 != 'RG_MINL':
            pf_kw['mess_err'] = 'still a problem with max_l %.3f' % track.mass

        max_l = self.peak_finder(track, 'LOG_L', 'MS_TO', eep2, **pf_kw)

        if max_l == -1:
            pf_kw['less_linear_fit'] = bool(np.abs(pf_kw['less_linear_fit']-1))
            print('SG_MAXL flipping less_linear_fit to %s (was %i with eep2: %s)' % (pf_kw['less_linear_fit'], max_l, eep2))
            max_l = self.peak_finder(track, 'LOG_L', 'MS_TO', eep2, **pf_kw)
            print('%i %.4f' % (max_l, track.mass))

        msto = track.iptcri[self.ptcri.get_ptcri_name('MS_TO', sandro=False)]
        if max_l == msto:
            print('SG_MAXL is at MS_TO!')
            print('XCEN at MS_TO (%i): %.3f' % (msto, track.data.XCEN[msto]))
            max_l = -1

        self.add_eep(track, 'SG_MAXL', max_l)
        return max_l

    def convective_core_test(self, track):
        '''
        only uses sandro's defs, so doesn't need load_critical_points
        initialized.
        '''
        ycols = ['QSCHW', 'QH1', 'QH2']
        age = track.data.AGE
        lage = np.log10(age)

        morigs = [t for t in self.ptcri.data_dict['M%.3f' % self.mass]
                  if t > 0 and t < len(track.data.LOG_L)]
        iorigs = [np.nonzero(track.data.MODE == m)[0][0] for m in morigs]
        try:
            inds = inds_between_ptcris(track, 'MS_BEG', 'POINT_B',
                                                  sandro=True)
        except IndexError:
            inds, = np.nonzero(age > 0.2)

        try:
            lage[inds]
        except IndexError:
            inds = np.arange(len(lage))
        plt.figure()
        ax = plt.axes()
        for ycol in ycols:
            ax.plot(lage[inds], track.data[ycol][inds], lw=2, label=ycol)
            ax.scatter(lage[iorigs[4]], track.data[ycol][iorigs[4]], s=100,
                       marker='o', color='red')
            if len(iorigs) >= 7:
                ax.scatter(lage[iorigs[7]], track.data[ycol][iorigs[7]], s=100,
                           marker='o', color='blue')
                xmax = lage[iorigs[7]] + .4
            else:
                xmax = ax.get_xlim()[1]

        xmin = lage[iorigs[4]] - .4
        ax.set_xlim(xmin, xmax)
        ax.set_title(self.mass)
        ax.legend(loc=0)

    def add_eep_with_age(self, track, eep_name, age):
        iage = np.argmin(np.abs(track.data.AGE - age))
        age_diff = np.min(np.abs(track.data.AGE - age))
        print('%g' % (age_diff/age), track.mass, eep_name)
        self.add_eep(track, eep_name, iage)


    def add_eep(self, track, eep_name, ind, hb=False):
        '''
        Will add or replace the index of Track.data to track.iptcri
        '''
        if hb is True:
            key_dict = self.ptcri.key_dict_hb
        else:
            key_dict = self.ptcri.key_dict

        track.iptcri[key_dict[eep_name]] = ind
        #if ind != 0:
        #    print('%s, %i' % (eep_name, ind))

    def peak_finder(self, track, col, eep1, eep2, max=False, diff_err=None,
                    sandro=True, more_than_one='max of max', mess_err=None,
                    ind_tol=3, dif_tol=0.01, less_linear_fit=False,
                    parametric_interp=True):
        '''
        finds some peaks! Usually interpolates and calls a basic diff finder,
        though some higher order derivs of the interpolation are sometimes used.
        '''
        # slice the array
        inds = inds_between_ptcris(track, eep1, eep2, sandro=sandro)
        # burn in
        #inds = inds[5:]

        if len(inds) < ind_tol:
            # sometimes there are not enough inds to interpolate
            print('Peak finder %s-%s M%.3f: less than %i points = %i. Skipping.'
                         % (eep1, eep2, track.mass, ind_tol, len(inds)))
            return 0

        if parametric_interp is True:
            # use age, so logl(age), logte(age) for parametric interpolation
            tckp, step_size, non_dupes = self.interpolate_te_l_age(track, inds)
            arb_arr = np.arange(0, 1, step_size)
            agenew, xnew, ynew = splev(arb_arr, tckp)
            dxnew, dynew, dagenew = splev(arb_arr, tckp, der=1)
            intp_col = ynew
            dydx = dxnew / dynew
            if col == 'LOG_TE':
                intp_col = xnew
                dydx = dynew / dxnew
        else:
            # interpolate logl, logte.
            xdata = track.data['LOG_TE'][inds]
            ydata = track.data['LOG_L'][inds]

            non_dupes = self.remove_dupes(xdata, ydata, 'lixo', just_two=True)
            xdata = xdata[non_dupes]
            ydata = ydata[non_dupes]
            k = 3
            if len(non_dupes) <= k:
                k = 1
                print('only %i indices to fit... %s-%s' %
                               (len(non_dupes), eep1, eep2))
                print('new spline_level %i' % k)

            tckp, u = splprep([xdata, ydata], s=0, k=k, nest=-1)

            ave_data_step = np.round(np.mean(np.abs(np.diff(xdata))), 4)
            min_step = 1e-2
            step_size = np.max([ave_data_step, min_step])
            xnew, ynew = splev(np.arange(0, 1, 1e-2), tckp)
            #dxnew, dynew = splev(np.arange(0, 1, step_size), tckp, der=1)
        if col == 'LOG_L':
            intp_col = ynew
            nintp_col = xnew
        else:
            intp_col = xnew
            nintp_col = ynew
            #dydx = dynew / dxnew

        # find the peaks!
        if less_linear_fit is True:
            if track.mass < 5.:
                axnew = xnew
                p = np.polyfit(nintp_col, intp_col, 1)
                m = p[0]
                b = p[1]
            else:
                axnew = np.arange(nintp_col.size)
                m = (intp_col[-1] - intp_col[0]) / (axnew[-1] - axnew[0])
                b = axnew[0]
            # subtract linear fit, find peaks
            peak_dict = utils.find_peaks(intp_col - (m * axnew + b))
        else:
            peak_dict = utils.find_peaks(intp_col)

        if max is True:
            if peak_dict['maxima_number'] > 0:
                imax = peak_dict['maxima_locations']
                if more_than_one == 'max of max':
                    almost_ind = imax[np.argmax(intp_col[imax])]
                elif more_than_one == 'last':
                    almost_ind = imax[-1]
                elif more_than_one == 'first':
                    almost_ind = imax[0]
            else:
                # no maxs found.
                if mess_err is not None:
                    print(mess_err)
                return -1

        else:
            if peak_dict['minima_number'] > 0:
                imin = peak_dict['minima_locations']
                if more_than_one == 'min of min':
                    if parametric_interp is True:
                        almost_ind = np.argmax(dydx)
                    else:
                        almost_ind = imin[np.argmin(intp_col[imin])]
                elif more_than_one == 'last':
                    almost_ind = imin[-1]
                elif more_than_one == 'first':
                    almost_ind = imin[0]

            else:
                # no mins found
                if mess_err is not None:
                    print(mess_err)
                return -1

        if parametric_interp is True:
            # closest point in interpolation to data
            ind, dif = utils.closest_match2d(almost_ind,
                                                  track.data[col][inds][non_dupes],
                                                  np.log10(track.data.AGE[inds][non_dupes]),
                                                  intp_col, agenew)
        else:
            # closest point in interpolation to data
            ind, dif = utils.closest_match2d(almost_ind, xdata, ydata,
                                                  xnew, ynew)

        if ind == -1:
            # didn't find anything.
            return ind

        if dif > dif_tol:
            # closest match was too far away from orig.
            if diff_err is not None:
                print(diff_err)
            else:
                print('bad match %s-%s M=%.3f' % (eep1, eep2,
                                                         track.mass))
            return -1
        return inds[non_dupes][ind]

    def load_critical_points(self, track, filename=None, ptcri=None,
                             eep_obj=None, hb=False, plot_dir=None,
                             diag_plot=True, debug=False):
        '''
        calls define_eep_stages
        iptcri is the critical point index rel to track.data
        mptcri is the model number of the critical point

        there is a major confusion here... ptcri is a super class or should be
        track specific? right now it's being copied everywhere. stupido.
        '''
        assert filename is not None or ptcri is not None, \
            'Must supply either a ptcri file or object'

        if ptcri is None:
            ptcri = critical_point(filename, eep_obj=eep_obj)
        self.ptcri = ptcri

        assert ptcri.Z == track.Z, \
            'Zs do not match between track and ptcri file'

        assert ptcri.Y == track.Y, \
            'Ys do not match between track and ptcri file'

        if hasattr(self.ptcri, 'eep'):
            # already loaded eep
            eep_obj = self.ptcri.eep

        if hb is True and len(ptcri.please_define_hb) > 0:
            # Initialize iptcri for HB
            track.iptcri = np.zeros(len(eep_obj.eep_list_hb), dtype=int)
            self.define_eep_stages(track, hb=hb, plot_dir=plot_dir,
                                   diag_plot=diag_plot, debug=debug)
        else:
            # Sandro's definitions. (I don't use his HB EEPs)
            mptcri = ptcri.data_dict['M%.3f' % track.mass]
            track.sptcri = \
                np.concatenate([np.nonzero(track.data.MODE == m)[0]
                                for m in mptcri])

            if len(ptcri.please_define) > 0:
                # Initialize iptcri
                track.iptcri = np.zeros(len(eep_obj.eep_list), dtype=int)

                # Get the values that we won't be replacing.
                pinds = np.array([i for i, a in enumerate(self.ptcri.eep.eep_list)
                                  if a in self.ptcri.sandro_eeps])

                sinds = np.array([i for i, a in enumerate(self.ptcri.sandro_eeps)
                                  if a in self.ptcri.eep.eep_list])
                track.iptcri[pinds] = mptcri[sinds] - 2

                # but if the track did not actually make it to that EEP, no -2!
                track.iptcri[track.iptcri < 0] = 0

                # and if sandro cut the track before it reached this point,
                # no index error!
                track.iptcri[track.iptcri > len(track.data.MODE)] = 0

                # define the eeps
                self.define_eep_stages(track, hb=hb, plot_dir=plot_dir,
                                       diag_plot=diag_plot, debug=debug)

            else:
                # copy sandros dict.
                track.iptcri = ptcri.data_dict['M%.3f' % track.mass]
        return track

    def interpolate_vs_age(self, track, col, inds, k=3, nest=-1, s=0.):
        '''
        a caller for scipy.optimize.splprep. Will also rid the array
        of duplicate values. Returns tckp, an input to splev.
        '''
        non_dupes = self.remove_dupes(track.data[col][inds],
                                      track.data.AGE[inds], 'lixo',
                                      just_two=True)

        if len(non_dupes) <= k:
            k = len(non_dupes) - 1
            print('only %i indices to fit... %s-%s' % (len(non_dupes)))
            print('new spline_level %i' % k)

        tckp, u = splprep([track.data[col][inds][non_dupes],
                           np.log10(track.data.AGE[inds][non_dupes])],
                          s=s, k=k, nest=nest)
        return tckp

    def interpolate_te_l_age(self, track, inds, k=3, nest=-1, s=0.,
                             min_step=1e-4):
        '''
        a caller for scipy.optimize.splprep. Will also rid the array
        of duplicate values. Returns tckp, an input to splev.
        '''
        non_dupes = self.remove_dupes(track.data.LOG_TE[inds],
                                      track.data.LOG_L[inds],
                                      track.data.AGE[inds])

        if len(non_dupes) <= k:
            k = 1
            print('only %i indices to fit...' % (len(non_dupes)))
            print('new spline_level %i' % k)

        tckp, u = splprep([np.log10(track.data.AGE[inds][non_dupes]),
                           track.data.LOG_TE[inds][non_dupes],
                           track.data.LOG_L[inds][non_dupes]],
                           s=s, k=k, nest=nest)

        xdata = track.data.LOG_TE[inds][non_dupes]
        ave_data_step = np.round(np.mean(np.abs(np.diff(xdata))), 4)
        step_size = np.max([ave_data_step, min_step])

        return tckp, step_size, non_dupes
