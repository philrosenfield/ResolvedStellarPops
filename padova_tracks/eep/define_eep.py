from __future__ import print_function
import numpy as np
import pprint
from ResolvedStellarPops import utils
import os
import matplotlib.pylab as plt
from scipy.interpolate import splev, splprep
from critical_point import critical_point
import copy
#from ..mass_config import low_mass
# from low mass and below XCEN = 0.3 for MS_TMIN
low_mass = 1.25
# from high mass and above find MS_BEG in this code
high_mass = 20.

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
    def __init__(self, input_obj=None):
        pass

    def define_eep_stages(self, track, hb=False, plot_dir=None,
                          diag_plot=True, debug=False):

        if hb is True:
            self.hb_eeps(track, diag_plot=diag_plot, plot_dir=plot_dir)
            return

        print('M=%.3f' % track.mass)
        import pdb; pdb.set_trace()
        self.check_pms_beg(track)
        # set all to zero
        [self.add_eep(track, cp, 0) for cp in self.ptcri.please_define]
        nsandro_pts = len(np.nonzero(track.sptcri != 0)[0])
        
        # if this is high mass or if Sandro's MS_BEG is wrong:
        if track.mass >= high_mass or track.data.XCEN[track.sptcri[4]] < .6:
            #print('M=%.4f is high mass' % track.mass)
            return self.add_high_mass_eeps(track)
            
        self.add_ms_eeps(track)

        ims_to = track.iptcri[self.ptcri.get_ptcri_name('MS_TO', sandro=False)]
        if ims_to == 0:
            if track.mass > low_mass:
                print('MS_TO and MS_TMIN found by AGE limits %.4f!' % track.mass)
            self.add_eep_with_age(track, 'MS_TMIN', (13.7e9/2.))
            self.add_eep_with_age(track, 'MS_TO', 13.7e9)

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

    def hb_eeps(self, track, diag_plot=True, plot_dir=None):
        '''
        define the HB EEPs.
        '''
        print('M=%.3f, HB' % track.mass)
        self.add_hb_beg(track)
        self.add_cen_eeps(track, hb=True)
        self.add_agb_eeps(track, diag_plot=diag_plot, plot_dir=plot_dir)
        return
        
    def check_pms_beg(self, track):
        print(track.mass, track.data.AGE[track.sptcri[0]], track.data.AGE[np.nonzero(np.round(track.data['AGE'], 1) > 0.2)[0][0]])
        if track.data.AGE[track.sptcri[0]] <= 0.2:
            self.add_eep(track, 'PMS_BEG',
                         np.nonzero(np.round(track.data['AGE'], 1) > 0.2)[0][0])
        return

    def add_high_mass_eeps(self, track):
        pf_kw = {'get_max': False, 'sandro': True,
                 'more_than_one': 'min of min',
                 'parametric_interp': True}

        ms_beg = self.peak_finder(track, 'LOG_L', 'PMS_END', 'POINT_B',
                                  **pf_kw)
        self.add_eep(track, 'MS_BEG', ms_beg)
        
        ms_to = np.nonzero(track.data.XCEN == 0)[0][0]
        self.add_eep(track, 'MS_TO', ms_to)
            
        inds = np.arange(ms_beg, ms_to)
        peak_dict = utils.find_peaks(track.data.LOG_L[inds])
        if peak_dict['minima_number'] > 2. or peak_dict['maxima_number'] > 2.:
            print('MS_BEG-MS_TO minima_number', peak_dict['minima_number'])
            print('MS_BEG-MS_TO maxima_number', peak_dict['maxima_number'])
            track = self.strip_instablities(track, inds)
        xdata = track.data.LOG_TE[inds]
        ms_tmin = inds[np.argmin(xdata)]
        self.add_eep(track, 'MS_TMIN', ms_tmin)
        
        # there is a switch for high mass tracks in the add_cen_eeps and
        # add_quiesscent_he_eep functions. If the mass is higher than
        # high_mass the functions use MS_TO as the initial EEP for peak_finder.
        cens = self.add_cen_eeps(track, istart=ms_to)
        heb_beg  = self.add_quiesscent_he_eep(track, cens[0], start=ms_to)
        
        # there are instabilities in massive tracks that are on the verge or
        # returning to the hot side (Teff>10,000) of the HRD before C_BUR.
        # The following is designed to cut the tracks before the instability.
        # If the star is M>55. and the last model doesn't reach Teff = 10**4,
        # The track is cut at the max LOG_L after the MS_TO, otherwise, that
        # value is the TPAGB (not actually TP-AGB, but end of the track).
        fin = len(track.data.LOG_L) - 1
        inds = np.arange(ms_to, fin)
        peak_dict = utils.find_peaks(track.data.LOG_L[inds])
        max_frac = peak_dict['maxima_number'] / float(len(inds))
        min_frac = peak_dict['minima_number'] / float(len(inds))
        if min_frac > 0.15 or max_frac > 0.15:
            print('MS_TO+ %i inds. Fracs: max %.2f, min %.2f' %
                  (len(inds), max_frac, min_frac))
            print('MS_TO+ minima_number', peak_dict['minima_number'])
            print('MS_TO+ maxima_number', peak_dict['maxima_number'])
            track = self.strip_instablities(track, inds)

        #tpagb = inds[np.argmax(track.data.LOG_L[inds])]
        #if track.mass >= 55. and track.data.LOG_TE[fin] < 4.:
        #    fin = tpagb
        self.add_eep(track, 'TPAGB', fin)
        if cens[-1] > fin:
            print('final point on track is cut before final ycen %.4f' % track.mass)
        # between ms_to and heb_beg need eeps that are meaningless at high mass:
        _, sg_maxl, rg_minl, rg_bmp1, rg_bmp2, rg_tip, _  = \
            map(int, np.round(np.linspace(ms_to, heb_beg, 7)))
        
        self.add_eep(track, 'SG_MAXL', sg_maxl)
        self.add_eep(track, 'RG_MINL', rg_minl)
        self.add_eep(track, 'RG_BMP1', rg_bmp1)
        self.add_eep(track, 'RG_BMP2', rg_bmp2)
        self.add_eep(track, 'RG_TIP', rg_tip)

        return np.concatenate(([ms_beg, ms_tmin, ms_to, heb_beg], cens, [fin]))
    
    def add_quiesscent_he_eep(self, track, ycen1, start='RG_TIP'):
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
        if type(ycen1) != str:
            inds = np.arange(start, ycen1)
        else:
            inds = self.ptcri.inds_between_ptcris(track, start, ycen1, sandro=False)
        eep_name = 'HE_BEG'

        if len(inds) == 0:
            print('No HE_BEG M=%.4f Z=%.4f' % (track.mass, track.Z))
            self.add_eep(track, eep_name, 0)
            return 0

        he_min = np.argmin(track.data.LY[inds])
        # Sometimes there is a huge peak in LY before the min, find it...
        npts = inds[-1] - inds[0] + 1
        subset = npts / 3
        he_max = np.argmax(track.data.LY[inds[:subset]])
        # Peak isn't as important as the ratio between the start and end
        rat = track.data.LY[inds[he_max]] / track.data.LY[inds[0]]
        # If the min is at the point next to the RG_TIP, or the ratio is huge,
        # get the min after the peak.
        if he_min == 0 or rat > 10:
            amin = np.argmin(track.data.LY[inds[he_max + 1:]])
            he_min = he_max + 1 + amin
        he_beg = inds[he_min]
        self.add_eep(track, eep_name, he_beg)
        return he_beg

    def add_cen_eeps(self, track, hb=False, tol=0.01, istart=None):
        '''
        Add YCEN_%.3f eeps, if YCEN=fraction not found to tol, will add 0 as
        the iptrcri, equivalent to not found.
        '''
        if istart is None:
            start = 'RG_BMP2'
            # if high mass start after ms_to
            pstart = self.ptcri.get_ptcri_name(start, sandro=False)
            istart = track.iptcri[pstart]

        if hb is False:
            # not Horizontal branch, start before rgb tip
            please_define = self.ptcri.please_define
        else:
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
        # it is an not physically meaningful eep, I'm keeping it here.
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
            #print('wrote %s' % figname)
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
            try:
                aind = [a for a in np.argsort(ddyddx) if ddyddx[a-1] > 0][0]
            except IndexError:
                return -1
            tmin_ind, _ = utils.closest_match2d(aind, inds, xdata, xnew, ynew)
            return inds[tmin_ind]

        def delta_te_eeps(track, ms_to, ms_tmin):
            xdata = track.data.LOG_L
            return np.abs(np.diff((xdata[ms_to], xdata[ms_tmin])))

        inds = self.ptcri.inds_between_ptcris(track, 'MS_BEG', 'POINT_C', sandro=True)

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
                #print('MS_TMIN found by XCEN=%.1f M=%.4f' % (xcen, track.mass))
            elif delta_te < .1:  # value to use interp instead
                # find the te min by interpolation.
                ms_tmin = second_derivative(xdata, inds)
                #print('MS_TMIN found by interp M=%.4f' % track.mass)
            #else:
                #print('found MS_TMIN the easy way, np.argmin(LOG_TE)')

        self.add_eep(track, 'MS_TMIN', ms_tmin)

        if ms_tmin == 0:
            ms_to = 0
        else:
            inds = self.ptcri.inds_between_ptcris(track, 'MS_TMIN', 'RG_BMP1',
                                       sandro=False)
            if len(inds) == 0:
                # No RGB_BMP1?
                inds = np.arange(ms_tmin, len(track.data.LOG_TE - 1))

            ms_to = inds[np.argmax(track.data.LOG_TE[inds])]

            delta_te = delta_te_eeps(track, ms_to, ms_tmin)
            
            if track.mass <= low_mass or delta_te < 0.01:
                # first try with parametric interpolation
                pf_kw = {'get_max': True, 'sandro': False,
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
                    ms_to = inds[np.nonzero(track.data.XCEN[inds] == 0)[0][0]]
                    print('MS_TO %.4f set to XCEN==0 as last resort' % track.mass)
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
            #print('RG_MINL found with less linear fit %s %.3f %.4f' %
            #             (eep1, track.mass, track.Z))
            min_l = self.peak_finder(track, 'LOG_L', eep1, 'RG_BMP1', **pf_kw)

        if min_l == -1:
            pf_kw['less_linear_fit'] = False
            #print('RG_MINL without parametric')
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
        else:
            extreme = 'max of max'
        pf_kw = {'get_max': True, 'sandro': False, 'more_than_one': extreme,
                 'parametric_interp': False, 'less_linear_fit': True}

        if eep2 != 'RG_MINL':
            pf_kw['mess_err'] = 'still a problem with SG_MAXL %.3f' % track.mass

        max_l = self.peak_finder(track, 'LOG_L', 'MS_TO', eep2, **pf_kw)

        if max_l == -1:
            pf_kw['less_linear_fit'] = bool(np.abs(pf_kw['less_linear_fit']-1))
            #print('SG_MAXL flipping less_linear_fit to %s (was %i with eep2: %s)' % (pf_kw['less_linear_fit'], max_l, eep2))
            max_l = self.peak_finder(track, 'LOG_L', 'MS_TO', eep2, **pf_kw)
            #print('%i %.4f' % (max_l, track.mass))

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
            inds = self.ptcri.inds_between_ptcris(track, 'MS_BEG', 'POINT_B',
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

    def add_eep_with_age(self, track, eep_name, age, tol=0.1):
        iage = np.argmin(np.abs(track.data.AGE - age))
        age_diff = np.min(np.abs(track.data.AGE - age))
        if (age_diff/age) > tol:
            print('possible bad age match for eep.')
            print('frac diff, mass, eep_name, age, final track age')
            print('%g' % (age_diff/age), track.mass, eep_name, '%g' % age, '%g' % track.data.AGE[-1])
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

    def peak_finder(self, track, col, eep1, eep2, get_max=False, diff_err=None,
                    sandro=True, more_than_one='max of max', mess_err=None,
                    ind_tol=3, dif_tol=0.01, less_linear_fit=False,
                    parametric_interp=True):
        '''
        finds some peaks! Usually interpolates and calls a basic diff finder,
        though some higher order derivs of the interpolation are sometimes used.
        '''
        # slice the array
        # either take inds or the EEP names
        if type(eep1) != str:
            inds = np.arange(eep1, eep2)
        else:
            inds = self.ptcri.inds_between_ptcris(track, eep1, eep2, sandro=sandro)

        if len(inds) < ind_tol:
            # sometimes there are not enough inds to interpolate
            print('Peak finder %s-%s M%.3f: less than %i points = %i. Skipping.'
                         % (eep1, eep2, track.mass, ind_tol, len(inds)))
            return 0

        # use age, so logl(age), logte(age) for parametric interpolation
        tckp, step_size, non_dupes = self._interpolate(track, inds,
                                                       parametric=parametric_interp)
        arb_arr = np.arange(0, 1, step_size)
        if parametric_interp is True:
            agenew, xnew, ynew = splev(arb_arr, tckp)
            dxnew, dynew, dagenew = splev(arb_arr, tckp, der=1)
            intp_col = ynew
            dydx = dxnew / dynew
            if col == 'LOG_TE':
                intp_col = xnew
                dydx = dynew / dxnew
        else:
            # interpolate logl, logte.
            xnew, ynew = splev(arb_arr, tckp)

        if col == 'LOG_L':
            intp_col = ynew
            nintp_col = xnew
        else:
            intp_col = xnew
            nintp_col = ynew

        # find the peaks!
        if less_linear_fit is True:
            if track.mass < 5.:
                axnew = xnew
                # calculate slope using polyfit
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

        if get_max is True:
            mstr = 'max'
        else:
            mstr = 'min'
            
        if peak_dict['%sima_number' % mstr] > 0:
            iextr = peak_dict['%sima_locations' % mstr]
            if more_than_one == 'max of max':
                almost_ind = iextr[np.argmax(intp_col[iextr])]
            elif more_than_one == 'min of min':
                if parametric_interp is True:
                    almost_ind = np.argmax(dydx)
                else:
                    almost_ind = iextr[np.argmin(intp_col[iextr])]
            elif more_than_one == 'last':
                almost_ind = iextr[-1]
            elif more_than_one == 'first':
                almost_ind = iextr[0]
        else:
            # no maxs found.
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
            ind, dif = utils.closest_match2d(almost_ind,
                                             track.data.LOG_TE[inds][non_dupes],
                                             track.data.LOG_L[inds][non_dupes],
                                             xnew, ynew)

        if ind == -1:
            # didn't find anything.
            return ind

        if dif > dif_tol:
            # closest match was too far away from orig.
            if diff_err is not None:
                print(diff_err)
            else:
                print('bad match %s-%s M=%.3f' % (eep1, eep2, track.mass))
            return -1
        return inds[non_dupes][ind]

    def load_critical_points(self, track, ptcri, hb=False, plot_dir=None,
                             diag_plot=True, debug=False):
        '''
        calls define_eep_stages
        iptcri is the critical point index rel to track.data
        mptcri is the model number of the critical point

        there is a major confusion here... ptcri is a super class or should be
        track specific? right now it's being copied everywhere. stupido.
        '''
        assert ptcri is not None, \
            'Must supply either a ptcri file or object'

        if type(ptcri) is str:
            ptcri = critical_point(ptcri)
        self.ptcri = ptcri
        eep_obj = self.ptcri.eep

        assert ptcri.Z == track.Z, \
            'Zs do not match between track and ptcri file %f != %f' % (ptcri.Z,
                                                                       track.Z)

        assert np.round(ptcri.Y, 2) == np.round(track.Y, 2),  \
            'Ys do not match between track and ptcri file %f != %f' % (ptcri.Y,
                                                                       track.Y)

        if hb is True and len(ptcri.please_define_hb) > 0:
            # Initialize iptcri for HB
            track.iptcri = np.zeros(len(eep_obj.eep_list_hb), dtype=int)
            self.define_eep_stages(track, hb=hb, plot_dir=plot_dir,
                                   diag_plot=diag_plot, debug=debug)
        else:
            # Sandro's definitions. (I don't use his HB EEPs)
            try:
                mptcri = ptcri.data_dict['M%.3f' % track.mass]
            except KeyError:
                print('M=%.4f not found in %s' % (track.mass,
                                                  os.path.join(self.ptcri.base,
                                                               self.ptcri.name)))
                track.flag = 'no ptcri mass'
                return track
            track.sptcri = \
                np.concatenate([np.nonzero(track.data.MODE == m)[0]
                                for m in mptcri])

            if len(ptcri.please_define) > 0:
                # Initialize iptcri
                track.iptcri = np.zeros(len(eep_obj.eep_list), dtype=int)

                # Get the values that we won't be replacing.
                pinds = np.array([i for i, a in enumerate(eep_obj.eep_list)
                                  if a in self.ptcri.sandro_eeps])

                sinds = np.array([i for i, a in enumerate(self.ptcri.sandro_eeps)
                                  if a in eep_obj.eep_list])
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

    def _interpolate(self, track, inds, k=3, nest=-1, s=0.,
                             min_step=1e-4, parametric=True,
                             linear=False):
        '''
        a caller for scipy.optimize.splprep. Will also rid the array
        of duplicate values. Returns tckp, an input to splev.
        if parametric_interp is True use AGE with LOG_TE and LOG_L
           if linear is also False use log10 Age
        
        note the dimensionality of tckp will change if using parametric_interp
        '''
        non_dupes = self.remove_dupes(track.data.LOG_TE[inds],
                                      track.data.LOG_L[inds],
                                      track.data.AGE[inds])
        
        if len(non_dupes) <= k:
            k = 1
            #print('only %i indices to fit...' % (len(non_dupes)))
            #print('new spline_level %i' % k)
        
        xdata = track.data.LOG_TE[inds][non_dupes]
        ydata = track.data.LOG_L[inds][non_dupes]
        
        if parametric is True:
            arr = [track.data.AGE[inds][non_dupes], xdata, ydata]
            if linear is False:
                arr[0] = np.log10(arr[0])
        else:
            arr = [xdata, ydata]
        
        tckp, u = splprep(arr, s=s, k=k, nest=nest)

        ave_data_step = np.round(np.mean(np.abs(np.diff(xdata))), 4)
        step_size = np.max([ave_data_step, min_step])

        return tckp, step_size, non_dupes

    def remove_dupes(self, inds1, inds2, inds3, just_two=False):
        '''
        Remove duplicates so as to not brake the interpolator.
        '''
        def unique_seq(seq):
            '''
            A fast uniquify of a sequence
            from http://www.peterbe.com/plog/uniqifiers-benchmark
            # submitted by Dave Kirby
            '''
            seen = set()
            return [i for i, x in enumerate(seq)
                    if x not in seen and not seen.add(x)]

        un_ind1 = unique_seq(inds1)
        un_ind2 = unique_seq(inds2)
        if not just_two:
            un_ind3 = unique_seq(inds3)
            non_dupes = list(set(un_ind1) & set(un_ind2) & set(un_ind3))
        else:
            non_dupes = list(set(un_ind1) & set(un_ind2))

        return non_dupes
    
    def strip_instablities(self, track, inds):
        return track
        peak_dict = utils.find_peaks(track.data.LOG_L[inds])
        extrema = np.sort(np.concatenate([peak_dict['maxima_locations'],
                                          peak_dict['minima_locations']]))
        # divide into jumps that are at least 50 models apart
        jumps, = np.nonzero(np.diff(extrema) > 50)
        if len(jumps) == 0:
            print('no istabilities found')
            return track
        if not hasattr(track, 'data_orig'):
            track.data_orig = copy.copy(track.data)
        # add the final point
        jumps = np.append(jumps, len(extrema) - 1)
        # instability is defined by having more than 20 extrema
        jumps = jumps[np.diff(np.append(jumps, extrema[-1])) > 20]
        # np.diff is off by one in the way I am about to use it
        # burn back some model points to blend better with the old curve...
        starts = jumps[:-1] + 1
        ends = jumps[1:]
        # the inds to smooth.
        for i in range(len(jumps)-1):
            moffset = (inds[extrema[ends[i]]] - inds[extrema[starts[i]]]) / 10
            poffset = moffset
            if i == len(jumps)-2:
                poffset = 0
            finds = np.arange(inds[extrema[starts[i]]] - moffset,
                              inds[extrema[ends[i]]] + poffset)
            tckp, step_size, non_dupes = self._interpolate(track, finds, s=0.2,
                                                                   linear=True)
            arb_arr = np.arange(0, 1, step_size)
            if len(arb_arr) > len(finds):
                arb_arr = np.linspace(0, 1, len(finds))
            else:
                print(len(finds), len(xnew))
            agenew, xnew, ynew = splev(arb_arr, tckp)
            #ax.plot(xnew, ynew)
            track.data['LOG_L'][finds] = ynew
            track.data['LOG_TE'][finds] = xnew
            track.data['AGE'][finds] = agenew
            print('LOG_L, LOG_TE, AGE interpolated from inds %i:%i' %
                  (finds[0], finds[-1]))
            track.header.append('LOG_L, LOG_TE, AGE interpolated from MODE %i:%i \n' % (track.data.MODE[finds][0], track.data.MODE[finds][-1]))
            
            self.check_strip_instablities(track)
        return track

    def check_strip_instablities(self, track):
        fig, (axs) = plt.subplots(ncols=2, figsize=(16, 10))
        for ax, xcol in zip(axs, ['AGE', 'LOG_TE']):
            for data, alpha in zip([track.data_orig, track.data], [0.3, 1]):
                ax.plot(data[xcol], data.LOG_L, color='k', alpha=alpha)
                ax.plot(data[xcol], data.LOG_L, ',', color='k')

            ax.set_xlabel('$%s$' % xcol.replace('_', r'\! '), fontsize=20)
            ax.set_ylabel('$LOG\! L$', fontsize=20)
        plt.show()
        

