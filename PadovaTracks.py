from copy import deepcopy
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splprep
import fileIO
import math_utils
import graphics.GraphicsUtils as rspg
import pprint
import logging
logger = logging.getLogger()

def add_comments_to_header(tracks_base, prefix, search_term):
    '''
    insert a # at every line before MODE. genfromtxt will skip the
    footer and doesn't need #... but that's a place to improve this
    function.
    '''
    tracks = os.path.join(tracks_base, prefix)
    track_names = fileIO.get_files(tracks, search_term)

    for name in track_names:
        with open(name, 'r') as t:
            lines = t.readlines()
        try:
            imode, = [i for (i, l) in enumerate(lines)
                      if l.strip().startswith('MODE ')]
        except ValueError:
            print '\n %s \n' % name

        lines[:imode + 1] = ['# ' + l for l in lines[:imode + 1]]

        oname = '%s.dat' % name

        with open(oname, 'w') as t:
            t.writelines(lines)


def quick_color_em(tracks_base, prefix, photsys='UVbright',
                   search_term='*F7_*PMS', fromHR2mags=None):
    '''
    This goes quickly through each directory and adds a [search_term].dat file
    that has # in the header and a [search_term].dat.[photsys]] file that is
    the output of Leo's fromHR2mags.

    sometimes leo's code has bad line endings or skips lines, i donno. so
    when reading in as TrackSet, you'll get loads of warnings...

    ex:
    tracks_base = '/Users/phil/research/parsec2match/S12_set/CAF09_S12D_NS/'
    prefix = 'S12D_NS_Z0.0001_Y0.249'
    quick_color_em(tracks_base, prefix)
    '''

    def color_tracks(tracks_base, prefix, cmd):
        tracks = os.path.join(tracks_base, prefix)
        track_names = fileIO.get_files(tracks, search_term)

        for name in track_names:
            z = float(name.split('Z')[1].split('_Y')[0])
            os.system(cmd % (name, z))
            print cmd % (name, z)
    if fromHR2mags is None:
        fromHR2mags = '/home/rosenfield/research/padova_apps/fromHR2mags/fromHR2mags'
    cmd = '%s %s ' % (fromHR2mags, photsys)
    # this is set for .PMS and .PMS.HB tracks
    cmd += '%s 5 6 2 %.4f'
    add_comments_to_header(tracks_base, prefix, search_term)
    search_term += '.dat'
    color_tracks(tracks_base, prefix, cmd)


class Track(object):
    '''
    Padova stellar evolutoni track object. 
    '''
    def __init__(self, filename, ptcri=None, min_lage=0.1, cut_long=False):
        (self.base, self.name) = os.path.split(filename)
        self.load_track(filename, min_lage=min_lage, cut_long=cut_long)
        self.filename_info()
        self.mass = self.data.MASS[0]
        if self.mass >= 12:
            # for high mass tracks, the mass starts much larger than it is
            # for (age<0.2). The mass only currect at the beginning of the MS.
            # Rather than forcing a ptcri load, we read the mass from the title.
            self.mass = float(self.name.split('_M')[1].split('.PMS')[0])
        self.ptcri = ptcri
        test = np.diff(self.data.AGE) >= 0
        if False in test:
            print 'Track has age decreasing!!', self.mass
            bads, = np.nonzero(np.diff(self.data.AGE) < 0)
            print self.data.MODE[bads]
            
    def calc_Mbol(self):
        '''
        Uses Z_sun = 4.77
        '''
        Mbol = 4.77 - 2.5 * self.data.LOG_L
        self.Mbol = Mbol
        return Mbol

    def calc_logg(self):
        '''
        cgs constant is -10.616
        '''
        logg = -10.616 + np.log10(self.mass) + 4.0 * self.data.LOG_TE - \
            self.data.LOG_L
        self.logg = logg
        return logg

    def calc_core_mu(self):
        '''
        Uses X, Y, C, and O.
        '''
        xi = np.array(['XCEN', 'YCEN', 'XC_cen', 'XO_cen'])
        ai = np.array([1., 4., 12., 16.])
        # fully ionized
        qi = ai/2.
        self.MUc = 1. / (np.sum((self.data[xi[i]] / ai[i]) *
                         (1 + qi[i]) for i in range(len(xi))))

    def filename_info(self):
        '''
        I wish I knew regex...
        '''
        (pref, __, smass) = self.name.split('.PMS')[0].split('_')
        #self.__setattr__[]
        #get that into attrs: 'Z0.0002Y0.4OUTA1.74M2.30'
        Z, Ymore = self.name.split('Z')[1].split('Y')
        for i, y in enumerate(Ymore):
            if y == '.':
                continue
            try:
                float(y)
            except:
                break
        self.Z = float(Z)
        self.Y = float(Ymore[:i])
        return

    def cut_long_track(self):
        '''
        cuts tracks at the first thermal pulse or c burning. Taken from
        Bressan's save_isoc_set This is not used.
        '''
        icburn = []
        ycen = self.data.YCEN
        if ycen[-1] == 0:
            lc = self.data.LC
            lneutr = self.data.LNEUTR
            icburn, = np.nonzero((ycen == 0) & ((lc > 2) | (lc > lneutr)))
        if len(icburn) > 0:
            # actually C-burning
            logger.info('Cutting at C-burning')
            itpagb = min(icburn)
        else:
            # beginning thin shell
            logger.info('Cutting at thin shell burning')
            ishell, = np.nonzero((ycen == 0) &
                                 (self.data.QCAROX > self.data.QHEL * 3. / 4.))
            if len(ishell) > 0:
                itpagb = np.min(ishell)
            else:
                itpagb = len(self.data) - 1
                ishell, = np.nonzero((self.data.LY > 1) &
                                     (self.data.QCAROX > 0.1))
                if len(ishell) > 0:
                    itpagb = np.min(ishell)

        itpagb = itpagb + 1
        self.data = self.data[np.arange(itpagb)]

    def load_track(self, filename, min_lage=0.1, cut_long=True):
        '''
        reads PMS file into a record array. Stores header as string self.header
        '''
        with open(filename, 'r') as f:
            lines = f.readlines()

        begin_track, = [i for i, l in enumerate(lines) if 'BEGIN TRACK' in l]
        self.header = lines[:begin_track]
        col_keys = lines[begin_track + 1].replace('#', '').strip().split()
        begin_track_skip = 2
        
        # Hack to read tracks that have been "colored"
        if 'information' in lines[begin_track + 2]:
            col_keys = self.add_to_col_keys(col_keys, lines[begin_track + 2])
            begin_track_skip += 1

        try:
            data = np.genfromtxt(filename,
                                 skiprows=begin_track + begin_track_skip,
                                 names=col_keys)
        except ValueError:
            # comp time is often a footer.
            data = np.genfromtxt(filename,
                                 skiprows=begin_track + begin_track_skip,
                                 names=col_keys, skip_footer=2,
                                 invalid_raise=False)

        # cut non-physical part of the model
        # NOTE it should be >= but sometimes Sandro's PMS_BEG
        #      will be one model number too soon for that.
        #ainds, = np.nonzero(data['AGE'] > min_lage)
        #data = data[ainds]
        self.data = data.view(np.recarray)
        self.col_keys = col_keys
        if cut_long:
            self.cut_long_track()
        return

    def add_to_col_keys(self, col_keys, additional_col_line):
        '''
        If fromHR2mags was run, A new line "Additional information Added:..."
        is added, this adds these column keys to the list.
        '''
        new_cols = additional_col_line.split(':')[1].strip().split()
        col_keys = list(np.concatenate((col_keys, new_cols)))
        return col_keys

    def format_track_file(self):
        '''
        add # comment to header save new file with .dat extension.
        Useful to prepare for fromHR2mags.
        '''
        self.header = ['# %s' % h for h in self.header]

        name_dat = '%s.dat' % self.name
        filename = os.path.join(self.base, name_dat)
        with open(filename, 'w') as f:
            f.writelines(self.header)
            f.write('# %s \n' % ' '.join(self.col_keys))
            np.savetxt(f, self.data, dtype=self.data.dtype)
        return filename


class DefineEeps(object):
    '''
    Define the stages if not simply using Sandro's defaults.
    * denotes stages defined here.

    1 MS_BEG   Starting of the central H-burning phase
    2 MS_TMIN* First Minimum in Teff for high-mass or Xc=0.30 for low-mass
                  stars (BaSTi)
    3 MS_TO*   Maximum in Teff along the Main Sequence - TURN OFF POINT (BaSTi)
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
    AGB_LY1     Helium (shell) fusion first overpowers hydrogen (shell) fusion
    AGB_LY2     Hydrogen wins again (before TPAGB).
       **For low-mass HB (<0.485) the hydrogen fusion is VERY low (no atm!),
            and never surpasses helium, this is still a to be done!!
    
    Not yet implemented, no TPAGB tracks decided:
    x When the energy produced by the CNO cycle is larger than that
    provided by the He burning during the AGB (Lcno > L3alpha)
    x The maximum luminosity before the first Thermal Pulse
    x The AGB termination
    '''
    def __init__(self):
        self.setup_eep_info()

    def setup_eep_info(self):
        '''
        I'd like to make something that'd record what was sent to peak_finder
        for each mass and Z...
        '''
        self.eep_info = {}

    def validate_eeps(self, hb=False):
        '''
        If there isn't an eep that I've made a function for, this should die.
        The order also can be important. This messes up the whole idea of
        being able to set the ycen values from an input file...
        '''
        if hb is True:
            default_list = ['HB_BEG', 'YCEN_0.550', 'YCEN_0.500', 'YCEN_0.400',
                            'YCEN_0.200', 'YCEN_0.100', 'YCEN_0.005',
                            'YCEN_0.000', 'AGB_LY1', 'AGB_LY2']

            eep_list = self.ptcri.please_define_hb
        else:
            default_list = ['MS_TMIN', 'MS_TO', 'SG_MAXL', 'RG_MINL', 'HE_BEG',
                            'YCEN_0.550', 'YCEN_0.500', 'YCEN_0.400',
                            'YCEN_0.200', 'YCEN_0.100', 'YCEN_0.005',
                            'YCEN_0.000']
            eep_list = self.ptcri.please_define

        assert default_list == eep_list, \
            ('Can not define all EEPs. Please check lists', eep_list)

    def define_eep_stages(self, track, hb=False, plot_dir=None,
                          diag_plot=True, debug=False):
        self.validate_eeps(hb=hb)

        if hb is True:
            logger.info('\n\n       HB Current Mass: %.3f' % track.mass)
            self.add_hb_beg(track)
            self.add_cen_eeps(track, hb=hb)
            self.add_agb_eeps(track, diag_plot=diag_plot, plot_dir=plot_dir)
            return

        logger.info('\n\n          Current Mass: %.3f' % track.mass)

        self.add_ms_eeps(track)

        nsandro_pts = len(np.nonzero(track.ptcri.sptcri != 0)[0])
        # ahem, recall...
        # low mass will at least go up to point_b.
        # 'PMS_BEG': 0, 'PMS_MIN': 1, 'PMS_END': 2, 'NEAR_ZAM': 3, 'MS_BEG': 4,
        # 'POINT_B': 5, 'POINT_C': 6, 'RG_BASE': 7, 'RG_BMP1': 8, 'RG_BMP2': 9,
        # 'RG_TIP': 10, 'Loop_A': 11, 'Loop_B': 12, 'Loop_C': 13, 'TPAGB': 14,
        ims_to = self.ptcri.iptcri[self.ptcri.get_ptcri_name('MS_TO',
                                                             sandro=False)]

        if ims_to == 0 or nsandro_pts <= 5:
            # should now make sure all other eeps are 0.
            [self.add_eep(cp, 0) for cp in self.ptcri.please_define[2:]]
        else:
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
                logger.debug('max_l near ms_to M=%.4f delta ind: %i' %
                             (track.mass, (imax_l - ims_to)))
                imax_l = self.add_max_l_eep(track, eep2='RG_BMP1')
                self.add_min_l_eep(track, eep1='SG_MAXL')
            # RG_TIP is from Sandro
            ihe_beg = 0
            self.add_eep('HE_BEG', ihe_beg)  # initilizing
            self.add_cen_eeps(track)
            ycen1 = self.ptcri.iptcri[self.ptcri.get_ptcri_name('YCEN_0.550',
                                                                sandro=False)]
            if ycen1 != 0:
                self.add_quiesscent_he_eep(track, 'YCEN_0.550')
                ihe_beg = self.ptcri.iptcri[self.ptcri.get_ptcri_name('HE_BEG',
                                                                      sandro=False)]
            if ihe_beg == 0 or nsandro_pts <= 10:
                # should now make sure all other eeps are 0.
                [self.add_eep(cp, 0) for cp in self.ptcri.please_define[5:]]

        if False in (np.diff(self.ptcri.iptcri[np.nonzero(self.ptcri.iptcri > 0)]) > 0):
            logger.error('EEPs are not monotonically increasing. M=%.3f' %
                         track.mass)
            logger.error(pprint.pprint(self.ptcri.iptcri))
        if debug is True:
            pass
  
    def remove_dupes(self, x, y, z, just_two=False):
        '''
        Duplicates will make the interpolation fail, and thus delay graduation
        dates. Here is where they die.
        '''

        inds = np.arange(len(x))
        if not just_two:
            mask, = np.nonzero(((np.diff(x) == 0) &
                                (np.diff(y) == 0) &
                                (np.diff(z) == 0)))
        else:
            mask, = np.nonzero(((np.diff(x) == 0) & (np.diff(y) == 0)))

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
        inds = inds_between_ptcris(track.ptcri, 'RG_TIP', ycen1, sandro=False)
        eep_name = 'HE_BEG'

        if len(inds) == 0:
            logger.error('no start HEB!!!! M=%.4f Z=%.4f' %
                         (track.mass, track.Z))
            self.add_eep(eep_name, 0)
            return 0

        min = np.argmin(track.data.LY[inds])
        # Sometimes there is a huge peak in LY before the min, find it...
        npts = inds[-1] - inds[0] + 1
        subset = npts/3
        max = np.argmax(track.data.LY[inds[:subset]])
        # Peak isn't as important as the ratio between the start and end
        rat = track.data.LY[inds[max]]/track.data.LY[inds[0]]
        # If the min is at the point next to the TRGB, or the ratio is huge,
        # get the min after the peak.
        if min == 0 or rat > 10:
            amin = np.argmin(track.data.LY[inds[max+1:]])
            min = max + 1 + amin
        he_beg = inds[min]
        self.add_eep(eep_name, he_beg)
        return he_beg

    def add_cen_eeps(self, track, hb=False):
        '''
        Add YCEN_[fraction] eeps, if YCEN=fraction found to 0.01, will add 0 as
        the iptrcri. (0.01 is hard coded)
        list of YCEN_[fraction] can be supplied by cens= otherwise taken from
        self.ptcri.please_define
        '''

        if hb is False:
            irgbmp2 = self.ptcri.get_ptcri_name('RG_BMP2', sandro=False)
            istart = self.ptcri.iptcri[irgbmp2]
            please_define = self.ptcri.please_define
        else:
            istart = 0
            please_define = self.ptcri.please_define_hb

        inds = np.arange(istart, len(track.data.YCEN))

        # use undefined central values instead of given list.
        cens = [i for i in please_define if i.startswith('YCEN')]
        # e.g., YCEN_0.50
        cens = [float(cen.split('_')[-1]) for cen in cens]
        icens = []
        for cen in cens:
            ind, dif = math_utils.closest_match(cen, track.data.YCEN[inds])
            icen = inds[ind]
            # some tolerance for a good match.
            if dif > 0.01:
                icen = 0
            self.add_eep('YCEN_%.3f' % cen, icen, hb=hb)
            # for monotonic increase, even if there is another flare up in
            # He burning, this limits the matching indices to begin at this
            # new eep index.
            inds = np.arange(icen, len(track.data.YCEN))
            icens.append(icen)
        return icens

    def add_hb_beg(self, track):
        # this is just the first line of the track with age > 0.2 yr.
        # it could be snipped in the load_track method because it's
        # unphysical but to be clear, I'm keeping it here too.
        ainds, = np.nonzero(track.data['AGE'] > 0.2)
        hb_beg = ainds[0]
        eep_name = 'HB_BEG'
        self.add_eep(eep_name, hb_beg, hb=True)
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
            logger.warning('HB AGB EEPS might not work for HPHB')

        ly = track.data.LY
        lx = track.data.LX
        norm_age = track.data.AGE/track.data.AGE[-1]

        ex_inds, = np.nonzero(track.data.YCEN == 0.00)

        diff_L = np.abs(ly[ex_inds] - lx[ex_inds])
        peak_dict = math_utils.find_peaks(diff_L)

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

        self.add_eep('AGB_LY1', agb_ly1, hb=True)
        self.add_eep('AGB_LY2', agb_ly2, hb=True)

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
            logger.info('wrote %s' % figname)
        return agb_ly1, agb_ly2

    def add_ms_eeps(self, track):
        '''
        Adds  MS_TMIN and MS_TO.

        MS_TMIN is either:
        a) previously found by hand
        b) XCEN=0.3 if low mass (low mass is set by hand)
        c) the log te min on the MS found by the second derivative of
            d^2 log_te / d model^2 where model is just the inds between
            MS_BEG and POINT_C (that is, model as in model number from the
            tracks)
        d) zero

        MS_TO is either:
        a) the max log te on the MS found either simply by the peak, or by
            subtracting off a linear fit in log_te vs model number 
            (see c. above)
        b) zero

        if there is an error, of course either can also be -1.

        more information: why inds for interpolation, not log l?
            if not using something like model number instead of log l,
            the tmin will get hidden by data with t < tmin but different
            log l. This is only a problem for very low Z.
            If I find the arg min of teff to be very close to MS_BEG it
            probably means the MS_BEG is at a lower Teff than Tmin.
        '''
        inds = inds_between_ptcris(track.ptcri, 'MS_BEG', 'POINT_C', sandro=True)
        if len(inds) == 0:
            ms_tmin = 0
        else:
            xdata = track.data.LOG_TE[inds]
            tmin_ind = np.argmin(xdata)
            ms_tmin = inds[tmin_ind]
            delta_te = np.abs(np.diff((track.data.LOG_L[ms_tmin],
                                       track.data.LOG_L[inds[0]])))
            if track.mass < 1.2:
                # use XCEN == 0.3
                dte = np.abs(track.data.XCEN[inds] - 0.3)
                tmin_ind = np.argmin(dte)
                # not used... but a quality control:
                dif = dte[tmin_ind]
            elif delta_te < .1:
                # find the te min by interpolation.
                mode = inds
                tckp, u = splprep([mode, xdata], s=0, k=3, nest=-1)
                arb_arr = np.arange(0, 1, 1e-2)
                xnew, ynew = splev(arb_arr, tckp)
                # second derivative, bitches.
                ddxnew, ddynew = splev(arb_arr, tckp, der=2)
                ddyddx = ddynew/ddxnew
                # not just argmin, but must be actual min...
                aind = [a for a in np.argsort(ddyddx) if ddyddx[a-1] > 0][0]
                tmin_ind, dif = math_utils.closest_match2d(aind, mode,
                                                           xdata, xnew, ynew)
                logger.debug('found tmin by interp M=%.4f' % track.mass)
            ms_tmin = inds[tmin_ind]
        self.add_eep('MS_TMIN', ms_tmin)

        if ms_tmin == 0:
            ms_to = 0
        else:
            inds = inds_between_ptcris(track.ptcri, 'MS_TMIN', 'RG_BMP1',
                                                  sandro=False)
            if len(inds) == 0:
                # No RGB_BM1?
                inds = np.arange(ms_tmin, len(track.data.LOG_TE-1))

            ms_to = inds[np.argmax(track.data.LOG_TE[inds])]

            delta_te_ms_to = np.abs(np.diff((track.data.LOG_L[ms_to],
                                             track.data.LOG_L[ms_tmin])))
            if track.mass < 1.2 or delta_te_ms_to < 0.01:
                pf_kw = {'max': True, 'sandro': False,
                         'more_than_one': 'max of max',
                         'parametric_interp': False}

                ms_to = self.peak_finder(track, 'LOG_TE', 'MS_TMIN', 'RG_BMP1',
                                         **pf_kw)

                delta_te_ms_to = np.abs(np.diff((track.data.LOG_L[ms_to],
                                        track.data.LOG_L[ms_tmin])))
                if ms_to == -1 or delta_te_ms_to < 0.01:
                    pf_kw['less_linear_fit'] = True
                    pf_kw['mess_err'] = 'still a problem with ms_to %.3f' % track.mass
                    ms_to = self.peak_finder(track, 'LOG_TE', 'MS_TMIN',
                                             'RG_BMP1', **pf_kw)
            if ms_to == -1:
                logger.error('no ms to? M=%.4f Z=%.4f' % (track.mass, track.Z))
                ms_to = 0
        self.add_eep('MS_TO', ms_to)
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

        if min_l == -1 or track.mass < 1.2:
            pf_kw = {'parametric_interp': False,
                     'more_than_one': 'min of min',
                     'sandro': False}

            pf_kw['less_linear_fit'] = True
            logger.debug('min_l with less lin %s %.3f %.4f' %
                         (eep1, track.mass, track.Z))
            min_l = self.peak_finder(track, 'LOG_L', eep1, 'RG_BMP1', **pf_kw)

        if min_l == -1:
            pf_kw['less_linear_fit'] = False
            logger.debug('try min_l without parametric')
            min_l = self.peak_finder(track, 'LOG_L', eep1, 'RG_BMP1', **pf_kw)

        if np.round(track.data.XCEN[min_l], 4) > 0:
            logger.error('XCEN at RG_MINL should be zero if low mass (M=%.4f). %.4f' %
                         (track.mass, track.data.XCEN[min_l]))
        self.add_eep('RG_MINL', min_l)
        return min_l

    def add_max_l_eep(self, track, eep2='RG_MINL'):
        '''
        Adds SG_MAXL between MS_TO and RG_MINL.
        '''
        if track.Z < 0.001 and track.mass > 8:
            extreme = 'first'
            if track.Z == 0.0005:
                if track.mass == 11.:
                    logger.error('%.4f doing it sg-maxl by hand bitches.' % track.mass)
                    self.add_eep('SG_MAXL', 1540)
                    return 1540
                elif track.mass == 12.:
                    logger.error('%.4f doing it sg-maxl by hand bitches.' % track.mass)
                    self.add_eep('SG_MAXL', 1535)
                    return 1515
        else:
            extreme = 'max of max'
        pf_kw = {'max': True, 'sandro': False, 'more_than_one': extreme,
                 'parametric_interp': False, 'less_linear_fit': True}

        if eep2 != 'RG_MINL':
            pf_kw['mess_err'] = 'still a problem with max_l %.3f' % track.mass

        max_l = self.peak_finder(track, 'LOG_L', 'MS_TO', eep2, **pf_kw)

        if max_l == -1:
            pf_kw['less_linear_fit'] = bool(np.abs(pf_kw['less_linear_fit']-1))
            logger.debug('max l flipping less_linear_fit to %s (was %i with eep2: %s)' % (pf_kw['less_linear_fit'], max_l, eep2))
            max_l = self.peak_finder(track, 'LOG_L', 'MS_TO', eep2, **pf_kw)
            logger.debug('%i %.4f' % (max_l, track.mass))

        msto = self.ptcri.iptcri[self.ptcri.get_ptcri_name('MS_TO', sandro=False)]
        if max_l == msto:
            logger.error('SG_MAXL is at MS_TO!')
            logger.error('XCEN at MS_TO (%i): %.3f' % (msto, track.data.XCEN[msto]))
            max_l = -1

        self.add_eep('SG_MAXL', max_l)
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
            inds = inds_between_ptcris(track.ptcri, 'MS_BEG', 'POINT_B',
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

    def add_eep(self, eep_name, ind, hb=False):
        '''
        Will add or replace the index of Track.data to self.ptcri.iptcri
        '''
        if hb is True:
            key_dict = self.ptcri.key_dict_hb
        else:
            key_dict = self.ptcri.key_dict

        self.ptcri.iptcri[key_dict[eep_name]] = ind
        logger.debug('%s, %i' % (eep_name, ind))

    def peak_finder(self, track, col, eep1, eep2, max=False, diff_err=None,
                    sandro=True, more_than_one='max of max', mess_err=None,
                    ind_tol=3, dif_tol=0.01, less_linear_fit=False,
                    parametric_interp=True):
        '''
        finds some peaks! Usually interpolates and calls a basic diff finder,
        though some higher order derivs of the interpolation are sometimes used.
        '''
        # slice the array
        inds = inds_between_ptcris(track.ptcri, eep1, eep2, sandro=sandro)
        # burn in
        #inds = inds[5:]

        if len(inds) < ind_tol:
            # sometimes there are not enough inds to interpolate
            logger.error('Peak finder %s-%s M%.3f: less than %i points = %i. Skipping.'
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
                logger.warning('only %i indices to fit... %s-%s' %
                               (len(non_dupes), eep1, eep2))
                logger.warning('new spline_level %i' % k)

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
            peak_dict = math_utils.find_peaks(intp_col - (m * axnew + b))
        else:
            peak_dict = math_utils.find_peaks(intp_col)

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
                    logger.error(mess_err)
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
                    logger.error(mess_err)
                return -1

        if parametric_interp is True:
            # closest point in interpolation to data
            ind, dif = math_utils.closest_match2d(almost_ind,
                                                  track.data[col][inds][non_dupes],
                                                  np.log10(track.data.AGE[inds][non_dupes]),
                                                  intp_col, agenew)
        else:
            # closest point in interpolation to data
            ind, dif = math_utils.closest_match2d(almost_ind, xdata, ydata,
                                                  xnew, ynew)

        if ind == -1:
            # didn't find anything.
            return ind

        if dif > dif_tol:
            # closest match was too far away from orig.
            if diff_err is not None:
                logger.debug(diff_err)
            else:
                logger.error('bad match %s-%s M=%.3f' % (eep1, eep2,
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
            self.ptcri.iptcri = np.zeros(len(eep_obj.eep_list_hb), dtype=int)
        else:
            # Sandro's definitions. (I don't use his HB EEPs)
            mptcri = ptcri.data_dict['M%.3f' % track.mass]
            track.ptcri.sptcri = \
                np.concatenate([np.nonzero(track.data.MODE == m)[0]
                                for m in mptcri])

            if len(ptcri.please_define) > 0:
                # Initialize iptcri
                self.ptcri.iptcri = np.zeros(len(eep_obj.eep_list), dtype=int)

                # Get the values that we won't be replacing.
                pinds = np.array([i for i, a in enumerate(self.ptcri.eep.eep_list)
                                  if a in self.ptcri.sandro_eeps])

                sinds = np.array([i for i, a in enumerate(self.ptcri.sandro_eeps)
                                  if a in self.ptcri.eep.eep_list])
                self.ptcri.iptcri[pinds] = mptcri[sinds] - 2
                
                # but if the track did not actually make it to that EEP, no -2!
                self.ptcri.iptcri[self.ptcri.iptcri < 0] = 0
                
                # and if sandro cut the track before it reached this point,
                # no index error!
                self.ptcri.iptcri[self.ptcri.iptcri > len(track.data.MODE)] = 0
                # BAD.
                #track.ptcri = deepcopy(self.ptcri)

                # define the eeps
                self.define_eep_stages(track, hb=hb, plot_dir=plot_dir,
                                       diag_plot=diag_plot, debug=debug)
                # SUPER BAD.
                #track.ptcri = deepcopy(self.ptcri)

            else:
                # copy sandros dict.
                self.ptcri.iptcri = ptcri.data_dict['M%.3f' % track.mass]
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
            logger.warning('only %i indices to fit... %s-%s' % (len(non_dupes)))
            logger.warning('new spline_level %i' % k)

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
            logger.warning('only %i indices to fit...' % (len(non_dupes)))
            logger.warning('new spline_level %i' % k)

        tckp, u = splprep([np.log10(track.data.AGE[inds][non_dupes]),
                           track.data.LOG_TE[inds][non_dupes],
                           track.data.LOG_L[inds][non_dupes]],
                           s=s, k=k, nest=nest)

        xdata = track.data.LOG_TE[inds][non_dupes]
        ave_data_step = np.round(np.mean(np.abs(np.diff(xdata))), 4)
        step_size = np.max([ave_data_step, min_step])
        
        arb_arr = np.arange(0, 1, step_size)
        return tckp, step_size, non_dupes


class TrackDiag(object):
    def __init__(self):
        pass

    def diagnostic_plots(self, track, inds=None, annotate=True, fig=None,
                         axs=None):

        xcols = ['AGE', 'AGE', 'LOG_TE']
        xreverse = [False, False, True]

        ycols = ['LOG_L', 'LOG_TE', 'LOG_L']
        yreverse = [False, False, False]

        plt_kws = [{'lw': 2, 'color': 'black'},
                   {'marker': 'o', 'ls': '', 'color': 'darkblue'}]

        if fig is None:
            fig = plt.figure(figsize=(10, 10))
        if axs is None:
            axs = []

        for i, (x, y, xr, yr) in enumerate(zip(xcols, ycols, xreverse,
                                               yreverse)):
            axs.append(plt.subplot(2, 2, i + 1))

            if x == 'AGE':
                xdata = np.log10(track.data[x])
            else:
                xdata = track.data[x]

            if inds is not None:
                axs[i].plot(xdata[inds], track.data[y][inds], **plt_kws[1])
            else:
                inds, = np.nonzero(track.data.AGE > 0.2)
                axs[i].plot(xdata[inds], track.data[y][inds], **plt_kws[0])

            axs[i].set_xlabel('$%s$' % x.replace('_', '\ '))
            axs[i].set_ylabel('$%s$' % y.replace('_', '\ '))

            if annotate is True:
                self.annotate_plot(axs[i], xdata, y)

            if xr is True:
                axs[i].set_xlim(axs[i].get_xlim()[::-1])
            if yr is True:
                axs[i].set_ylim(axs[i].get_ylim()[::-1])
            axs[i].set_title('$%s$' %
                             self.name.replace('_', '\ ').replace('.PMS', ''))

        return fig, axs

    def plot_track(self, track, xcol, ycol, reverse_x=False, reverse_y=False,
                   ax=None, inds=None, plt_kw={}, annotate=False, clean=False,
                   ainds=None, sandro=False, cmd=False, convert_mag_kw={},
                   xdata=None, ydata=None, hb=False, xnorm=False, ynorm=False,
                   arrow_on_line=False):
        '''
        ainds is passed to annotate plot, and is to only plot a subset of crit
        points.
        sandro = True will plot sandro's ptcris.
        '''
        if ax is None:
            plt.figure()
            ax = plt.axes()

        if len(plt_kw) != 0:
            # not sure why, but every time I send marker='o' it also sets
            # linestyle = '-' ...
            if 'marker' in plt_kw.keys():
                if not 'ls' in plt_kw.keys() or not 'linestyle' in plt_kw.keys():
                    plt_kw['ls'] = ''

        if clean is True and inds is None:
            # Non physical inds go away.
            inds, = np.nonzero(track.data.AGE > 0.2)
        if ydata is None:
            ydata = track.data[ycol]

        if xdata is None:
            if cmd is True:
                if len(convert_mag_kw) != 0:
                    import astronomy_utils
                    photsys = convert_mag_kw['photsys']
                    dmod = convert_mag_kw.get('dmod', 0.)
                    Av = convert_mag_kw.get('Av', 0.)
                    Mag1 = track.data[xcol]
                    Mag2 = track.data[ycol]
                    mag1 = astronomy_utils.Mag2mag(Mag1, xcol, photsys,
                                                   Av=Av, dmod=dmod)
                    mag2 = astronomy_utils.Mag2mag(Mag2, ycol, photsys,
                                                   Av=Av, dmod=dmod)
                    xdata = mag1 - mag2
                    ydata = mag2
                else:
                    xdata = track.data[xcol] - track.data[ycol]
            else:
                xdata = track.data[xcol]

        if xnorm is True:
            xdata /= np.max(xdata)

        if ynorm is True:
            ydata /= np.max(ydata)

        if inds is not None:
            #inds = [i for i in inds if i > 0]
            ax.plot(xdata[inds], ydata[inds], **plt_kw)
        else:
            ax.plot(xdata, ydata, **plt_kw)

        if reverse_x:
            ax.set_xlim(ax.get_xlim()[::-1])

        if reverse_y:
            ax.set_ylim(ax.get_ylim()[::-1])

        if annotate:
            ax = self.annotate_plot(track, ax, xdata, ydata, inds=ainds,
                                    sandro=sandro, hb=hb, cmd=cmd)
        if arrow_on_line is True:
            # hard coded to be 10 equally spaced points...
            ages = np.linspace(np.min(track.data.AGE[inds]), np.max(track.data.AGE[inds]), 10)
            indz, difs = zip(*[math_utils.closest_match(i, track.data.AGE[inds]) for i in ages])
            # I LOVE IT arrow on line... AOL BUHSHAHAHAHAHA
            aol_kw = deepcopy(plt_kw)
            if 'color' in aol_kw:
                aol_kw['fc'] = aol_kw['color']
                del aol_kw['color']
            indz = indz[indz>0]
            print track.data.LOG_L[inds][np.array([indz])]
            rspg.arrow_on_line(ax, xdata, ydata, indz, plt_kw=plt_kw)
        return ax

    def annotate_plot(self, track, ax, xcol, ycol, inds=None, sandro=False,
                      cmd=False, hb=False):
        '''
        if a subset of ptcri inds are used, set them in inds. If you want
        sandro's ptcri's sandro=True, will also change the face color of the
        label bounding box so you can have both on the same plot.
        '''
        if sandro is False:
            ptcri = track.ptcri.iptcri
            fc = 'blue'
        else:
            ptcri = track.ptcri.sptcri
            fc = 'red'

        ptcri_kw = {'sandro': sandro, 'hb': hb}
        if inds is None:
            #inds = np.array([p for p in ptcri if p > 0])
            inds = ptcri
            labels = ['$%s$' %
                      track.ptcri.get_ptcri_name(i, **ptcri_kw).replace('_', '\ ')
                      for i in range(len(inds))]
        else:
            iplace = np.array([np.nonzero(ptcri == i)[0][0] for i in inds])
            labels = ['$%s$' %
                      track.ptcri.get_ptcri_name(int(i), **ptcri_kw).replace('_', '\ ')
                      for i in iplace]

        if type(xcol) == str:
            xdata = track.data[xcol]
        else:
            xdata = xcol

        if type(ycol) == str:
            ydata = track.data[ycol]
        else:
            ydata = ycol

        if cmd is True:
            xdata = xdata - ydata
        # label stylings
        bbox = dict(boxstyle='round, pad=0.5', fc=fc, alpha=0.5)
        arrowprops = dict(arrowstyle='->', connectionstyle='arc3, rad=0')

        for i, (label, x, y) in enumerate(zip(labels, xdata[inds],
                                          ydata[inds])):
            # varies the labels placement... default is 20, 20
            xytext = ((-1.) ** (i - 1.) * 20, (-1.) ** (i + 1.) * 20)
            ax.annotate(label, xy=(x, y), xytext=xytext, fontsize=10,
                        textcoords='offset points', ha='right', va='bottom',
                        bbox=bbox, arrowprops=arrowprops)
        return ax

    def maxmin(self, track, col, inds=None):
        '''
        returns the max and min of a column in Track.data. use inds to index.
        WHY IS THIS HERE NOT SOMEWHERE NICER? PRETTIER, WITH A VIEW?
        '''
        arr = track.data[col]
        if inds is not None:
            arr = arr[inds]
        ma = np.max(arr)
        mi = np.min(arr)
        return (ma, mi)

    def check_ptcris(self, track, hb=False, plot_dir=None, sandro_plot=False,
                    xcol='LOG_TE', ycol='LOG_L'):
        '''
        plot of the track, the interpolation, with each eep labeled
        '''
        all_inds, = np.nonzero(track.data.AGE > 0.2)

        iptcri, = np.nonzero(self.ptcri.iptcri > 0)
        ptcri_kw = {'sandro': False, 'hb': hb}
        last = self.ptcri.get_ptcri_name(int(iptcri[-1]), **ptcri_kw)

        if hb is False:
            plots = [['PMS_BEG', 'PMS_MIN', 'PMS_END', 'MS_BEG'],
                     ['MS_BEG', 'MS_TMIN', 'MS_TO', 'SG_MAXL', 'RG_MINL'],
                     ['RG_MINL', 'RG_BMP1', 'RG_BMP2', 'RG_TIP'],
                     ['RG_TIP', 'HE_BEG', 'YCEN_0.550', 'YCEN_0.500',
                      'YCEN_0.400'],
                     ['YCEN_0.400', 'YCEN_0.200', 'YCEN_0.100'],
                     ['YCEN_0.100', 'YCEN_0.000', 'TPAGB']]
        else:
            plots = [['HB_BEG', 'YCEN_0.500', 'YCEN_0.400', 'YCEN_0.200',
                     'YCEN_0.100', 'YCEN_0.005'],
                     ['YCEN_0.005', 'AGB_LY1', 'AGB_LY2']]

        for i, plot in enumerate(plots):
            if last in plot:
                nplots = i + 1

        line_pltkw = {'color': 'black'}
        point_pltkw = {'marker': 'o', 'ls': ''}
        (fig, axs) = rspg.setup_multiplot(nplots, **{'figsize': (12, 8)})

        for i, ax in enumerate(np.ravel(axs)):
            if i == len(plots):
                continue
            inds = [self.ptcri.get_ptcri_name(cp, **ptcri_kw)
                    for cp in plots[i]]
            inds = self.ptcri.iptcri[inds][np.nonzero(self.ptcri.iptcri[inds])[0]]
            if np.sum(inds) == 0:
                continue

            ax = self.plot_track(track, xcol, ycol, ax=ax, inds=all_inds,
                                 reverse_x=True, plt_kw=line_pltkw)
            ax = self.plot_track(track, xcol, ycol, ax=ax, inds=inds,
                                 plt_kw=point_pltkw, annotate=True, ainds=inds,
                                 hb=hb)

            if hasattr(self, 'match_data'):
                # over plot the match interpolation
                for col in [xcol, ycol]:
                    if col == 'LOG_L':
                        tmp = (4.77 - self.match_data.T[3]) / 2.5
                    if 'age' in col.lower():
                        tmp = self.match_data.T[2]
                        #if not 'log' in col.lower():
                        #    tmp = 10 ** tmp
                    if col == 'LOG_TE':
                        tmp = self.match_data.T[2]
                    if col == xcol:
                        x = tmp
                    if col == ycol:
                        y = tmp

                ax.plot(x, y, lw=2, color='green')

            xmax, xmin = self.maxmin(track, xcol, inds=inds)
            ymax, ymin = self.maxmin(track, ycol, inds=inds)

            if np.diff((xmin, xmax)) == 0:
                xmin -= 0.1
                xmax += 0.1

            if np.diff((ymin, ymax)) == 0:
                ymin -= 0.5
                ymax += 0.5

            offx = 0.05
            offy = 0.1
            ax.set_xlim(xmax + offx, xmin - offx)
            ax.set_ylim(ymin - offy, ymax + offy)
            #ax.set_xlim(goodlimx)
            #ax.set_ylim(goodlimy)
            ax.set_xlabel('$%s$' % xcol.replace('_', r'\! '), fontsize=20)
            ax.set_ylabel('$%s$' % ycol.replace('_', r'\! '), fontsize=20)
            if 'age' in xcol:
                ax.set_xscale('log')
        title = 'M = %.3f Z = %.4f Y = %.4f' % (track.mass, track.Z, track.Y)
        fig.suptitle(title, fontsize=20)
        if hb is True:
            extra = '_HB'
        else:
            extra = ''
        if xcol != 'LOG_TE':
            extra += '_%s' % xcol

        figname = 'ptcri_Z%g_Y%g_M%.3f%s.png' % (track.Z, track.Y, track.mass,
                                                 extra)
        if plot_dir is not None:
            figname = os.path.join(plot_dir, figname)
        plt.savefig(figname)
        plt.close()
        logger.info('wrote %s' % figname)

        if hb is False and sandro_plot is True:
            self.plot_sandro_ptcri(track, plot_dir=plot_dir)

    def plot_sandro_ptcri(self, track, plot_dir=None):
        ax = self.plot_track(track, 'LOG_TE', 'LOG_L', reverse_x=1,
                             inds=np.nonzero(track.data.AGE > 0.2)[0])

        ax = self.plot_track(track, 'LOG_TE', 'LOG_L', ax=ax, annotate=True,
                             sandro=True)
        title = 'M = %.3f Z = %.4f Y = %.4f' % (track.mass, track.Z, track.Y)
        ax.set_title(title, fontsize=20)
        figname = 'sandro_ptcri_Z%g_Y%g_M%.3f.png' % (track.Z, track.Y,
                                                      track.mass)
        if plot_dir is not None:
            figname = os.path.join(plot_dir, figname)
        plt.savefig(figname)
        plt.close()
        return


def inds_between_ptcris(ptcri, name1, name2, sandro=True):
    '''
    returns the indices from [name1, name2)
    this is iptcri, not mptcri
    they will be the same inds that can be used in Track.data
    '''
    if sandro is True:
        # this must be added in Tracks.load_critical_points!
        inds = ptcri.sptcri
    else:
        inds = ptcri.iptcri

    try:
        first = inds[ptcri.get_ptcri_name(name1, sandro=sandro)]
    except IndexError:
        first = 0

    try:
        second = inds[ptcri.get_ptcri_name(name2, sandro=sandro)]
    except IndexError:
        second = 0

    inds = np.arange(first, second)
    return inds

class critical_point(object):
    '''
    class to hold ptcri data from Sandro's ptcri file and input eep_obj
    which tells which critical points of Sandro's to ignore and which new
    ones to define. Definitions of new eeps are in the Track class.
    '''
    def __init__(self, filename, eep_obj=None):
        self.load_ptcri(filename, eep_obj=eep_obj)
        self.base, self.name = os.path.split(filename)
        self.get_args_from_name(filename)
    
    def get_args_from_name(self, filename):
        '''
        god i wish i knew regex
        '''
        zstr = filename.split('_Z')[-1]
        self.Z = float(zstr.split('_')[0])
        ystr = filename.replace('.dat', '').split('_Y')[-1].split('_')[0]
        if ystr.endswith('.'):
            ystr = ystr[:-1]
        self.Y = float(ystr)

    def get_ptcri_name(self, val, sandro=True, hb=False):
        if sandro is True:
            search_dict = self.sandros_dict
        elif hb is True:
            search_dict = self.key_dict_hb
        else:
            search_dict = self.key_dict

        if type(val) == int:
            return [name for name, pval in search_dict.items()
                    if pval == val][0]
        elif type(val) == str:
            return [pval for name, pval in search_dict.items()
                    if name == val][0]

    def load_ptcri(self, filename, eep_obj=None):
        '''
        reads the ptcri*dat file. If there is an eep_obj, it will flag the
        missing eeps in the ptcri file and only read the eeps that match both
        the eep_list and the ptcri file.
        should be part of eep...
        '''
        with open(filename, 'r') as f:
            lines = f.readlines()

        # the lines have the path name, and the path has F7.
        begin, = [i for i in range(len(lines)) if lines[i].startswith('#')
                  and 'F7' in lines[i]]

        # the final column is a filename.
        col_keys = lines[begin + 1].replace('#', '').strip().split()[3:-1]

        # useful to save what Sandro defined
        self.sandro_eeps = col_keys
        self.sandros_dict = dict(zip(col_keys, range(len(col_keys))))

        # ptcri file has filename as col #19 so skip the last column
        usecols = range(0, 18)
        # invalid_raise will skip the last rows that Sandro uses to fake the
        # youngest MS ages (600Msun).
        data = np.genfromtxt(filename, usecols=usecols, skip_header=begin + 2,
                             invalid_raise=False)
        self.data = data
        self.masses = data[:, 1]

        # ptcri has all track data, but this instance only cares about one mass.
        data_dict = {}
        for i in range(len(data)):
            str_mass = 'M%.3f' % self.masses[i]
            data_dict[str_mass] = data[i][3:].astype('int')

        self.data_dict = data_dict
        self.please_define = []
        self.please_define_hb = []

        if eep_obj is not None:
            self.key_dict = dict(zip(eep_obj.eep_list,
                                     range(len(eep_obj.eep_list))))
            self.please_define = [c for c in eep_obj.eep_list
                                  if c not in col_keys]
            
            if eep_obj.eep_list_hb is not None:
                self.key_dict_hb = dict(zip(eep_obj.eep_list_hb,
                                        range(len(eep_obj.eep_list_hb))))
                # there is no mixture between Sandro's HB eeps since there
                # are no HB eeps in the ptcri files. Define them all here.
                self.please_define_hb = eep_obj.eep_list_hb
            
            self.eep = eep_obj
        else:
            self.please_define = []
            self.key_dict = self.sandros_dict

    def load_sandro_eeps(self, track):
        try:
            mptcri = self.data_dict['M%.3f' % track.mass]
        except KeyError:
            print 'No M%.3f in ptcri.data_dict.' % track.mass
            return -1
        track.sptcri = \
            np.concatenate([np.nonzero(track.data.MODE == m)[0]
                                    for m in mptcri])
        #print track.mass, track.sptcri
        #track.ptcri.sptcri = self.sptcri


class eep(object):
    '''
    a simple class to hold eep data. Gets added as an attribute to ptcri class.
    '''
    def __init__(self, eep_list, eep_lengths=None, eep_list_hb=None,
                 eep_lengths_hb=None):
        self.eep_list = eep_list
        self.nticks = eep_lengths
        self.eep_list_hb = eep_list_hb
        self.nticks_hb = eep_lengths_hb


class TrackSet(object):
    '''
    
    '''
    def __init__(self, inputs):
        
        if inputs.ptcrifile_loc is not None or inputs.ptcri_file is not None:
            self.load_ptcri_eep(inputs)
        else:
            self.ptcri = None

        self.tracks_base = os.path.join(inputs.tracks_dir, inputs.prefix)
        if inputs.hb_only is False:
            self.load_tracks(track_search_term=inputs.track_search_term,
                             masses=inputs.masses)
        if inputs.hb is True:
            self.load_tracks(track_search_term=inputs.hbtrack_search_term,
                             hb=inputs.hb,
                             masses=inputs.masses)

    def load_ptcri_eep(self, inputs):
        '''
        load the ptcri and eeps, simple call to the objects.
        way isn't this in eep?
        '''
        self.ptcri = None
        self.eep = None
        if hasattr(inputs, 'ptcri_file'):
            self.ptcri_file = inputs.ptcri_file
        else:
            self.prefix = inputs.prefix
            if inputs.from_p2m is True:
                # this is the equivalent of Sandro's ptcri files, but mine.
                search_term = 'p2m*%s*dat' % self.prefix
                self.ptcri_file, = fileIO.get_files(inputs.ptcrifile_loc,
                                                    search_term)
                logger.info('reading ptcri from saved p2m file.')
            else:
                search_term = 'pt*%s*dat' % self.prefix
                self.ptcri_file, = fileIO.get_files(inputs.ptcrifile_loc,
                                                    search_term)

        if inputs.eep_list is not None:
            eep_kw = {'eep_lengths': inputs.eep_lengths,
                      'eep_list_hb': inputs.eep_list_hb,
                      'eep_lengths_hb': inputs.eep_lengths_hb}
            self.eep = eep(inputs.eep_list, **eep_kw)

        self.ptcri = critical_point(self.ptcri_file, eep_obj=self.eep)

    def load_tracks(self, track_search_term='*F7_*PMS', hb=False, masses=None):
        '''
        loads tracks or hb tracks, can load subset if masses (list, float, or
        string) is set. If masses is string, it must be have format like:
        '%f < 40' and it will use masses that are less 40.
        '''
        track_names = np.array(fileIO.get_files(self.tracks_base,
                               track_search_term))
        assert len(track_names) != 0, \
            'No tracks found: %s/%s' % (self.tracks_base, track_search_term)
        mass = np.array([os.path.split(t)[1].split('_M')[1].split('.P')[0]
                         for t in track_names], dtype=float)
        track_names = track_names[np.argsort(mass)]
        mass = mass[np.argsort(mass)]

        # only do a subset of masses
        if masses is not None:
            if type(masses) == float:
                masses = [masses]
            elif type(masses) == str:
                track_masses = [i for i in range(len(mass))
                                if eval(masses % mass[i])]
            if type(masses) == list:
                track_masses = []
                for set_mass in masses:
                    try:
                        track_masses.append(list(mass).index(set_mass))
                    except ValueError:
                        pass
                track_masses = np.array(track_masses)
        else:
            track_masses = np.argsort(mass)


        track_str = 'track'
        mass_str = 'masses'
        if hb is True:
            track_str = 'hb%s' % track_str
            mass_str = 'hb%s' % mass_str
        self.__setattr__('%s_names' % track_str, track_names[track_masses])

        self.__setattr__('%ss' % track_str, [Track(track, ptcri=self.ptcri,
                                                   min_lage=0., cut_long=0)
                                             for track in track_names[track_masses]])

        self.__setattr__('%s' % mass_str,
                         np.round([t.mass for t in
                                   self.__getattribute__('%ss' % track_str)], 3))

    def save_ptcri(self, filename=None, hb=False):
        #assert hasattr(self, ptcri), 'need to have ptcri objects loaded'
        if hb is True:
            tracks = self.hbtracks
        else:
            tracks = self.tracks

        if filename is None:
            base, name = os.path.split(self.ptcri_file)
            filename = os.path.join(base, 'p2m_%s' % name)
            if hb is True:
                filename = filename.replace('p2m', 'p2m_hb')

        sorted_keys, inds = zip(*sorted(self.ptcri.key_dict.items(),
                                        key=lambda (k, v): (v, k)))

        header = '# critical points in F7 files defined by sandro, basti, and phil \n'
        header += '# i mass lixo %s fname \n' % (' '.join(sorted_keys))
        with open(filename, 'w') as f:
            f.write(header)
            linefmt = '%2i %.3f 0.0 %s %s \n'
            for i, track in enumerate(tracks):
                self.ptcri.please_define = []
                # this line should just slow everything down, why is it here?
                self.load_critical_points(track, eep_obj=self.eep, hb=hb,
                                          ptcri=self.ptcri, diag_plot=False)
                ptcri_str = ' '.join(['%5d' % p for p in track.ptcri.iptcri])
                f.write(linefmt % (i+1, track.mass, ptcri_str,
                                   os.path.join(track.base, track.name)))

        logger.info('wrote %s' % filename)

    def plot_all_tracks(self, tracks, xcol, ycol, annotate=True, ax=None,
                        reverse_x=False, sandro=True, cmd=False,
                        convert_mag_kw={}, hb=False, plot_dir=None,
                        zoomin=True, one_plot=False):
        '''
        It would be much easier to discern breaks in the sequences if you did
        three separate plots: PMS_BEG to MS_BEG,
        MS_BEG to RG_TIP, and RG_TIP to TPAGB.
        As it stands, one is trying to tell RGB features from AGB features.
        Likewise, there is such a small color difference between some of the
        different points that I'm not entire sure what I'm seeing.

        I see a sharp break in the RGB bump and RGB tip sequences.
        Are those visible in the isochrones?
        '''
        line_pltkw = {'color': 'black', 'alpha': 0.1}

        if one_plot is True:
            for t in tracks:
                td = TrackDiag()
                all_inds, = np.nonzero(t.data.AGE > 0.2)
                ax = td.plot_track(t, xcol, ycol, ax=ax, inds=all_inds,
                                     plt_kw=line_pltkw, cmd=cmd,
                                     convert_mag_kw=convert_mag_kw, hb=hb)
            return ax

        ptcri_kw = {'sandro': sandro, 'hb': hb}

        if hb is False:
            if sandro is True:
                plots = [['PMS_BEG', 'PMS_MIN', 'PMS_END', 'NEAR_ZAM', 'MS_BEG'],
                         ['MS_BEG', 'POINT_B', 'POINT_C'],
                         ['POINT_C', 'RG_BASE', 'RG_BMP1', 'RG_BMP2', 'RG_TIP'],
                         ['Loop_A', 'Loop_B', 'Loop_C', 'TPAGB']]
                fig_extra = ['pms', 'ms', 'rg', 'loop']
            else:
                plots = [['PMS_BEG', 'PMS_MIN', 'PMS_END', 'MS_BEG'],
                         ['MS_BEG', 'MS_TMIN', 'MS_TO', 'SG_MAXL', 'RG_MINL',
                          'RG_BMP1', 'RG_BMP2', 'RG_TIP'],
                         ['RG_TIP', 'HE_BEG', 'YCEN_0.550', 'YCEN_0.500'],
                         ['YCEN_0.550', 'YCEN_0.500', 'YCEN_0.400', 'YCEN_0.200',
                          'YCEN_0.100', 'YCEN_0.000', 'TPAGB']]
                fig_extra = ['pms', 'ms', 'rg', 'ycen']
        else:
            plots = [['HB_BEG', 'YCEN_0.500', 'YCEN_0.400', 'YCEN_0.200',
                     'YCEN_0.100', 'YCEN_0.005', 'AGB_LY1', 'AGB_LY2']]
            # overwriting kwargs!!
            ptcri_kw['sandro'] = False
            fig_extra = ['hb']

        assert len(fig_extra) == len(plots), \
            'need correct plot name extensions.'

        xlims = np.array([])
        ylims = np.array([])
        for j in range(len(plots)):
            fig, ax = plt.subplots()
            if annotate is True:
                point_pltkw = {'marker': '.', 'ls': '', 'alpha': 0.5}
                cols = rspg.discrete_colors(len(plots[j]), colormap='spectral')
                labs = ['$%s$' % p.replace('_', '\_') for p in plots[j]]

            didit = 0
            xlimi = np.array([])
            ylimi = np.array([])
            for t in tracks:
                if sandro is False:
                    ptcri = t.iptcri
                else:
                    ptcri = t.sptcri
                ainds = [t.ptcri.get_ptcri_name(cp, **ptcri_kw)
                         for cp in plots[j]]
                ainds = [i for i in ainds if i < len(ptcri)]
                
                inds = ptcri[ainds]
                

                if np.sum(inds) == 0:
                    continue

                some_inds = np.arange(inds[0], inds[-1])

                ax = self.plot_track(t, xcol, ycol, ax=ax, inds=some_inds,
                                     plt_kw=line_pltkw, cmd=cmd, clean=False,
                                     convert_mag_kw=convert_mag_kw)

                #line_pltkw['alpha'] = 1.
                #ax = self.plot_track(t, xcol, ycol, ax=ax, inds=some_inds,
                #                     plt_kw=line_pltkw, cmd=cmd,
                #                     convert_mag_kw=convert_mag_kw)

                xlims = np.append(xlims, np.array(ax.get_xlim()))
                ylims = np.append(ylims, np.array(ax.get_ylim()))

                if annotate is True:
                    xdata = t.data[xcol]
                    ydata = t.data[ycol]

                    if cmd is True:
                        xdata = t.data[xcol] - t.data[ycol]
                    pls = []
                    for i in range(len(inds)):
                        x = xdata[inds[i]]
                        y = ydata[inds[i]]
                        xlimi = np.append(xlimi, (np.min(x), np.max(x)))
                        ylimi = np.append(ylimi, (np.min(y), np.max(y)))
                        pl, = ax.plot(x, y, color=cols[i], **point_pltkw)
                        pls.append(pl)
                    ax.text(xdata[inds[0]], ydata[inds[0]], '%.3f' % t.mass,
                            fontsize=8, ha='right')
                    # only save the legend if all the points are made
                    if len(inds) == len(plots[j]):
                        didit += 1
                        if didit == 1:
                            plines = pls

            if zoomin is True:
                ax.set_xlim(np.min(xlimi), np.max(xlimi))
                ax.set_ylim(np.min(ylimi), np.max(ylimi))
            else:
                ax.set_xlim(np.min(xlims), np.max(xlims))

            if reverse_x is True:
                ax.set_xlim(ax.get_xlim()[::-1])

            if annotate is True:
                ax.legend(plines, labs, numpoints=1, loc=0, frameon=False)

            ylab = ycol.replace('_', '\ ')
            xlab = xcol.replace('_', '\ ')
            figname = '%s_%s_%s_%s.png' % (self.prefix, xcol, ycol,
                                           fig_extra[j])

            if cmd is True:
                xlab = '%s-%s' % (xlab, ylab)

            ax.set_xlabel('$%s$' % xlab)
            ax.set_ylabel('$%s$' % ylab)

            if plot_dir is not None:
                figname = os.path.join(plot_dir, figname)
            plt.savefig(figname)
            logger.info('wrote %s' % figname)
            #plt.close()
        return

    def squish(self, *attrs, **kwargs):
        '''
        bad coder: I took this from Galaxies.galaxy. Some day I can make
        it all great, but first I need to have the same data fmts for all these
        things...

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

        kwargs: inds choose which tracks to include (all by default)
        new_attrs: if you don't like the attributes set.
        '''
        inds = kwargs.get('inds', np.arange(len(self.tracks)))
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

            new_list = [self.tracks[j].data[attr] for j in inds]
            # is attr an array of arrays, or is it now an array?
            try:
                new_val = np.concatenate(new_list)
            except ValueError:
                new_val = np.array(new_list)

            self.__setattr__(new_attr, new_val)

    def all_inds_of_eep(self, eep_name):
        '''
        get all the ind for all tracks of some eep name, for example 
        want ms_to of the track set? set eep_name = point_c if sandro==True.
        '''
        inds = []
        for track in self.tracks:
            check = track.ptcri.load_sandro_eeps(track)
            if check == -1:
                inds.append(-1)
                continue
            eep_ind = track.ptcri.get_ptcri_name(eep_name)
            if len(track.ptcri.sptcri) <= eep_ind:
                inds.append(-1)
                continue
            data_ind = track.ptcri.sptcri[eep_ind]
            inds.append(data_ind)
        return inds


class MatchTracks(object):
    '''
    a simple check of the output from TracksForMatch. I want it to run on the
    same input file as TracksForMatch.
    '''
    def __init__(self, inputs):

        self.tracks_base = inputs.outfile_dir
        self.prefix = inputs.prefix
        # hard coding where the match files are kept tracks_base/match/
        all_track_names = fileIO.get_files(self.tracks_base,
                                           inputs.track_search_term)

        self.hbtrack_names = [t for t in all_track_names if 'HB' in t]
        self.track_names = [t for t in all_track_names
                            if t not in self.hbtrack_names]
        self.tracks = [self._load_track(t) for t in self.track_names]
        for i, t in enumerate(self.tracks):
            test = np.diff(t['logAge']) > 0
            if False in test:
                bads, = np.nonzero(np.diff(t['logAge']) < 0)
                bads1, = np.nonzero(np.diff(t['logAge']) == 0)
                if len(bads) != 0:
                    print 'Age not monotonicly increasing!'
                    print self.track_names[i], bads, t['logAge'][bads]
                if len(bads1) != 0:
                    print 'Identical values', self.track_names[i], len(t['logAge'])
                    for j in range(len(bads1)):
                        print t[bads1[j]], bads1[j]
                        print t[bads1[j] + 1], bads1[j]+1
                        print ''
        self.hbtracks = [self._load_track(t) for t in self.hbtrack_names]

        self.eep_list = inputs.eep_list
        self.eep_lengths = inputs.eep_lengths
        self.eep_list_hb = inputs.eep_list_hb

        pat_kw = {'eep_list': self.eep_list,
                  'eep_lengths': self.eep_lengths,
                  'plot_dir': inputs.outfile_dir}

        self._plot_all_tracks(self.tracks, **pat_kw)

        pat_kw[xcol] = 'logAge'
        self._plot_all_tracks(self.tracks, **pat_kw)

        if self.eep_list_hb is not None:
            pat_kw[extra] = '_HB'
            self._plot_all_tracks(self.hbtracks, **pat_kw)
            del pat_kw[xcol]
            self._plot_all_tracks(self.hbtracks, **pat_kw)


    def _load_track(self, filename):
        '''
        '''
        # the filename actually contains Mbol, but I convert it in genfromtxt.
        names = 'logAge', 'MASS', 'LOG_TE', 'LOG_L', 'logg', 'CO'
        data = np.genfromtxt(filename, names=names,
                             converters={3: lambda m: (4.77 - float(m)) / 2.5})
        data = data.view(np.recarray)
        return data

    def _plot_all_tracks(self, tracks, eep_list=None, eep_lengths=None,
                         plot_dir=None, extra='', xcol='LOG_TE', ycol='LOG_L'):
        if extra == '':
            extra = '_%s' % xcol
        else:
            extra += '_%s' % xcol

        if eep_lengths is not None:
            eep_lengths = map(int, np.insert(np.cumsum(eep_lengths), 0, 1))
        line_pltkw = {'color': 'black', 'alpha': 0.3}
        point_pltkw = {'marker': '.', 'ls': '', 'alpha': 0.5}
        cols = rspg.discrete_colors(len(eep_list), colormap='spectral')
        labs = [p.replace('_', '\_') for p in eep_list]

        fig, ax = plt.subplots(figsize=(16, 9))
        # fake lengend
        [ax.plot(9999, 9999, color=cols[i], label=labs[i], **point_pltkw)
         for i in range(len(eep_list))]

        [ax.plot(t[xcol], t[ycol], **line_pltkw) for t in tracks]
        xlims = np.array([])
        ylims = np.array([])
        for t in tracks:
            for i in range(len(eep_lengths)):
                x = t[xcol]
                y = t[ycol]
                ind = eep_lengths[i] - 1

                if (len(x) < ind):
                    continue
                ax.plot(x[ind], y[ind], color=cols[i], **point_pltkw)
                xlims = np.append(xlims, (np.min(x[ind]), np.max(x[ind])))
                ylims = np.append(ylims, (np.min(y[ind]), np.max(y[ind])))
        ax.set_title('$%s$' % self.prefix.replace('_', '\ '))
        ax.set_xlim(np.max(xlims), np.min(xlims))
        ax.set_ylim(np.min(ylims), np.max(ylims))
        ax.set_xlabel('$%s$' % xcol.replace('_', '\! '), fontsize=20)
        ax.set_ylabel('$%s$' % ycol, fontsize=20)
        ax.legend(loc=0, numpoints=1, frameon=0)
        figname = 'match_%s%s.png' % (self.prefix, extra)
        if plot_dir is not None:
            figname = os.path.join(plot_dir, figname)
        plt.savefig(figname, dpi=300)
    

class TracksForMatch(TrackSet, DefineEeps, TrackDiag):
    '''
    This class is for interpolating tracks for use in MATCH. While the
    DefineEeps code is made for one track at a time, TracksForMatch takes a
    track set as input.
    '''
    def __init__(self, inputs):
        # load all tracks
        TrackSet.__init__(self, inputs)

        DefineEeps.__init__(self)

        if inputs.do_interpolation is True:
            self.match_interpolation(inputs)

    def match_interpolation(self, inputs):
        if inputs.hb_only is False:
            for track in self.tracks:
                # do the work! Assign eeps either from sandro, or eep_list and
                # make some diagnostic plots.
                track = self.load_critical_points(track, ptcri=self.ptcri,
                                                  plot_dir=inputs.plot_dir,
                                                  diag_plot=inputs.diag_plot,
                                                  debug=inputs.debug)
                if len(np.nonzero(track.ptcri.iptcri>0)[0]) < 3:
                    print 'skipping track because there is no ms_beg.', track.name
                    continue
                # make match output files.
                self.prepare_track(track, outfile_dir=inputs.outfile_dir)

                if inputs.diag_plot is True:
                    # make diagnostic plots
                    self.check_ptcris(track, plot_dir=inputs.plot_dir)
                    self.check_ptcris(track, plot_dir=inputs.plot_dir,
                                      xcol='AGE')

            # make summary diagnostic plots
            self.plot_all_tracks(self.tracks, 'LOG_TE', 'LOG_L', sandro=False,
                                 reverse_x=True, plot_dir=inputs.plot_dir)

        else:
            logger.info('Only doing HB.')

        # do the same as above but for HB.
        if inputs.hb is True:
            #self.hbtracks = []
            self.hbtrack_names = fileIO.get_files(self.tracks_base,
                                                  inputs.hbtrack_search_term)
            for track in self.hbtracks:
                track = self.load_critical_points(track, ptcri=self.ptcri,
                                                  hb=inputs.hb,
                                                  plot_dir=inputs.plot_dir,
                                                  debug=inputs.debug)
                #self.hbtracks.append(track)
                self.prepare_track(track, outfile_dir=inputs.outfile_dir,
                                   hb=inputs.hb)
                if inputs.diag_plot is True:
                    self.check_ptcris(track, hb=inputs.hb,
                                      plot_dir=inputs.plot_dir)

            self.plot_all_tracks(self.hbtracks, 'LOG_TE', 'LOG_L', hb=inputs.hb,
                                 reverse_x=True, plot_dir=inputs.plot_dir)
        try:
            fh.close()
            ch.close()
        except:
            pass

    def prepare_track(self, track, outfile='default', hb=False,
                      outfile_dir=None):
        if outfile == 'default':
            if outfile_dir is None or outfile_dir is 'default':
                outfile_dir = track.base
            outfile = os.path.join('%s' % outfile_dir,
                                   'match_%s.dat' % track.name.replace('.PMS', ''))
            header = '# logAge Mass logTe Mbol logg C/O \n'

        if hasattr(self.ptcri, 'eep'):
            if hb is True:
                nticks = self.ptcri.eep.nticks_hb
            else:
                nticks = self.ptcri.eep.nticks
        else:
            logger.info('using default spacing between eeps')
            nticks = np.repeat(200, len(self.ptcri.iptcri) - 1)

        assert nticks is not None, 'invalid eep_lengths, check eep list.'

        logTe = np.array([])
        logL = np.array([])
        Age = np.array([])
        new_eep_dict = {}
        tot_pts = 0
        ptcri_kw = {'sandro': False, 'hb': hb}
        for i in range(len(np.nonzero(track.ptcri.iptcri >= 0)[0]) - 1):
            this_eep = track.ptcri.get_ptcri_name(i, **ptcri_kw)
            next_eep = track.ptcri.get_ptcri_name(i+1, **ptcri_kw)
            ithis_eep = track.ptcri.iptcri[i]
            inext_eep = track.ptcri.iptcri[i+1]
            mess = '%.3f %s=%i %s=%i' % (track.mass,
                                         this_eep, ithis_eep,
                                         next_eep, inext_eep)

            if i != 0 and self.ptcri.iptcri[i+1] == 0:
                # except for PMS_BEG which == 0, skip if no iptcri.
                # this is not an error, just the end of the track.
                #logger.error(mess)
                #logger.error('skipping %s-%s\ncause the second eep is zippo.'
                #               % (this_eep, next_eep))
                continue
            
            if ithis_eep == -1:
                print mess
                continue

            inds = np.arange(ithis_eep, inext_eep + 1)
            if len(inds) == 0:
                logger.error(mess)
                logger.error(
                    'skipping %s-%s cause there are no inds between these crit pts.'
                    % (this_eep, next_eep))
                continue

            if len(inds) == 1:
                # include the last ind.
                inds = np.arange(ithis_eep, inext_eep + 1)
            tckp, _, _ = self.interpolate_te_l_age(track, inds)
            agenew, tenew, lnew = splev(np.linspace(0, 1, nticks[i] + 1), tckp)
            test = np.diff(agenew) > 0
            bads, = np.nonzero(test==False)
            if False in test:
                tckp, _, _ = self.interpolate_te_l_age(track, inds, k=1)
            agenew, tenew, lnew = splev(np.linspace(0, 1, nticks[i] + 1), tckp)
            test = np.diff(agenew) > 0
            bads, = np.nonzero(test==False)
            new_eep_dict[this_eep] = tot_pts
            tot_pts += nticks[i]
            logTe = np.append(logTe, tenew[:-1])
            logL = np.append(logL, lnew[:-1])
            Age = np.append(Age, 10 ** agenew[:-1])

            if False in test:
                print track.name
                print '\n AGE NOT MONOTONICALLY INCREASING', track.mass
                print 10**agenew[bads]
                print mess
                fig, (axs) = plt.subplots(ncols=2, figsize=(16, 10))
                for ax, xcol in zip(axs, ['AGE', 'LOG_TE']):
                    ax.scatter(track.data[xcol][track.ptcri.iptcri],
                               track.data.LOG_L[track.ptcri.iptcri],
                               s=60, c='k')
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                    ax.plot(track.data[xcol], track.data.LOG_L, color='k')
                    ax.plot(track.data[xcol], track.data.LOG_L, ',', color='k')
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                for ax, xcol in zip(axs, [10 ** agenew, tenew]):
                    ax.plot(xcol, lnew, lw=2, alpha=0.4)
                    ax.scatter(xcol, lnew, s=15, c=np.arange(len(xcol)),
                               cmap=plt.cm.Spectral)
                    ax.scatter(xcol[bads], lnew[bads], s=40,
                               c=np.arange(len(bads)),
                               cmap=plt.cm.Spectral)
                    ax.set_xscale('log')
                fig.suptitle('$%s$' % track.name.replace('_', r'\! '))
                plt.show()
        #  This was to make Leo's isochrones files... incomplete...
        #print new_eep_dict
        #track.write_trilegal_isotrack_ptcri(Age, logL, logTe, new_eep_dict)
        #line = ' '.join(map(str, np.sort(new_eep_dict.values())))
        #line += '\t !M=%.6f' % track.mass
        #print line

        Mbol = 4.77 - 2.5 * logL
        logg = -10.616 + np.log10(track.mass) + 4.0 * logTe - logL
        logAge = np.log10(Age)
        # CO place holder!
        CO = np.zeros(len(logL))
        mass_arr = np.repeat(track.mass, len(logL))
        to_write = np.column_stack((logAge, mass_arr, logTe, Mbol, logg, CO))

        with open(outfile, 'w') as f:
            f.write(header)
            np.savetxt(f, to_write, fmt='%.8f')
        logger.info('wrote %s' % outfile)
        self.match_data = to_write


def do_entire_set(inputs):
    tracks_dir = inputs.tracks_dir
    if inputs.prefixs == 'all':
        prefixs = [d for d in os.listdir(tracks_dir)
                   if os.path.isdir(os.path.join(tracks_dir, d))]
    else:
        prefixs = inputs.prefixs

    del inputs.prefixs
    assert type(prefixs) == list, 'prefixs must be a list'

    for prefix in prefixs:
        logger.info('\n\n Current mix: %s \n\n' % prefix)
        these_inputs = set_outdirs(inputs, prefix)
        tm = TracksForMatch(these_inputs)
        tm.save_ptcri(hb=this_dict.hb)
        MatchTracks(these_inputs)
        plt.close('all')



def set_outdirs(inputs, prefix):
    if not hasattr(inputs, 'tracks_dir'):
        print 'No tracks_dir set, using current location'
        inputs.tracks_dir = os.getcwd()

    new_inputs = deepcopy(inputs)
    new_inputs.prefix = prefix
    wkd = os.path.join(inputs.tracks_dir, new_inputs.prefix)
    if hasattr(inputs, 'plot_dir') and inputs.plot_dir == 'default':
        new_inputs.plot_dir = os.path.join(wkd, 'plots')
        fileIO.ensure_dir(new_inputs.plot_dir)

    if hasattr(inputs, 'outfile_dir') and inputs.outfile_dir == 'default':
        new_inputs.outfile_dir = os.path.join(wkd, 'match')
        fileIO.ensure_dir(new_inputs.outfile_dir)

    return new_inputs


def initialize_inputs():
    '''
    Load default inputs, the eep lists, and number of equally spaced points
    between eeps. Input file will overwrite these.
    '''
    input_dict =  {'eep_list': ['PMS_BEG', 'PMS_MIN',  'PMS_END', 'MS_BEG',
                                'MS_TMIN', 'MS_TO', 'SG_MAXL', 'RG_MINL',
                                'RG_BMP1', 'RG_BMP2', 'RG_TIP', 'HE_BEG',
                                'YCEN_0.550', 'YCEN_0.500', 'YCEN_0.400',
                                'YCEN_0.200', 'YCEN_0.100', 'YCEN_0.005',
                                'YCEN_0.000', 'TPAGB'],
                   'eep_lengths': [60, 60, 60, 199, 100, 100, 70, 370, 30, 400,
                                  10, 150, 100, 80, 100, 80, 80, 140, 200],
                   'eep_list_hb': ['HB_BEG', 'YCEN_0.550', 'YCEN_0.500',
                                  'YCEN_0.400', 'YCEN_0.200', 'YCEN_0.100',
                                  'YCEN_0.005', 'YCEN_0.000', 'AGB_LY1',
                                  'AGB_LY2'],
                   'eep_lengths_hb': [150, 100, 80, 100, 80, 80, 140, 100, 100],
                   'track_search_term': '*F7_*PMS',
                   'hbtrack_search_term':'*F7_*HB',
                   'from_p2m': False,
                   'hb_only': False,
                   'masses': None,
                   'do_interpolation': True,
                   'debug': False,
                   'hb': True}
    return input_dict


class f4_file(object):
    def __init__(self, filename):
        self.base, self.name = os.path.split(filename)
        self.load_f4_file(filename)

    def load_f4_file(self, filename):
        '''
        #S MODELL              ALTER        Q_BOT        QINTE    TINTE         B_SLX        B_SLNU
        #H MODELL                SLX        T_BOT         QHEL     THEL         B_SLY         B_SEG
        #C    CNO                SLY       RH_BOT          lgL     lgTe         B_SLC        HM_CHE
        Rg MODELL                 V1           V2           V3       V4            V5            V6        H            HE3          HE4          C            C13          N14          N15          O16          O17          O18          NE20         NE22         MG25         LI7          BE7          F19          MG24         MG26         NE21         NA23         AL26         AL27         SI28         Deut         ZH
        S       1  0.10000000000E+00  1.000000000  0.000000000   7.5711  0.000000E+00  0.000000E+00 7.000000E-01 4.400000E-05 2.800000E-01 3.610000E-03 4.440000E-05 1.200000E-03 4.740000E-06 1.080000E-02 4.190000E-06 2.440000E-05 5.730000E-04 7.700000E-05 1.840000E-04 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00
        H       0  0.10239192793E+01  0.000000000  0.000000000   0.0000  0.000000E+00  0.000000E+00 7.000000E-01 4.400000E-05 2.800000E-01 3.610000E-03 4.440000E-05 1.200000E-03 4.740000E-06 1.080000E-02 4.190000E-06 2.440000E-05 5.730000E-04 7.700000E-05 1.840000E-04 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00
        C      -2  0.00000000000E+00  0.000000000  6.112131886   4.6639  0.000000E+00  0.000000E+00 7.000000E-01 4.400000E-05 2.800000E-01 3.610000E-03 4.440000E-05 1.200000E-03 4.740000E-06 1.080000E-02 4.190000E-06 2.440000E-05 5.730000E-04 7.700000E-05 1.840000E-04 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00
        S       2  0.10000000000E+00  1.000000000  0.000000000   7.5705  0.000000E+00  0.000000E+00 7.000000E-01 4.400000E-05 2.800000E-01 3.610000E-03 4.440000E-05 1.200000E-03 4.740000E-06 1.080000E-02 4.190000E-06 2.440000E-05 5.730000E-04 7.700000E-05 1.840000E-04 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00
        H       0  0.10018991298E+01  0.000000000  0.000000000   0.0000  0.000000E+00  0.000000E+00 7.000000E-01 4.400000E-05 2.800000E-01 3.610000E-03 4.440000E-05 1.200000E-03 4.740000E-06 1.080000E-02 4.190000E-06 2.440000E-05 5.730000E-04 7.700000E-05 1.840000E-04 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00
        C      -2  0.00000000000E+00  0.000000000  6.111904543   4.6636  0.000000E+00  0.000000E+00 7.000000E-01 4.400000E-05 2.800000E-01 3.610000E-03 4.440000E-05 1.200000E-03 4.740000E-06 1.080000E-02 4.190000E-06 2.440000E-05 5.730000E-04 7.700000E-05 1.840000E-04 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00


        F4 file is in blocks of 3 rows for each model
        Row 1:  SURFACE  (S)
        Row 2:  HBURNING (H)
        Row 3:  CENTRE   (C)

        For each block the columns are:
        1: the region (S,H,C)

        2S:     MODELL  = the model number 
        2H:     allways = 0 not defined
        2C:     CNO burning still not defined

        3S:   ALTER  age
        3H:     SLX  LX/L_tot_surf
        3C:     SLY  LY/L_tot_surf

        4S:     Q_BOT     m/Mtot  at the bottom of the envelope convective region (including envelope overshoot)
        4H:     T_BOT     log10(T)
        4C:    RH_BOT     log10(rho)

        5S:    QINTE     m/Mtot where H=0.5*Hsurf as in F7
        5H:     QHEL     max(m/Mtot where H=0.)   as in F7
        5C:      lgL     Surface total luminosity as in F7  (L_TOT)

        6S:   TINTE      log10(T) at QINTE
        6H:    THEL      log10(T) at QHEL
        6C:    lgTe      surface Te (log10)

        7S:   B_SLX      L_H/L_tot_surf at the bottom of the conv envelope
        7H:   B_SLY      same for He
        7C:   B_SLC      same for Carbon

        7S:   B_SLNU     same for neutrinos
        7H:    B_SEG     same for gravitational energy (L_GRAV/L_TOT)
        7C:   HM_CHE     min time step size in chemistry routine

        8-end_S:  composition as indicated, H HE3 etc.. at the surface
        8-end_H:  composition as indicated, H HE3 etc.. at the H zone
        8-end_C:  composition as indicated, H HE3 etc.. at the C zone
        '''
        '/Users/phil/research/BRratio/models/model_grid/PH_COV0.5_ENV0.50_Z0.01_Y0.2663/PH_COV0.5_ENV0.50_Z0.01_Y0.2663/Z0.01Y0.2663OUTA1.74_F4_M5.00'
        #import copy
        #data = fileIO.readfile(filename, col_key_line=3)
        self.surface = fileIO.readfile(filename, col_key_line=3)[::3]
        self.hburning = fileIO.readfile(filename, col_key_line=3)[1::3]
        self.center = fileIO.readfile(filename, col_key_line=3)[2::3]
        self.surface.dtype.names = tuple('Surface MODE ALTER Q_BOT QINTE TINTE B_SLX B_SLNU'.split()) + self.surface.dtype.names[8:]
        self.hburning.dtype.names = tuple('Hburning MODE SLX T_BOT QHEL THEL B_SLY B_SEG'.split()) + self.hburning.dtype.names[8:]
        self.center.dtype.names = tuple('Center CNO SLY RH_BOT LOG_L LOG_TE B_SLC HM_CHE'.split()) + self.center.dtype.names[8:]
        

def verify_ptcri_file(ptcri_file, track_files):
    ptcri =  critical_point.load_ptcri(ptcri_file)
    tracks = [Track(t, min_lage=0) for t in track_files]
    for track in tracks:
        modes = ptcri.data_dict['M%.3f' % track.mass]
        modes = ptcri.data_dict['M%.3f' % track.mass]
        print track.mass
        mmodes = modes[modes > 0]
        print track.data.MODE[mmodes-2]
        print np.sort(np.concatenate([np.nonzero(track.data.MODE == m)[0]
                                        for m in mmodes]))

if __name__ == '__main__':
    inputs = fileIO.input_file(sys.argv[1], default_dict=initialize_inputs())
    # if prefix not prefixs, set the location of plots if given default.
    if hasattr(inputs, 'prefix'):
        inputs = set_outdirs(inputs, inputs.prefix)

    logfile = sys.argv[1].replace('inp', 'log')
    fh = logging.FileHandler(logfile)
    fmt = '%(asctime)s - %(levelname)s - %(module)s: %(lineno)d - %(message)s'
    formatter = logging.Formatter(fmt)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    import pdb
    pdb.set_trace()

    if hasattr(inputs, 'prefixs'):
        do_entire_set(inputs)
    else:
        tm = TracksForMatch(inputs)
        tm.save_ptcri(hb=inputs.hb)
        MatchTracks(inputs)
