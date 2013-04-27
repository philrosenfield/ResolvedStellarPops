from copy import deepcopy
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splprep
import fileIO
import math_utils
import graphics.GraphicsUtils as rspg
import logging
import pprint
logger = logging.getLogger()


def quick_color_em(tracks_base, prefix, photsys='UVbright', search_term='*F7_*PMS'):
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
    def add_comments_to_header(tracks_base, prefix, search_term):
        '''
        insert a # at every line
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


    def color_tracks(tracks_base, prefix, cmd):
        tracks = os.path.join(tracks_base, prefix)
        track_names = fileIO.get_files(tracks, search_term)

        for name in track_names:
            z = float(name.split('Z')[1].split('_Y')[0])
            os.system(cmd % (name, z))

    cmd = '/Users/phil/research/Italy/fromHR2mags/fromHR2mags %s ' % photsys
    # this is set for .PMS and .PMS.HB tracks
    cmd += '%s 5 6 2 %.3f'
    add_comments_to_header(tracks_base, prefix, search_term)
    search_term += '.dat'
    color_tracks(tracks_base, prefix, cmd)


class Track(object):
    def __init__(self, filename, ptcri=None, min_lage=0.2, cut_long=False):
        (self.base, self.name) = os.path.split(filename)
        self.load_track(filename, min_lage=min_lage, cut_long=cut_long)
        self.filename_info()
        self.mass = self.data.MASS[0]
        self.ptcri = ptcri

    def calc_Mbol(self):
        Mbol = 4.77 - 2.5 * self.data.LOG_L
        self.Mbol = Mbol
        return Mbol
    
    def calc_logg(self):
        logg = -10.616 + np.log10(self.mass) + 4.0 * self.data.LOG_TE - self.data.LOG_L
        self.logg = logg
        return logg

    def filename_info(self):
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
        Bressan's save_isoc_set
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

    def load_track(self, filename, min_lage=0.2, cut_long=True):
        '''
        reads PMS file into a record array. Stores header as string self.header
        '''
        with open(filename, 'r') as f:
            lines = f.readlines()

        begin_track, = [i for i, l in enumerate(lines) if 'BEGIN TRACK' in l]
        self.header = lines[:begin_track]
        col_keys = lines[begin_track + 1].replace('#', '').strip().split()
        begin_track_skip = 2

        if 'information' in lines[begin_track + 2]:
            col_keys = self.add_to_col_keys(col_keys, lines[begin_track + 2])
            begin_track_skip += 1

        # dtype = [(c, 'd') for c in col_keys]
        try:
            data = np.genfromtxt(filename,
                                 skiprows=begin_track + begin_track_skip,
                                 names=col_keys)
        except ValueError:
            data = np.genfromtxt(filename,
                                 skiprows=begin_track + begin_track_skip,
                                 names=col_keys, skip_footer=2,
                                 invalid_raise=False)

        # cut non-physical part of the model
        # NOTE it should be >= but sometimes Sandro's PMS_BEG
        #      will be one model number too soon for that.
        ainds, = np.nonzero(data['AGE'] > min_lage)
        data = data[ainds]
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
        #to_write = [np.column_stack(self.data[c]) for c in columns_list]
        #to_write = np.squeeze(np.array(to_write).T)
        with open(filename, 'w') as f:
            f.writelines(self.header)
            f.write('# %s \n' % ' '.join(self.col_keys))
            np.savetxt(f, self.data, dtype=self.data.dtype)
        return filename

    def write_trilegal_isotrack_ptcri(self, age, logl, logte, new_eep_dict,
                                      outfile=None):
        '''
        in dev... not finished.
        '''
        if self.mass >= 1.65:
            extra = 'INT2'
        else:
            return

        if outfile is None:
            outfile = 'ptcri_%s_Z%.4f_Y%.3f.dat.%s' % \
                      (os.path.split(self.base)[1], self.Z, self.Y, extra)

        with open(outfile, 'a') as f:
            for i in range(len(age)):
                extra = ''
                if i in new_eep_dict.values():
                    key, = [k for (k, v) in new_eep_dict.items() if v == i]
                    if key == 'PMS_BEG':
                        age[i] = 0.
                        extra = ' M=%.6f PMS_BEG 0 \n' % self.mass
                    else:
                        extra += ' %s %i \n' % (key, i)
                else:
                    extra += ' %i \n' % i
                isoc = '%.12e %.5f %.5f' % (age[i], logl[i], logte[i])

                f.write(isoc+extra)


class DefineEeps(object):
    def __init__(self):
        self.setup_eep_info()

    def setup_eep_info(self):
        self.eep_info = {'MS_TMIN_XCEN': [],
                         'SG_MAXL_XCEN': [],
                         'MS_TO_parametric': [],
                         'MS_BUSTED': [],
                         'ms_tmin_xcen': {'S12D_NS_Z0.0001_Y0.249': 1.2,
                                          'S12D_NS_Z0.0002_Y0.249': 1.15,
                                          'S12D_NS_Z0.0005_Y0.249': 1.15,
                                          'S12D_NS_Z0.001_Y0.25': 1.10,
                                          'S12D_NS_Z0.002_Y0.252': 1.15,
                                          'S12D_NS_Z0.004_Y0.256': 1.15,
                                          'S12D_NS_Z0.006_Y0.259': 1.15,
                                          'S12D_NS_Z0.008_Y0.263': 1.20,
                                          'S12D_NS_Z0.014_Y0.273': 1.20,
                                          'S12D_NS_Z0.017_Y0.279': 1.20,
                                          'S12D_NS_Z0.01_Y0.267': 1.20,
                                          'S12D_NS_Z0.02_Y0.284': 1.20,
                                          'S12D_NS_Z0.03_Y0.302': 1.20,
                                          'S12D_NS_Z0.04_Y0.321': 1.15,
                                          'S12D_NS_Z0.05_Y0.339': 1.10,
                                          'S12D_NS_Z0.06_Y0.356': 1.10},
                        'ms_tmin_byhand': {'S12D_NS_Z0.0001_Y0.249': {},
                                           'S12D_NS_Z0.0002_Y0.249':{},
                                           'S12D_NS_Z0.0005_Y0.249':{},
                                           'S12D_NS_Z0.001_Y0.25': {},
                                           'S12D_NS_Z0.002_Y0.252': {},
                                           'S12D_NS_Z0.004_Y0.256': {},
                                           'S12D_NS_Z0.006_Y0.259': {1.1: 1436},
                                           'S12D_NS_Z0.008_Y0.263': {1.1: 1452, 1.15: 1413},
                                           'S12D_NS_Z0.014_Y0.273': {1.1: 1408, 1.15: 1412},
                                           'S12D_NS_Z0.017_Y0.279': {1.1: 1443, 1.15: 1365},
                                           'S12D_NS_Z0.01_Y0.267': {1.1: 1380, 1.15: 1421},
                                           'S12D_NS_Z0.02_Y0.284': {1.1: 1544, 1.15: 1570},
                                           'S12D_NS_Z0.03_Y0.302': {1.1: 1525, 1.15: 1460},
                                           'S12D_NS_Z0.04_Y0.321': {1.1: 1570, 1.15: 1458},
                                           'S12D_NS_Z0.05_Y0.339': {},
                                           'S12D_NS_Z0.06_Y0.356': {1.05: 1436}}}
    
    def define_eep_stages(self, track, hb=False, plot_dir=None,
                          diag_plot=True):
        '''
        must define the stages here if not using Sandro's defaults.
        * denotes stages defined here.

        1 MS_BEG  Starting of the central H-burning phase
        2 MS_TMIN* First Minimum in Teff for high-mass or Xc=0.30 for low-mass
        stars
        3 MS_TO*   Maximum in Teff along the Main Sequence - TURN OFF POINT
        4 SG_MAXL*   Maximum in logL for high-mass or Xc=0.0 for low-mass stars
        5 RG_BASE Minimum in logL for high-mass or Base of the RGB for low-mass
        stars
        6 RG_BMP1 The maximum luminosity during the RGB Bump
        7 RG_BMP2 The minimum luminosity during the RGB Bump
        8 RG_TIP  Tip of the RGB

        Skipping BaSTi's (only 10 points):
        x Start quiescent central He-burning phase

        9 YCEN_0.55* Central abundance of He equal to 0.55
        10 YCEN_0.50* Central abundance of He equal to 0.50
        11 YCEN_0.40* Central abundance of He equal to 0.40
        12 YCEN_0.20* Central abundance of He equal to 0.20
        13 YCEN_0.10* Central abundance of He equal to 0.10
        14 YCEN_0.00* Central abundance of He equal to 0.00
        15 C_BUR Starting of the central C-burning phase

        Not yet implemented, no TPAGB tracks decided:
        x When the energy produced by the CNO cycle is larger than that
        provided by the He burning during the AGB (Lcno > L3alpha)
        x The maximum luminosity before the first Thermal Pulse
        x The AGB termination
        '''
        if hb is True:
            logger.info('\n\n       HB Current Mass: %.3f' % track.hbmass)
        else:
            logger.info('\n\n          Current Mass: %.3f' % track.mass)
        ptcri = self.ptcri

        if hb is True:
            default_list = ['HB_BEG', 'YCEN_0.500', 'YCEN_0.400', 'YCEN_0.200',
                            'YCEN_0.100', 'YCEN_0.005', 'AGB_LY1', 'AGB_LY2']
            eep_list = ptcri.please_define_hb

            assert default_list == eep_list, \
                'Can not define all HB EEPs. Please check lists'

            self.add_hb_beg(track)
            self.hb_eeps(track)
            self.add_agb_eeps(track, diag_plot=diag_plot, plot_dir=plot_dir)
            return

        default_list = ['MS_TMIN', 'MS_TO', 'SG_MAXL', 'RG_MINL', 'HE_BEG',
                        'YCEN_0.550', 'YCEN_0.500', 'YCEN_0.400', 'YCEN_0.200',
                        'YCEN_0.100', 'YCEN_0.000']

        eep_list = ptcri.please_define

        assert default_list == eep_list, \
            'Can not define all EEPs. Please check lists'

        self.add_ms_eeps(track)
        # even though ms_tmin comes first, need to bracket with ms_to

        ims_to = self.ptcri.iptcri[self.ptcri.get_ptcri_name('MS_TO',
                                                             sandro=False)]

        if ims_to == 0:
            # should now make sure all other eeps are 0.
            [self.add_eep(cp, 0) for cp in default_list[2:]]
        else:
            self.add_min_l_eep(track)
            self.add_max_l_eep(track)
            # RG_TIP is from Sandro
            # do the YCEN values first, because HE_BEG uses first YCEN.
            self.add_cen_eeps(track)
            self.add_quiesscent_he_eep(track, 'YCEN_0.550')
            #self.add_cburn_eep()

        assert not False in (np.diff(np.nonzero(self.ptcri.iptcri)[0]) >= 0), \
            'EEPs are not monotonically increasing. M=%.3f' % track.mass

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

    def add_cburn_eep(self, track):
        '''
        Just takes Sandro's, will need to expand here for TPAGB...
        '''
        eep_name = 'C_BUR'
        c_bur = self.ptcri.sandros_dict[eep_name]
        self.add_eep(eep_name, c_bur)

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
        inds = self.ptcri.inds_between_ptcris('RG_BMP2', ycen1, sandro=False)

        if len(inds) == 0:
            print self.ptcri.iptcri
            print 'no start HEB!!!!', track.mass, track.Z
            return

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
        print 'he beg', he_beg
        eep_name = 'HE_BEG'
        self.add_eep(eep_name, he_beg)

    def add_cen_eeps(self, track, xcen_eeps=False, xcen=False, cens=None,
                     hb=False):
        '''
        Add YCEN_[fraction] eeps, if YCEN=fraction found to 0.01, will add 0 as
        the iptrcri. (0.01 is hard coded)
        list of YCEN_[fraction] can be supplied by cens= otherwise taken from
        self.ptcri.please_define
        '''
        if xcen:
            fcol = 'PMS_BEG'
            col = 'XCEN'
        else:
            fcol = 'RG_BASE'
            col = 'YCEN'

        if xcen_eeps is False:
            fcol = 'RG_BMP2'

        if hb is False:
            inds = np.arange(self.ptcri.iptcri[self.ptcri.get_ptcri_name(fcol,
                                               sandro=xcen_eeps)],
                             len(track.data[col]))
        else:
            inds = np.arange(len(track.data[col]))

        if cens is None:
            # use undefined central values instead of given list.
            cens = [i for i in self.ptcri.please_define if i.startswith(col)]
            # e.g., YCEN_0.50
            cens = [float(cen.split('_')[-1]) for cen in cens]

        for cen in cens:
            ind, dif = math_utils.closest_match(cen, track.data[col][inds])
            icen = inds[0] + ind
            # some tolerance for a good match.
            if dif > 0.01:
                icen = 0
            self.add_eep('%s_%.3f' % (col, cen), icen, hb=hb)

    def hb_eeps(self, track, cens=None):
        '''
        '''
        self.add_hb_beg(track)
        if cens is None:
            cens = [0.5, 0.4, 0.2, 0.1, 0.005]

        self.add_cen_eeps(track, cens=cens, hb=True)

    def add_hb_beg(self, track):
        # this is just the first line of the track with age > 0.2 yr.
        # it could be snipped in the load_track method because it's
        # unphysical but to be clear, I'm keeping it here too.
        ainds, = np.nonzero(track.data['AGE'] > 0.2)
        hb_beg = ainds[0]
        eep_name = 'HB_BEG'
        self.add_eep(eep_name, hb_beg, hb=True)

    def add_agb_eeps(self, track, diag_plot=True, plot_dir=None):
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

    def add_ms_eeps(self, track):
        '''
        Adds  MS_TMIN and MS_TO.
        MS_TO: This is the MAX Teff between MS_BEG and MS_TMIN.

        Note: MS_TMIN could be XCEN = 0.3 if no actual MS_TMIN (low masses)
              (0.3 is hard coded).

        If no MS_TMIN, assumes no MS_TO coming after it.
        '''
        def use_xcen(track, val=0.3, error_mass=1.2,  ex_inds=None):
            '''
            method to use XCEN == val to define ms_tmin
            '''
            if track.mass >= error_mass:
                logger.error('Using XCEN=0.3 for T_MIN: M=%.3f' % track.mass)

            inds = track.ptcri.inds_between_ptcris('MS_BEG', 'RG_BMP1',
                                                   sandro=True)
            if ex_inds is not None:
                inds = list(set(ex_inds) & set(inds))

            if len(inds) == 0:
                ms_tmin = 0
            else:
                ind, dif = math_utils.closest_match(val, track.data.XCEN[inds])
                ms_tmin = inds[ind]
                if ind == -1:
                    logger.error('no ms_tmin!')
                    ms_tmin = 0
                if dif > 0.01:
                    logger.error('MS_TMIN: bad match for xcen = %.1f.' % val)
            self.eep_info['MS_TMIN_XCEN'].append(track.mass)
            return ms_tmin
 
        byhand_dict = self.eep_info['ms_tmin_byhand']
        if len(byhand_dict[self.prefix]) != 0 and byhand_dict[self.prefix].has_key(track.mass):
            print 'ms_tmin by hand. %.4f %.3f' % (track.Z, track.mass) 
            ms_tmin = byhand_dict[self.prefix][track.mass]
        else:
            inds = track.ptcri.inds_between_ptcris('MS_BEG', 'POINT_C', sandro=True)
            if len(inds) == 0:
                ms_tmin = 0
            else:
                xdata = track.data.LOG_TE[inds]
                tmin_ind = np.argmin(xdata)
                ms_tmin = inds[tmin_ind]

                if track.mass < self.eep_info['ms_tmin_xcen'][self.prefix]:
                    # use XCEN == 0.3
                    tmin_ind = np.argmin(np.abs(track.data.XCEN[inds] - 0.3))
                    # not used... but a QC:
                    dif = np.abs(track.data.XCEN[inds[tmin_ind]] - 0.3)
                elif tmin_ind > 1e9:  # I don't think this is useful! 
                    # find the arg min of teff between these points and get
                    # something very close to MS_BEG probably means the MS_BEG
                    # is at a lower Teff than Tmin.
                    mode = np.arange(len(xdata))
                    tckp, u = splprep([mode, xdata], s=0, k=3, nest=-1)
                    # if not using something like model number instead of log l,
                    # the tmin will get hidden by data with t < tmin but different
                    # log l, this is only a problem for very low Z.
                    arb_arr = np.arange(0, 1, 1e-2)
                    xnew, ynew = splev(arb_arr, tckp)
                    # second derivative, bitches.
                    ddxnew, ddynew = splev(arb_arr, tckp, der=2)
                    # diff displaces the index by one. 
                    aind = np.argmin(np.diff(ddynew/ddxnew)) + 1
                    tmin_ind, dif = math_utils.closest_match(ynew[aind], xdata)

                ms_tmin = inds[tmin_ind]
        self.add_eep('MS_TMIN', ms_tmin)

        if ms_tmin == 0:
            ms_to = 0
        else:
            pf_kw = {'max': True, 'sandro': False, 'more_than_one': 'max of max', 
                     'parametric_interp': False}
            ms_to = self.peak_finder(track, 'LOG_TE', 'MS_TMIN', 'RG_BMP1',
                                     **pf_kw)
            if ms_to == -1:
                pf_kw['parametric_interp'] = True
                ms_to = self.peak_finder(track, 'LOG_TE', 'MS_TMIN', 'RG_BMP1',
                                         **pf_kw)
            if ms_to == -1 or ms_to == ms_tmin:
                logger.warning(
                    'Finding MS_TO (%i) by inflections in the HRD slope. M=%.3f'
                    % (ms_to, track.mass))

                inds = self.ptcri.inds_between_ptcris('MS_TMIN', 'RG_BMP1',
                                                      sandro=False)
                if ex_inds is not None:
                    inds = list(set(ex_inds) & set(inds))
                non_dupes = self.remove_dupes(track.data.LOG_TE[inds],
                                              track.data.LOG_L[inds],
                                              'lixo', just_two=True)
                tckp, u = splprep([track.data.LOG_TE[inds][non_dupes],
                                   track.data.LOG_L[inds][non_dupes]], s=0)
                tenew, lnew = splev(np.linspace(0, 1, 100), tckp)
                # subtract off a straight line fit to get more contrast...
                slope, intercept = np.polyfit(tenew, lnew, 1)
                highc_lnew = lnew - (slope * tenew + intercept)
                peak_dict = math_utils.find_peaks(highc_lnew)
                # if more than one max is found, take the max of the maxes.
                imax = peak_dict['maxima_locations']
                if len(imax) == 0:
                    logger.warning('inflections in HRD slope did not work. Forcing ms_tmin to be at xcen val.')
                    ms_tmin = use_xcen(track, ex_inds=ex_inds)
                    pf_kw['parametric_interp'] = False
                    ms_to = self.peak_finder(track, 'LOG_TE', 'MS_TMIN', 'RG_BMP1',
                                             **pf_kw)
                    if ms_to == -1 or ms_tmin == ms_to:
                        logger.error('Can not find ms_to for %s. Skipping.' % track.name)
                        ms_to = 0
                        self.add_eep('MS_TMIN', 0)
                        self.eep_info['MS_BUSTED'].append(track.mass)
                else:
                    # this index refers to interpolation
                    almost_ind = imax[np.argmax(highc_lnew[imax])]
                    # find equiv point on track grid
                    ind, diff = math_utils.closest_match2d(almost_ind,
                                                           track.data.LOG_TE[inds],
                                                           track.data.LOG_L[inds],
                                                           tenew, lnew)
                    ms_to = inds[ind]

            self.eep_info['MS_TO_parametric'].append(pf_kw['parametric_interp'])
            
        self.add_eep('MS_TO', ms_to)
        return
    
    def add_min_l_eep(self, track):
        '''
        The MIN L before the RGB for high mass or the base of the
        RGB for low mass.
        '''
        min_l = self.peak_finder(track, 'LOG_L', 'MS_TO', 'RG_BMP1',
                                 sandro=False, more_than_one='last')

        if min_l == -1:
            # no max found, need to get base of RGB.
            # this will be the deflection point in HRD space between
            # MS_TO and RG_BMP1.
            if track.mass > 1.20:
                logger.warning('Using base of RG for RG_MINL: M=%.3f' %
                               track.mass)
            inds = self.ptcri.inds_between_ptcris('MS_TO', 'RG_BMP1',
                                                  sandro=False)
            if inds[0] == 0:
                logger.error(
                    'RG_MINL finder will not work with no MS_TO (ie, MS_TO=0)')
            # interpolate...
            non_dupes = self.remove_dupes(track.data.LOG_TE[inds],
                                          track.data.LOG_L[inds],
                                          0, just_two=True)
            tckp, u = splprep([track.data.LOG_TE[inds][non_dupes],
                               track.data.LOG_L[inds][non_dupes]], s=0)
            tenew, lnew = splev(np.linspace(0, 1, 200), tckp)
            slope, intercept = np.polyfit(tenew, lnew, 1)
            highc_lnew = lnew - (slope * tenew + intercept)
            peak_dict = math_utils.find_peaks(highc_lnew)
            if len(peak_dict['minima_locations']) != 0:
                # if more than one max is found, take the max of the maxes.
                imin = peak_dict['minima_locations']
                # this index refers to interpolation    
                almost_ind = imin[np.argmin(highc_lnew[imin])]
            else:
                # sometimes (very low mass) there is not much a discernable
                # minimum, this forces a discontinuity, and finds it.
                almost_ind = np.argmin(lnew / (slope * tenew + intercept))

            # find equiv point on track grid
            ind, diff = math_utils.closest_match2d(almost_ind,
                                                   track.data.LOG_TE[inds],
                                                   track.data.LOG_L[inds], 
                                                   tenew, lnew)
            min_l = inds[ind]

        if np.round(track.data.XCEN[min_l], 4) > 0:
            logger.error('XCEN at RG_MINL should be zero if low mass. %.4f' %
                           track.data.XCEN[min_l])
        self.add_eep('RG_MINL', min_l)

    def add_max_l_eep(self, track):
        '''
        Adds SG_MAXL between MS_TO and RG_BASE. For low mass, there will be no
        SG_MAXL and will add XCEN = 0.0 (0.0 is hard coded)
        '''
        max_l = self.peak_finder(track, 'LOG_L', 'MS_TO', 'RG_MINL', max=True,
                                 sandro=False, more_than_one='last',
                                 parametric_interp=False)

        if track.mass < self.eep_info['ms_tmin_xcen'][self.prefix]:
            ex_inds, = np.nonzero(track.data.XCEN == 0.)
            inds = self.ptcri.inds_between_ptcris('MS_TO', 'RG_MINL',
                                                  sandro=False)
            inds = list(set(ex_inds) & set(inds))
            if len(inds) == 0:
                logger.error(
                    'XCEN=0.0 happens after RG_MINL, RG_MINL is too early.')
                max_l = 0
            else:
                max_l = inds[0]
                self.eep_info['SG_MAXL_XCEN'].append(track.mass)
            msto = self.ptcri.iptcri[self.ptcri.get_ptcri_name('MS_TO',
                                                               sandro=False)]
            if max_l == msto:
                logger.error('SG_MAXL is at MS_TO!')
                logger.error('XCEN at MS_TO (%i): %.3f' %
                               (msto, track.data.XCEN[msto]))

        self.add_eep('SG_MAXL', max_l)

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
            inds = self.ptcri.inds_between_ptcris('MS_BEG', 'POINT_B',
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
                    ind_tol=3, dif_tol=0.01, extra_inds=None,
                    parametric_interp=True):
        '''
        finds some peaks! Usually interpolates and calls a basic diff finder,
        though some higher order derivs of the interpolation are sometimes used.
        '''
        
        # slice the array
        inds = self.ptcri.inds_between_ptcris(eep1, eep2, sandro=sandro)
        # burn in
        inds = inds[5:]

        if extra_inds is not None:
            inds = list(set(inds) & set(extra_inds))

        if len(inds) < ind_tol:
            # sometimes there are not enough inds to interpolate
            if track.mass > 0.5:
                logger.warning('Peak finder %s-%s M%.3f: less than %i points = %i. Skipping.'
                               % (eep1, eep2, track.mass, ind_tol, len(inds)))
            return 0

        if parametric_interp is True:
            # use age, so logl(age), logte(age) for parametric interpolation
            tckp, step_size, non_dupes = self.interpolate_te_l_age(track, inds)
            tenew, lnew, agenew = splev(np.arange(0, 1, step_size), tckp)
            dtnew, dlnew, dagenew = splev(np.arange(0, 1, step_size), tckp, der=1)
            intp_col = lnew
            dydx = dtnew / dlnew
            if col == 'LOG_TE':
                intp_col = tenew
                dydx = dlnew / dtnew
        else:
            # interpolate logl, logte.
            cols = ['LOG_L', 'LOG_TE']
            ycol, = [a for a in cols if a != col]
            xdata = track.data[col][inds]
            ydata = track.data[ycol][inds]

            non_dupes = self.remove_dupes(xdata, ydata, 'lixo', just_two=True)
            xdata = xdata[non_dupes]
            ydata = ydata[non_dupes]
            k = 3
            if len(non_dupes) <= k:
                k = len(non_dupes) - 1
                logger.warning('only %i indices to fit... %s-%s' % (len(non_dupes), eep1, eep2))
                logger.warning('new spline_level %i' % k)

            tckp, u = splprep([xdata, ydata], s=0, k=k, nest=-1)

            ave_data_step = np.round(np.mean(np.abs(np.diff(xdata))), 4)
            min_step = 1e-4
            step_size = np.max([ave_data_step, min_step])
            xnew, ynew = splev(np.arange(0, 1, step_size), tckp)
            dxnew, dynew = splev(np.arange(0, 1, step_size), tckp, der=1)
            intp_col = xnew
            dydx = dynew / dxnew

        # find the peaks!
        peak_dict = math_utils.find_peaks(intp_col)

        if max is True:
            if peak_dict['maxima_number'] > 0:
                # if more than one max is found, take the max of the maxes.
                imax = peak_dict['maxima_locations']
                # this index refers to interpolation
                if more_than_one == 'max of max':
                    almost_ind = imax[np.argmax(intp_col[imax])]
                elif more_than_one == 'last':
                    almost_ind = imax[-1]
                else:
                    logger.error('not ready yet...')
            else:
                # no maxs found.
                if mess_err is not None:
                    logger.error(mess_err)
                return -1

        else:
            if peak_dict['minima_number'] > 0:
                if more_than_one == 'first':
                    almost_ind = np.min(peak_dict['minima_locations'])
                elif more_than_one == 'min of min':
                    almost_ind = np.argmax(dydx)
                    print 'a', almost_ind
                elif more_than_one == 'last':
                    almost_ind = peak_dict['minima_locations'][-1]
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
                             diag_plot=True):
        '''
        iptcri is the critical point index rel to track.data
        mptcri is the model number of the critical point
        '''
        assert filename is not None or ptcri is not None, \
            'Must supply either a ptcri file or object'

        if ptcri is None:
            ptcri = critical_point(filename, eep_obj=eep_obj)

        self.ptcri = ptcri

        # Sandro's definitions.
        if hb is True:
            self.ptcri.iptcri = []
            please_define = ptcri.please_define_hb
        else:
            mptcri = ptcri.data_dict['M%.3f' % track.mass]
            self.ptcri.iptcri = \
                np.sort(np.concatenate([np.nonzero(track.data.MODE == m)[0]
                                        for m in mptcri]))

            # sandro's points, just for comparison.
            self.ptcri.sptcri = self.ptcri.iptcri
            please_define = ptcri.please_define

        if len(please_define) > 0:
            if hasattr(self.ptcri, 'eep'):
                # already loaded eep
                eep_obj = self.ptcri.eep
                if hb is True:
                    space_for_new = np.zeros(len(eep_obj.eep_list_hb),
                                             dtype='int')
                else:
                    space_for_new = np.zeros(len(eep_obj.eep_list), dtype='int')

                self.ptcri.iptcri = space_for_new

            if hb is False:
                # Get the values that we won't be replacing.
                # For HB, there are no values from Sandro.
                for eep_name in self.ptcri.sandro_eeps:
                    if eep_name in self.ptcri.eep.eep_list:
                        # model number in track
                        mode_num = mptcri[self.ptcri.sandros_dict[eep_name]]
                        # which eep index would this replace
                        ieep = self.ptcri.eep.eep_list.index(eep_name)
                        if mode_num == 0 or mode_num > len(track.data.LOG_L):
                            # this track doesn't have this eep either because
                            # it just doesn't get there, or it was too long, and
                            # cut before I got the track set.
                            iorig = 0
                        else:
                            # new ptcri
                            iorig, = np.nonzero(track.data.MODE == mode_num)
                            logger.debug('new ptcri %s %i' % (eep_name, iorig))
                        self.ptcri.iptcri[ieep] = iorig
                        
            track.ptcri = deepcopy(self.ptcri)

            self.define_eep_stages(track, hb=hb, plot_dir=plot_dir,
                                   diag_plot=diag_plot)
            track.ptcri = deepcopy(self.ptcri)

        else:
            self.ptcri.iptcri = ptcri.data_dict['M%.3f' % track.mass]
            
        assert ptcri.Z == track.Z, \
            'Zs do not match between track and ptcri file'

        assert ptcri.Y == track.Y, \
            'Ys do not match between track and ptcri file'

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

    def interpolate_te_l_age(self, track, inds, k=3, nest=-1, s=0., min_step=1e-4):
        '''
        a caller for scipy.optimize.splprep. Will also rid the array
        of duplicate values. Returns tckp, an input to splev.
        '''
        non_dupes = self.remove_dupes(track.data.LOG_TE[inds],
                                      track.data.LOG_L[inds],
                                      track.data.AGE[inds])

        if len(non_dupes) <= k:
            k = len(non_dupes) - 1
            logger.warning('only %i indices to fit...' % (len(non_dupes)))
            logger.warning('new spline_level %i' % k)

        tckp, u = splprep([track.data.LOG_TE[inds][non_dupes],
                           track.data.LOG_L[inds][non_dupes],
                           np.log10(track.data.AGE[inds][non_dupes])],
                          s=s, k=k, nest=nest)

        xdata = track.data.LOG_TE[inds][non_dupes]
        ave_data_step = np.round(np.mean(np.abs(np.diff(xdata))), 4)
        step_size = np.max([ave_data_step, min_step])

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
                   ax=None, inds=None, plt_kw={}, annotate=False, clean=True,
                   ainds=None, sandro=False, cmd=False, convert_mag_kw={},
                   xdata=None, ydata=None, hb=False, xnorm=False, ynorm=False):
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
                    import ResolvedStellarPops as rsp
                    photsys = convert_mag_kw['photsys']
                    dmod = convert_mag_kw.get('dmod', 0.)
                    Av = convert_mag_kw.get('Av', 0.)
                    Mag1 = track.data[xcol]
                    Mag2 = track.data[ycol]
                    mag1 = rsp.astronomy_utils.Mag2mag(Mag1, xcol, photsys,
                                                       Av=Av, dmod=dmod)
                    mag2 = rsp.astronomy_utils.Mag2mag(Mag2, ycol, photsys,
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
            inds = [i for i in inds if i > 0]      
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

    def check_ptcris(self, track, hb=False, plot_dir=None):
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
                     ['YCEN_0.100', 'YCEN_0.000', 'C_BUR']]
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

            ax = self.plot_track(track, 'LOG_TE', 'LOG_L', ax=ax, inds=all_inds,
                                 reverse_x=True, plt_kw=line_pltkw)
            ax = self.plot_track(track, 'LOG_TE', 'LOG_L', ax=ax, inds=inds,
                                 plt_kw=point_pltkw, annotate=True, ainds=inds,
                                 hb=hb)

            if hasattr(self, 'match_data'):
                logl = (4.77 - self.match_data.T[3]) / 2.5
                ax.plot(self.match_data.T[2], logl, lw=2, color='green')

            tmax, tmin = self.maxmin(track, 'LOG_TE', inds=inds)
            lmax, lmin = self.maxmin(track, 'LOG_L', inds=inds)

            if np.diff((tmin, tmax)) == 0:
                tmin -= 0.1
                tmax += 0.1

            if np.diff((lmin, lmax)) == 0:
                lmin -= 0.5
                lmax += 0.5

            offx = 0.05
            offy = 0.1
            ax.set_xlim(tmax + offx, tmin - offx)
            ax.set_ylim(lmin - offy, lmax + offy)
            #ax.set_xlim(goodlimx)
            #ax.set_ylim(goodlimy)
            ax.set_xlabel('$LOG\ TE$', fontsize=20)
            ax.set_ylabel('$LOG\ L$', fontsize=20)

        title = 'M = %.3f Z = %.4f Y = %.4f' % (track.mass, track.Z, track.Y)
        fig.suptitle(title, fontsize=20)
        if hb is True:
            extra = '_HB'
        else:
            extra = ''
        figname = 'ptcri_Z%g_Y%g_M%.3f%s.png' % (track.Z, track.Y, track.mass,
                                                 extra)
        if plot_dir is not None:
            figname = os.path.join(plot_dir, figname)
        plt.savefig(figname)
        plt.close()
        logger.info('wrote %s' % figname)

        if hb is False:
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

    def inds_between_ptcris(self, name1, name2, sandro=True):
        '''
        returns the indices from [name1, name2)
        this is iptcri, not mptcri
        they will be the same inds that can be used in Track.data
        '''
        if sandro is True:
            # this must be added in Tracks.load_critical_points!
            ptcri = self.sptcri
        else:
            ptcri = self.iptcri

        try:
            first = ptcri[self.get_ptcri_name(name1, sandro=sandro)]
        except IndexError:
            first = 0

        try:
            second = ptcri[self.get_ptcri_name(name2, sandro=sandro)]
        except IndexError:
            second = 0

        inds = np.arange(first, second)
        return inds

    def load_ptcri(self, filename, eep_obj=None):
        '''
        reads the ptcri*dat file. If there is an eep_obj, it will flag the
        missing eeps in the ptcri file and only read the eeps that match both
        the eep_list and the ptcri file.
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
        data = np.genfromtxt(filename, usecols=usecols, skip_header=begin + 2)
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
            please_define = [c for c in eep_obj.eep_list if c not in col_keys]
            self.eep = eep_obj
            self.key_dict = dict(zip(eep_obj.eep_list,
                                     range(len(eep_obj.eep_list))))
            self.please_define = please_define
            if eep_obj.eep_list_hb is not None:
                self.key_dict_hb = dict(zip(eep_obj.eep_list_hb,
                                        range(len(eep_obj.eep_list_hb))))
                # there is no mixture between Sandro's HB eeps since there
                # are no HB eeps in the ptcri files. Define them all here.
                self.please_define_hb = eep_obj.eep_list_hb
        else:
            self.please_define = []
            self.key_dict = self.sandros_dict

        
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
    def __init__(self, tracks_dir=None, prefix=None, ptcrifile_loc=None,
                 eep_list=None, eep_lengths=None, eep_list_hb=None,
                 eep_lengths_hb=None, hb=False, track_search_term='*F7_*PMS',
                 hbtrack_search_term='*F7_*HB', plot_dir=None, masses=None,
                 ptcri_file=None, **kwargs):

        if ptcrifile_loc is not None or ptcri_file is not None:
            self.load_ptcri_eep(prefix=prefix, ptcri_file=ptcri_file,
                                ptcrifile_loc=ptcrifile_loc,
                                eep_list=eep_list, eep_lengths=eep_lengths, 
                                eep_list_hb=eep_list_hb,
                                eep_lengths_hb=eep_lengths_hb)
        else:
            self.ptcri = None

        self.tracks_base = os.path.join(tracks_dir, prefix)
        self.load_tracks(track_search_term=track_search_term, masses=masses)
        if hb is True:
            self.load_tracks(track_search_term=hbtrack_search_term, hb=hb,
                             masses=masses)

    def load_ptcri_eep(self, prefix=None, ptcri_file=None, ptcrifile_loc=None,
                       eep_list=None, eep_lengths=None, eep_list_hb=None,
                       eep_lengths_hb=None, from_p2m=False):
        '''
        load the ptcri and eeps, simple call to the objects.
        '''
        self.ptcri = None
        self.eep = None
        if ptcri_file is not None:
            self.ptcri_file = ptcri_file
        else:
            self.prefix = prefix
            if from_p2m is True:
                search_term = 'p2m*%s*dat' % prefix
                self.ptcri_file, = fileIO.get_files(ptcrifile_loc, search_term)
                print 'reading ptcri from saved p2m file.'
            else:        
                search_term = 'pt*%s*dat' % prefix        
                self.ptcri_file, = fileIO.get_files(ptcrifile_loc, search_term)

        if eep_list is not None:
            eep_kw = {'eep_lengths': eep_lengths,
                      'eep_list_hb': eep_list_hb,
                      'eep_lengths_hb': eep_lengths_hb}
            self.eep = eep(eep_list, **eep_kw)

        self.ptcri = critical_point(self.ptcri_file, eep_obj=self.eep)
            
    def load_tracks(self, track_search_term='*F7_*PMS', hb=False, masses=None):
        '''
        loads tracks or hb tracks, can load subset if masses (list or float)
        is set.
        '''
        track_names = np.array(fileIO.get_files(self.tracks_base,
                               track_search_term))
        assert len(track_names) != 0, \
            'No tracks found: %s/%s' % (self.tracks_base, track_search_term)
        mass = map(float, [t.split('_')[-1].split('.P')[0].replace('M', '') for t in track_names])
        track_masses = np.argsort(mass)

        # only do a subset of masses
        if masses is not None:
            if type(masses) == float:
                masses = [masses]
            track_masses = [t for t in mass if t in masses]

        # ordered by mass
        track_str = 'track'
        mass_str = 'masses'
        if hb is True:
            track_str = 'hb%s'% track_str
            mass_str = 'hb%s' % mass_str
        self.__setattr__('%s_names' % track_str, track_names[track_masses])

        self.__setattr__('%ss' % track_str, [Track(track, ptcri=self.ptcri,
                                                   min_lage=0., cut_long=0)
                                             for track in self.track_names])

        self.__setattr__('%s' % mass_str,
                         np.round([t.mass for t in
                                   self.__getattribute__('%ss' % track_str)],
                                   3))

    def save_ptcri(self, filename=None):
        #assert hasattr(self, ptcri), 'need to have ptcri objects loaded'
        if filename is None:
            base, name = os.path.split(self.ptcri_file)
            filename = os.path.join(base, 'p2m_%s' % name)

        sorted_keys, inds = zip(*sorted(self.ptcri.key_dict.items(),
                                        key=lambda (k, v): (v, k)))

        header = '# critical points in F7 files defined by sandro, basti, and phil \n'
        header += '# i mass lixo %s fname \n' % (' '.join(sorted_keys))
        with open(filename, 'w') as f:
            f.write(header)
            linefmt = '%2i %.3f 0.0 %s %s \n'
            for i, track in enumerate(self.tracks):
                self.ptcri.please_define = []
                self.load_critical_points(track, eep_obj=self.eep,
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
        MS_BEG to RG_TIP, and RG_TIP to C_BUR.
        As it stands, one is trying to tell RGB features from AGB features.
        Likewise, there is such a small color difference between some of the
        different points that I'm not entire sure what I'm seeing.

        I see a sharp break in the RGB bump and RGB tip sequences.
        Are those visible in the isochrones?
        '''
        line_pltkw = {'color': 'black', 'alpha': 0.3}

        if one_plot is True:
            for t in tracks:
                all_inds, = np.nonzero(t.data.AGE > 0.2)
                ax = self.plot_track(t, xcol, ycol, ax=ax, inds=all_inds,
                                     plt_kw=line_pltkw, cmd=cmd,
                                     convert_mag_kw=convert_mag_kw, hb=hb)
            return ax

        ptcri_kw = {'sandro': sandro, 'hb': hb}

        if hb is False:
            plots = [['PMS_BEG', 'PMS_MIN', 'PMS_END', 'MS_BEG'],

                     ['MS_BEG', 'MS_TMIN', 'MS_TO', 'SG_MAXL', 'RG_MINL',
                      'RG_BMP1', 'RG_BMP2', 'RG_TIP'],

                     ['RG_TIP', 'HE_BEG', 'YCEN_0.550', 'YCEN_0.500',
                      'YCEN_0.400', 'YCEN_0.200', 'YCEN_0.100', 'YCEN_0.000',
                      'C_BUR']]

            fig_extra = ['pms', 'ms', 'rg']
        else:
            plots = [['HB_BEG', 'YCEN_0.500', 'YCEN_0.400', 'YCEN_0.200',
                     'YCEN_0.100', 'YCEN_0.005', 'AGB_LY1', 'AGB_LY2']]
            # overwriting kwargs!!
            ptcri_kw['sandro'] = False
            fig_extra = ['hb']


        xlims = np.array([])
        ylims = np.array([])
        for j in range(len(plots)):
            fig, ax = plt.subplots()
            if annotate is True:
                point_pltkw = {'marker': 'o', 'ls': '', 'alpha': 0.5}
                cols = rspg.discrete_colors(len(plots[j]), colormap='spectral')
                labs = [p.replace('_', '\_') for p in plots[j]]

            didit = 0
            xlimi = np.array([])
            ylimi = np.array([])
            for t in tracks:
                print t.name
                all_inds, = np.nonzero(t.data.AGE > 0.2)

                ainds = [t.ptcri.get_ptcri_name(cp, **ptcri_kw)
                         for cp in plots[j]]

                inds = t.ptcri.iptcri[ainds][np.nonzero(t.ptcri.iptcri[ainds])[0]]

                if np.sum(inds) == 0:
                    continue

                ax = self.plot_track(t, xcol, ycol, ax=ax, inds=all_inds,
                                     plt_kw=line_pltkw, cmd=cmd,
                                     convert_mag_kw=convert_mag_kw)

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
            figname = '%s_%s_%s_%s.png' % (self.prefix, xcol, ycol, fig_extra[j])

            if cmd is True:
                xlab = '%s-%s' % (xlab, ylab)

            ax.set_xlabel('$%s$' % xlab)
            ax.set_ylabel('$%s$' % ylab)

            if plot_dir is not None:
                figname = os.path.join(plot_dir, figname)
            plt.savefig(figname)
            logger.info('wrote %s' % figname)
            plt.close()
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


class MatchTracks(object):
    '''
    a simple check of the output from TracksForMatch. I want it to run on the
    same input file as TracksForMatch.
    '''
    def __init__(self, outfile_dir=None, eep_list=None, eep_lengths=None,
                 track_search_term='match_*dat', eep_list_hb=None,
                 eep_lengths_hb=None, prefix=None, **kwargs):

        self.tracks_base = outfile_dir
        self.prefix = prefix
        all_track_names = fileIO.get_files(self.tracks_base, track_search_term)
        self.hbtrack_names = [t for t in all_track_names if 'HB' in t]
        self.track_names = [t for t in all_track_names
                            if t not in self.hbtrack_names]
        self.tracks = [self._load_track(t) for t in self.track_names]
        self.hbtracks = [self._load_track(t) for t in self.hbtrack_names]

        self.eep_list = eep_list
        self.eep_lengths = eep_lengths
        self.eep_lengths_hb = eep_lengths_hb
        self.eep_list_hb = eep_list_hb
        self._plot_all_tracks(self.tracks, eep_list=eep_list,
                              eep_lengths=eep_lengths, plot_dir=outfile_dir)
        if self.eep_list_hb is not None:
            self._plot_all_tracks(self.hbtracks, eep_list=eep_list_hb,
                                  eep_lengths=eep_lengths_hb, 
                                  plot_dir=outfile_dir, extra='_HB')

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
                         plot_dir=None, extra=''):

        if eep_lengths is not None:
            eep_lengths = map(int, np.insert(np.cumsum(eep_lengths), 0, 1))
        line_pltkw = {'color': 'black', 'alpha': 0.3}
        point_pltkw = {'marker': 'o', 'ls': '', 'alpha': 0.5}
        cols = rspg.discrete_colors(len(eep_list), colormap='spectral')
        labs = [p.replace('_', '\_') for p in eep_list]

        fig, ax = plt.subplots()
        # fake lengend
        [ax.plot(9999, 9999, color=cols[i], label=labs[i], **point_pltkw)
         for i in range(len(eep_lengths))]

        [ax.plot(t.LOG_TE, t.LOG_L, **line_pltkw) for t in tracks]
        xlims = np.array([])
        ylims = np.array([])
        for t in tracks:
            for i in range(len(eep_lengths)):
                x = t.LOG_TE
                y = t.LOG_L
                ind = eep_lengths[i] - 1
                # print ind, labs[i], len(eep_lengths), i, len(t.LOG_TE)
                if (len(x) < ind):
                    continue
                ax.plot(x[ind], y[ind], color=cols[i], **point_pltkw)
                xlims = np.append(xlims, (np.min(x[ind]), np.max(x[ind])))
                ylims = np.append(ylims, (np.min(y[ind]), np.max(y[ind])))

        ax.set_xlim(np.max(xlims), np.min(xlims))
        ax.set_ylim(np.min(ylims), np.max(ylims))
        ax.legend(loc=0, numpoints=1, frameon=0)
        figname = 'match_%s%s.png' % (self.prefix, extra)
        if plot_dir is not None:
            figname = os.path.join(plot_dir, figname)
        plt.savefig(figname, dpi=300)


class TracksForMatch(TrackSet, DefineEeps, TrackDiag):
    def __init__(self, tracks_dir=None, prefix=None, ptcrifile_loc=None,
                 eep_list=None, eep_lengths=None, eep_list_hb=None,
                 eep_lengths_hb=None, hb=False, track_search_term='*F7_*PMS',
                 hbtrack_search_term='*F7_*HB', plot_dir=None,
                 outfile_dir=None, masses=None, diag_plot=None):

        TrackSet.__init__(self, tracks_dir=tracks_dir, prefix=prefix,
                          ptcrifile_loc=ptcrifile_loc, eep_list=eep_list,
                          eep_lengths=eep_lengths, eep_list_hb=eep_list_hb,
                          eep_lengths_hb=eep_lengths_hb, hb=hb, masses=masses,
                          track_search_term=track_search_term,
                          hbtrack_search_term=hbtrack_search_term,
                          plot_dir=plot_dir, outfile_dir=outfile_dir)
        DefineEeps.__init__(self)

        for track in self.tracks:
            # do the work! Assign eeps either from sandro, or eep_list and
            # make some diagnostic plots.
            track = self.load_critical_points(track, ptcri=self.ptcri,
                                              plot_dir=plot_dir,diag_plot=diag_plot)

            # make match output files.
            self.prepare_track(track, outfile_dir=outfile_dir)

            # make diagnostic plots
            self.check_ptcris(track, plot_dir=plot_dir)

        # make summary diagnostic plots
        self.plot_all_tracks(self.tracks, 'LOG_TE', 'LOG_L', sandro=False,
                             reverse_x=True, plot_dir=plot_dir)

        logger.info(pprint.pprint(self.eep_info))

        # do the same as above but for HB.
        if hb is True:
            self.hbtracks = []
            self.hbtrack_names = fileIO.get_files(self.tracks_base,
                                                  hbtrack_search_term)
            for track in self.hbtrack_names:
                track = self.load_critical_points(track, ptcri=self.ptcri,
                                                  hb=hb, plot_dir=plot_dir)
                self.hbtracks.append(track)
                self.prepare_track(track, outfile_dir=outfile_dir, hb=hb)
                self.check_ptcris(track, hb=hb, plot_dir=plot_dir)

            self.plot_all_tracks(self.hbtracks, 'LOG_TE', 'LOG_L', hb=hb,
                                 reverse_x=True, plot_dir=plot_dir)
        fh.close()
        ch.close()

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
        for i in range(len(np.nonzero(track.ptcri.iptcri > 0)[0]) - 1):
            this_eep = track.ptcri.get_ptcri_name(i, **ptcri_kw)
            next_eep = track.ptcri.get_ptcri_name(i+1, **ptcri_kw)
            ithis_eep = track.ptcri.iptcri[i]
            inext_eep = track.ptcri.iptcri[i+1]
            mess = '%.3f %s=%i %s=%i' % (track.mass,
                                         this_eep, ithis_eep,
                                         next_eep, inext_eep)

            if i != 0 and self.ptcri.iptcri[i+1] == 0:
                # except for PMS_BEG which == 0, skip if no iptcri.
                logger.error(mess)
                logger.error('skipping %s-%s\ncause the second eep is zippo.'
                               % (this_eep, next_eep))
                continue

            inds = np.arange(ithis_eep, inext_eep)
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
            tenew, lnew, agenew = splev(np.linspace(0, 1, nticks[i]), tckp)
            new_eep_dict[this_eep] = tot_pts
            tot_pts += nticks[i]
            logTe = np.append(logTe, tenew)
            logL = np.append(logL, lnew)
            Age = np.append(Age, 10 ** agenew)

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
            np.savetxt(f, to_write, fmt='%.6f')
        logger.info('wrote %s' % outfile)
        self.match_data = to_write


class ExamineTracks(TrackSet, DefineEeps, TrackDiag):
    def __init__(self, trackset_kw={}, masses=None):
        trackset_kw.update({'masses': masses})
        TrackSet.__init__(self, **trackset_kw)
        DefineEeps.__init__(self)

    def select_track(self, mass, hb=False):
        return self.tracks[list(self.masses).index(mass)]

    def ptcri_inds(self, eep, hb=False, sandro=False):
        '''
        makes a new attribute 'eep'_inds which is a list of the data index 
        for a critical point at each mass.
        example, 
        '''
        assert self.ptcri is not None, 'must have critical points loaded'
        
        eep_inds = []
        for track in self.tracks:
            if not hasattr(track.ptcri, 'iptcri'):
                self.load_critical_points(track, eep_obj=self.eep, ptcri=self.ptcri)
            pind = track.ptcri.get_ptcri_name(eep, hb=hb, sandro=sandro)
            eep_inds.append(track.ptcri.iptcri[pind])
        
        self.__setattr__('%s_inds' % eep.lower().replace('.', '_'), eep_inds)

    def eep_on_plots(self, eep, xcol, ycol, hb=False, sandro=False, ax=None,
                     write_mass=False):
        if not hasattr(self, '%s_inds' % eep.lower()):
            self.ptcri_inds(eep, hb=hb, sandro=sandro)

        inds = self.__getattribute__('%s_inds' % eep.lower())
        
        if ax is None:
            fig, ax = plt.subplots()
            if xcol == 'LOG_TE':
                ax.set_xlim(ax.get_xlim()[::-1])

        ax = self.plot_all_tracks(self.tracks, xcol, ycol, annotate=False,
                                  ax=ax, sandro=sandro, hb=hb, plot_dir=None,
                                  one_plot=True)

        for i, track in enumerate(self.tracks):
            if inds[i] == 0:
                # track is too short (probably too low mass) to have this eep.
                continue
            xdata = track.data[xcol]
            ydata = track.data[ycol]
            ax.plot(xdata[inds[i]], ydata[inds[i]], 'o')
            if write_mass is True:
                ax.text(xdata[inds[i]], ydata[inds[i]], '%.3f' % track.mass)
        

        ax.set_xlabel('$%s$' % xcol.replace('_', '\ '))
        ax.set_ylabel('$%s$' % ycol.replace('_', '\ '))
        return ax
    

def all_sets_eep_plots(eep, input_dict={}):
    tracks_dir = input_dict['tracks_dir']
    prefixs = [d for d in os.listdir(tracks_dir)
               if os.path.isdir(os.path.join(tracks_dir, d))]
    axs = []
    for prefix in prefixs:
        input_dict['prefix'] = prefix
        print prefix
        et = ExamineTracks(trackset_kw = input_dict)
        ax = et.eep_on_plots(eep, 'LOG_TE', 'LOG_L')
        ax.set_title('$%s$' % prefix.replace('_', '\ '))
        axs.append(ax)
    return axs


def do_entire_set(input_dict={}):
    tracks_dir = input_dict['tracks_dir']
    if input_dict['prefixs'] == 'all':
        prefixs = [d for d in os.listdir(tracks_dir)
                   if os.path.isdir(os.path.join(tracks_dir, d))]
    else:
        prefixs = input_dict['prefixs']

    del input_dict['prefixs']
    assert type(prefixs) == list, 'prefixs must be a list'

    for prefix in prefixs:
        logger.info('\n\n Current mix: %s \n\n' % prefix)
        this_dict = set_outdirs(input_dict, prefix)
        tm = TracksForMatch(**this_dict)
        tm.save_ptcri()
        MatchTracks(**this_dict)

def set_outdirs(indict, prefix):
    newdict = deepcopy(indict)
    newdict['prefix'] = prefix
    wkd = os.path.join(indict['tracks_dir'], newdict['prefix'])
    if indict.has_key('plot_dir') and indict['plot_dir'] == 'default':
        newdict['plot_dir'] = os.path.join(wkd, 'plots')

    if indict.has_key('outfile_dir') and indict['outfile_dir'] == 'default':
        newdict['outfile_dir'] = os.path.join(wkd, 'match')
    return newdict

def default_params(input_dict):
    # if prefix not prefixs, set the location of plots if given default.
    if input_dict.has_key('prefix'):
        input_dict = set_outdirs(input_dict, input_dict.get('prefix'))

    input_dict['eep_list'] = ['PMS_BEG', 'PMS_MIN',  'PMS_END', 'MS_BEG',
                              'MS_TMIN', 'MS_TO', 'SG_MAXL', 'RG_MINL',
                              'RG_BMP1', 'RG_BMP2', 'RG_TIP', 'HE_BEG',
                              'YCEN_0.550', 'YCEN_0.500', 'YCEN_0.400',
                              'YCEN_0.200', 'YCEN_0.100', 'YCEN_0.000', 'C_BUR']

    input_dict['eep_lengths'] = [60, 60, 60, 199, 100, 100, 70, 370, 30, 400,
                                 10, 150, 100, 100, 80, 80, 140, 150]

    if input_dict.has_key('hb') and input_dict['hb'] is True:
        input_dict['eep_list_hb'] = ['HB_BEG', 'YCEN_0.500', 'YCEN_0.400', 
                                     'YCEN_0.200', 'YCEN_0.100', 'YCEN_0.005',
                                     'AGB_LY1', 'AGB_LY2']
        eep_lengths_hb = [129, 50, 120, 70, 80, 150, 250]

    return input_dict

def check_basti():
    track_base = '/Users/phil/research/parsec2match/stellarmodels/msz83sss_eta02_wfc3ir'
    track_names = os.listdir(track_base)
    #names = 'lage        M    logL  logTe     F218W   F225W   F275W   F336W   F390W   F438W   F475W   F555W   F606W   F625W   F775W   F814W'.split()
    names = 'lage        M    logL  logTe     F098M   F105W   F110W   F125W   F126N   F127M   F128N   F130N   F132N   F139M   F140W   F153M   F160W   F164N   F167N'.split()
    tracks = [np.genfromtxt(os.path.join(track_base, t), names=names) for t in track_names]
    for t in tracks:
        fig, ax = plt.subplots()
        if len(t['logTe']) <= 1200:
            continue
        ax.plot(t['logTe'], t['logL'])
        ax.plot(t['logTe'][1200], t['logL'][1200], 'o')

        ax.set_xlim(ax.get_xlim()[::-1])

if __name__ == '__main__':
    import pdb
    input_dict = default_params(fileIO.load_input(sys.argv[1]))
    logfile = sys.argv[1].replace('inp', 'log')
    fh = logging.FileHandler(logfile)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    pdb.set_trace()
    if input_dict.has_key('prefixs'):
        do_entire_set(input_dict=input_dict)
    else:
        tm = TracksForMatch(**input_dict)
        tm.save_ptcri()
        MatchTracks(**input_dict)
