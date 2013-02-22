import traceback
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep, splprep
import fileIO
import math_utils
import graphics.GraphicsUtils as rspg    

class Track(object):
    def __init__(self, filename, ptcri=None, min_lage=0.2, cut_long=True,
                 eep=None):
        (self.base, self.name) = os.path.split(filename)
        print filename
        self.load_track(filename, min_lage=min_lage, cut_long=cut_long)
        self.filename_info()
        self.mass = self.data.MASS[0]

        if ptcri is not None:
            if type(ptcri) == str:
                pass
                #self.load_critical_points(ptcri_file=ptcri, eep_obj=eep)
            else:
                pass
                #self.load_critical_points(ptcri=ptcri, eep_obj=eep)
        else:
            self.ptcri = ptcri

    def filename_info(self):
        (pref, __, smass) = self.name.replace('.PMS','').split('_')
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
            print 'Cutting at C-burning'
            itpagb = min(icburn)
        else:
            # beginning thin shell
            print 'Cutting at thin shell burning'
            ishell, = np.nonzero((ycen == 0) & (self.data.QCAROX > self.data.QHEL*3./4.))
            if len(ishell) > 0:
                itpagb = np.min(ishell)
            else:
                itpagb = len(self.data) - 1
                ishell, = np.nonzero((self.data.LY > 1) & (self.data.QCAROX > 0.1))
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

        begin_track, = [i for i,l in enumerate(lines) if 'BEGIN TRACK' in l]
        self.header = lines[:begin_track]            
        col_keys = lines[begin_track + 1].replace('#', '').strip().split()
        begin_track_skip = 2

        if 'information' in lines[begin_track + 2]:
            col_keys = self.add_to_col_keys(col_keys, lines[begin_track + 2])
            begin_track_skip += 1

        dtype = [(c, 'd') for c in col_keys]
        try:
            data = np.genfromtxt(filename,
                                 skiprows=begin_track + begin_track_skip,
                                 names=col_keys)
        except ValueError:
            data = np.genfromtxt(filename,
                                 skiprows=begin_track + begin_track_skip,
                                 names=col_keys,
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
        new_cols = additional_col_line.split(':')[1].strip().split()
        col_keys = list(np.concatenate((col_keys, new_cols)))
        return col_keys
    
    def format_track_file(self):
        '''
        add # comment to header. Useful for fromHR2mags.
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
        
    
    def low_mass_cut(self):
        ims_beg, =  np.nonzero(self.data.MODE == self.ptcri.sandros_dict['MS_BEG'])
        
        hburn, = np.nonzero((self.data.AGE > 0.2) &
                           (self.data.LX > 0) &
                           (self.data.XCEN > 0) &
                           (self.data.LOG_Tc > 7))
        
        hburn = hburn[ims_beg:]
        

    def define_eep_stages(self):
        '''
        must define the stages here if not using Sandro's defaults.
        * denotes stages defined here.

        1 MS_BEG  Starting of the central H-burning phase
        2 MS_TMIN* First Minimum in Teff for high-mass or Xc=0.30 for low-mass stars
        3 MS_TO*   Maximum in Teff along the Main Sequence - TURN OFF POINT
        4 SG_MAXL*   Maximum in logL for high-mass or Xc=0.0 for low-mass stars
        5 RG_BASE Minimum in logL for high-mass or Base of the RGB for low-mass stars
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
        x When the energy produced by the CNO cycle is larger than that provided by the He burning during the AGB (Lcno > L3alpha)
        x The maximum luminosity before the first Thermal Pulse
        x The AGB termination
        '''
        xcen_eeps = False
        ptcri = self.ptcri
        if xcen_eeps is True:
            default_list = ['XCEN_%.2f' % frac for
                             frac in np.linspace(0., 1, 21)][::-1]

            default_list.extend(['YCEN_0.55', 'YCEN_0.50', 'YCEN_0.40',
                                 'YCEN_0.20', 'YCEN_0.10', 'YCEN_0.00'])
            
            eep_list = ptcri.please_define

            self.add_cen_eeps(xcen_eeps=xcen_eeps, xcen=True)
            self.add_cen_eeps(xcen_eeps=xcen_eeps)
            self.add_cburn_eep()
        else:
            default_list = ['MS_TMIN', 'MS_TO', 'SG_MAXL', 'RG_MINL', 'YCEN_0.55',
                            'YCEN_0.50', 'YCEN_0.40', 'YCEN_0.20', 'YCEN_0.10',
                            'YCEN_0.00']
            eep_list = ptcri.please_define
            assert default_list == eep_list, \
                'Can not define all EEPs. Please check lists'
                           
            self.add_ms_eeps()
            # even though ms_tmin comes first, need to bracket with ms_to
    
            ims_to = self.ptcri.iptcri[self.ptcri.get_ptcri_name('MS_TO',
                                                                 sandro=False)]
    
            ims_tmin = self.ptcri.iptcri[self.ptcri.get_ptcri_name('MS_TMIN',
                                                                   sandro=False)]
            if ims_to == 0:
                # should now make sure all other eeps are 0.
                [self.add_eep(cp, 0) for cp in default_list[2:]]
            else:
                #self.add_ms_tmin_eep()
                self.add_min_l_eep()
                self.add_max_l_eep()
                self.add_cen_eeps()
                #self.add_cburn_eep()
        
        assert not False in (np.diff(np.nonzero(self.ptcri.iptcri)[0]) >=  0), \
                'EEPs are not monotonically increasing. M=%.3f' % self.mass


    def add_ms_beg_eep(self):
        # points between PMS and when 0.2 frac of H has been burned
        ipms = self.ptcri.iptcri[self.ptcri.get_ptcri_name('PMS_BEG')]

        some_hburn, = np.nonzero(self.data.XCEN > 0.5)
        inds = np.arange(ipms, np.max(some_hburn))

        # deriv of log_te
        range_te = np.max(self.data.LOG_TE) - np.min(self.data.LOG_TE)
        dte = np.diff(self.data.LOG_TE) / range_te

        # deriv of log_l
        range_l = np.max(self.data.LOG_L[inds]) - np.min(self.data.LOG_L)
        dl = np.diff(self.data.LOG_L) / range_l
        
        # dt
        # I think this is right, though it could be inds[1:]
        dt = self.data.Dtime[:-1]

        # partial derivitive
        ds_dt = np.sqrt(dte ** 2 + dl ** 2) / dt
        dl_dt = dl / dt * np.max(self.data.Dtime)
        dte_dt = dte / dt * np.max(self.data.Dtime)

        # min deriv
        min_val = np.min(ds_dt[inds])
        min_arg = np.argmin(ds_dt[inds])
        ind = ipms + min_arg        
        iij, = np.nonzero(ds_dt[:ind] >= 10 * min_val)

        ms_beg = np.max(iij)
        # for low mass tracks:    
        no_xcen, = np.nonzero(self.data.XCEN == 0)
        if len(no_xcen) == 0:
            # find the max after the max
            for i in range(ms_beg, len(ds_dt)):
                imax = np.argmax(ds_dt[i:])
                if imax == 0:
                    ms_beg = i + ipms
                    break

        if self.data.MODE[ms_beg] - self.ptcri.sandros_dict['MS_BEG'] != 0:
            print self.data.MODE[ms_beg], self.ptcri.sandros_dict['MS_BEG'], self.mass
            ms_beg, diff = math_utils.closest_match(1., self.data.LX)
            ms_beg -= 2
            print 'OR', self.data.MODE[ms_beg], self.ptcri.sandros_dict['MS_BEG'], self.mass


    def add_cburn_eep(self):
        '''
        Just takes Sandro's, will need to expand here for TPAGB...
        '''
        eep_name = 'C_BUR'
        c_bur = self.ptcri.sandros_dict[eep_name]
        self.add_eep(eep_name, c_bur)


    def add_cen_eeps(self, xcen_eeps=False, xcen=False):
        '''
        Add YCEN_[fraction] eeps, if YCEN=fraction found to 0.01, will add 0 as
        the iptrcri. (0.01 is hard coded)
        '''
        if xcen:
            fcol = 'PMS_BEG'
            col = 'XCEN'
        else:
            fcol = 'RG_BASE'
            col = 'YCEN'
        
        if xcen_eeps is False:
            fcol = 'RG_BMP2'
        
        inds = np.arange(self.ptcri.iptcri[self.ptcri.get_ptcri_name(fcol, sandro=xcen_eeps)], len(self.data[col]))

        #iheb, = np.nonzero((self.data.LY > 0.) & (self.data.XCEN == 0.))
        cens = [i for i in self.ptcri.please_define if i.startswith(col)]
        for cen in cens:
            # e.g., YCEN_0.50
            frac = float(cen.split('_')[-1])
            ind, dif = math_utils.closest_match(frac, self.data[col][inds])
            icen = inds[0] + ind
            # some tolerance for a good match.
            if dif > 0.01:
                #print '%s is found, but %.2f off: %.3f M=%.3f' % (ycen, dif, self.data.YCEN[iheb[ind]], self.mass)
                icen = 0
            self.add_eep(cen, icen)


    def hb_eeps(self):
        pass
        '''
        ax = ts.plot_all_tracks('LOG_TE', 'LOG_L', annotate=False)
        for t in ts.tracks:
            ycs = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
            inds = np.array([np.argmin(abs(t.data.YCEN-yc)) for yc in ycs])
            ax.plot(t.data.LOG_TE[inds], t.data.LOG_L[inds], 'o')
        '''
    

    def add_ms_eeps(self):
        '''
        Adds  MS_TMIN and MS_TO. 
        MS_TO: This is the MAX Teff between MS_BEG and MS_TMIN.
        
        Note: MS_TMIN could be XCEN = 0.3 if no actual MS_TMIN (low masses)
              (0.3 is hard coded)

        If no MS_TO, assumes no MS_TMIN coming after it. 
        '''

        ex_inds, = np.nonzero(self.data.XCEN > 0.)

        ms_tmin = self.peak_finder('LOG_TE', 'MS_BEG', 'RG_BMP1', max=False,
                                   more_than_one='min of min',
                                   extra_inds=ex_inds, parametric_interp=False)
        ms_beg = self.ptcri.iptcri[self.ptcri.get_ptcri_name('MS_BEG', sandro=False)]
        if ms_tmin == -1 or self.mass < 1.2:
            print 'Using XCEN=0.3 for T_MIN: M=%.3f' % self.mass
            inds = self.ptcri.inds_between_ptcris('MS_BEG', 'RG_BMP1', sandro=False)
            inds = list(set(ex_inds) & set(inds))
            # low mass, no ms_tmin, use xcen = 0.3
            if len(inds) == 0:
                ms_tmin = 0
            else:
                ind, dif = math_utils.closest_match(0.3, self.data.XCEN[inds])
                ms_tmin = inds[ind]
                if ind == -1:
                    print 'no ms_tmin!'
                    ms_tmin = 0
                if dif > 0.01:
                    print 'bad match for xcen here.'

        self.add_eep('MS_TMIN', ms_tmin)

        if ms_tmin == 0:
            ms_to = 0
        else:
            if self.mass > 8.:
                ex_inds = None
            ms_to = self.peak_finder('LOG_TE', 'MS_TMIN', 'RG_BMP1', max=True,
                                     sandro=False, more_than_one='max of max',
                                     extra_inds=ex_inds, parametric_interp=False)
            if ms_to == -1:
                ms_to = self.peak_finder('LOG_TE', 'MS_TMIN', 'RG_BMP1', max=True,
                                         sandro=False, more_than_one='max of max',
                                         extra_inds=ex_inds, parametric_interp=True)

            if ms_to == -1 or ms_to == ms_tmin:
                print 'Finding MS_TO (%i) by inflections in the HRD slope. M=%.3f' % (ms_to, self.mass)

                inds = self.ptcri.inds_between_ptcris('MS_TMIN', 'RG_BMP1',
                                                      sandro=False)
                inds = list(set(ex_inds) & set(inds))
                non_dupes = self.remove_dupes(self.data.LOG_TE[inds],
                                              self.data.LOG_L[inds],
                                              'lixo', just_two=True)
                tckp, u = splprep([self.data.LOG_TE[inds][non_dupes],
                                   self.data.LOG_L[inds][non_dupes]],s=0)
                tenew, lnew = splev(np.linspace(0, 1, 100), tckp)
                # subtract off a straight line fit to get more contrast...
                slope, intercept = np.polyfit(tenew, lnew, 1)
                highc_lnew = lnew - (slope * tenew + intercept)
                peak_dict = math_utils.find_peaks(highc_lnew)
                # if more than one max is found, take the max of the maxes.
                imax = peak_dict['maxima_locations']
                # this index refers to interpolation
                almost_ind = imax[np.argmax(highc_lnew[imax])]
                # find equiv point on track grid
                ind, diff = math_utils.closest_match2d(almost_ind,
                                                       self.data.LOG_TE[inds],
                                                       self.data.LOG_L[inds],
                                                       tenew, lnew)
                ms_to = inds[ind]

                #rg_bmp1 = self.ptcri.iptcri[self.ptcri.get_ptcri_name('RG_BMP1', sandro=False)]
                #print 'XCEN of MS_TMIN: %i and RG_BMP1:%i' % (ms_tmin, rg_bmp1) 
                #print self.data.XCEN[ms_tmin], self.data.XCEN[rg_bmp1]
                
        self.add_eep('MS_TO', ms_to)
        return


    def add_min_l_eep(self):
        '''
        The MIN L before the RGB for high mass or the base of the 
        RGB for low mass.
        '''
        min_l = self.peak_finder('LOG_L', 'MS_TO', 'RG_BMP1', sandro=False,
                                 more_than_one='min of min')

        if min_l == -1:
            # no max found, need to get base of RGB.
            # this will be the deflection point in HRD space between
            # MS_TO and RG_BMP1.
            print 'Using base of RG for RG_MINL: M=%.3f' % self.mass            
            inds = self.ptcri.inds_between_ptcris('MS_TO', 'RG_BMP1', sandro=False)
            if inds[0] == 0:
                print 'RG_MINL finder will not work with no MS_TO (ie, MS_TO = 0)'
            # interpolate...
            non_dupes = self.remove_dupes(self.data.LOG_TE[inds], self.data.LOG_L[inds], 0, just_two=True)
            tckp, u = splprep([self.data.LOG_TE[inds][non_dupes],
                               self.data.LOG_L[inds][non_dupes]],s=0)
            tenew, lnew = splev(np.linspace(0, 1, 200), tckp)
            slope, intercept = np.polyfit(tenew, lnew, 1)
            highc_lnew = lnew - (slope * tenew + intercept)
            peak_dict = math_utils.find_peaks(highc_lnew)
            # if more than one max is found, take the max of the maxes.
            imin = peak_dict['minima_locations']
            # this index refers to interpolation
            almost_ind = imin[np.argmin(highc_lnew[imin])]
            # find equiv point on track grid
            ind, diff = math_utils.closest_match2d(almost_ind,
                                                   self.data.LOG_TE[inds],
                                                   self.data.LOG_L[inds],
                                                   tenew, lnew)
            min_l = inds[ind]

        if self.data.XCEN[min_l] > 0:
            print 'XCEN at RG_MINL should be zero if low mass. %f' % self.data.XCEN[min_l]
        self.add_eep('RG_MINL', min_l)

    
    def add_max_l_eep(self):
        '''
        Adds SG_MAXL between MS_TO and RG_BASE. For low mass, there will be no SG_MAXL
        and will add XCEN = 0.0 (0.0 is hard coded)
        '''

        max_l = self.peak_finder('LOG_L', 'MS_TO', 'RG_MINL', max=True,
                                 sandro=False, more_than_one='last',
                                 parametric_interp=False)

        if max_l == -1:
            print 'Using XCEN=0.0 for SG_MAXL: M=%.3f' % self.mass
            ex_inds, = np.nonzero(self.data.XCEN == 0.)
            inds = self.ptcri.inds_between_ptcris('MS_TO', 'RG_MINL', sandro=False)
            inds = list(set(ex_inds) & set(inds))
            if len(inds) == 0:
                print 'XCEN=0.0 happens after RG_MINL, RG_MINL is too early.'
                max_l = 0
            else:
                max_l = inds[0]

            msto = self.ptcri.iptcri[self.ptcri.get_ptcri_name('MS_TO', sandro=False)]
            if max_l == msto:
                print 'SG_MAXL is at MS_TO!'
                print 'XCEN at MS_TO (%i): %.3f' % (msto, self.data.XCEN[msto])
            '''
            if ind == -1 or dif < 0.01:
                print ind, dif
                print inds
                msto = self.ptcri.iptcri[self.ptcri.get_ptcri_name('MS_TO', sandro=False)]
                print 'SG_MAXL is not at XCEN=0.0: M=%.3f' % self.mass
                print 'XCEN at MS_TO (%i): %.3f' % (msto, self.data.XCEN[msto])
                max_l = 0
            else:
            '''

        self.add_eep('SG_MAXL', max_l)

    def convective_core_test(self):
        '''
        only uses sandro's defs, so doesn't need load_critical_points 
        initialized.
        '''
        ycols = ['QSCHW', 'QH1', 'QH2']
        age = self.data.AGE
        lage = np.log10(age)
        
        morigs = [t for t in self.ptcri.data_dict['M%.3f' % self.mass] if t > 0 and t < len(self.data.LOG_L)]
        iorigs = [np.nonzero(self.data.MODE == m)[0][0] for m in morigs]
        try:
            inds = self.ptcri.inds_between_ptcris('MS_BEG', 'POINT_B', sandro=True)
        except IndexError:
            inds, = np.nonzero(age > 0.2)

        try:
            lixo = lage[inds]
        except IndexError:
            inds = np.arange(len(lage))
        fig = plt.figure()
        ax = plt.axes()
        for ycol in ycols:
            ax.plot(lage[inds], self.data[ycol][inds], lw=2, label=ycol)
            #ax.scatter(lage[iorigs], self.data[ycol][iorigs], s=20, marker='o', color='black')
            ax.scatter(lage[iorigs[4]], self.data[ycol][iorigs[4]], s=100, marker='o', color='red')
            if len(iorigs) >= 7:
                ax.scatter(lage[iorigs[7]], self.data[ycol][iorigs[7]], s=100, marker='o',color='blue')
                xmax = lage[iorigs[7]] + .4
            else:
                xmax = ax.get_xlim()[1]

        xmin = lage[iorigs[4]] - .4
        ax.set_xlim(xmin, xmax)
        ax.set_title(self.mass)
        ax.legend(loc=0)
    
        for t in ts:
            inds = t.ptcri.inds_between_ptcris('MS_BEG', 'MS_TO', sandro=False)
            if inds[-1] > len(t.data.QH1):
                inds = np.arange(inds[0], len(t.data.QH1))
            qhm = np.mean(t.data.QH1[inds])
            maxqsch = np.max(t.data.QSCHW[inds])
            gtqs = np.greater(t.data.QH1[inds], t.data.QSCHW[inds])
            print '%.3f %.3f %i %i %.3f' % (qhm, maxqsch, np.sum(gtqs), len(inds), t.mass)

        
    def add_eep(self, eep_name, ind):
        '''
        Will add or replace the index of Track.data to self.ptcri.iptcri
        '''
        self.ptcri.iptcri[self.ptcri.key_dict[eep_name]] = ind
        #print eep_name, ind

    
    def peak_finder(self, col, eep1, eep2, max=False, diff_err=None,
                    sandro=True, more_than_one='max of max', mess_err=None,
                    ind_tol=3, dif_tol=0.01, extra_inds=None,
                    parametric_interp=True):

        inds = self.ptcri.inds_between_ptcris(eep1, eep2, sandro=sandro)

        if extra_inds is not None:
            inds = list(set(inds) & set(extra_inds))

        if len(inds) < ind_tol:
            print 'Peak finder %s-%s: less than %i points.' % (eep1, eep2, ind_tol)
            print inds
            return 0

        if parametric_interp is True:
            tckp = self.interpolate_te_l_age(inds)
            tenew, lnew, agenew = splev(np.linspace(0, 1, 500), tckp)
            intp_col = lnew
            if col == 'LOG_TE':
                intp_col = tenew
        else:
            cols = ['LOG_L', 'LOG_TE']
            ycol, = [a for a in cols if a != col]
            xdata = self.data[col][inds]
            ydata = self.data[ycol][inds]

            non_dupes = self.remove_dupes(xdata, ydata, 'lixo', just_two=True)
            
            k=3
            if len(non_dupes) <= k:
                k = len(non_dupes) - 1
                print 'only %i indices to fit...' % len(non_dupes)
                print 'new spline_level %i' % k
    
            tckp, u = splprep([xdata[non_dupes], ydata[non_dupes]], s=0, k=k,
                              nest=-1)
    
            xnew, ynew = splev(np.linspace(0, 1, 500), tckp)
            intp_col = xnew

            '''
            dtenew, dlnew, dagenew = splev(np.linspace(0, 1, 200), tckp, der=1)
            # find equiv point on track grid
            
            almost_ind, = np.nonzero(np.diff(np.sign(dlnew)) == 2)
            if len(almost_ind) == 0:
                tckp, u = splprep([self.data.LOG_L[inds], self.data.LOG_TE[inds]],s=0)
                lnew, tenew = splev(np.linspace(0, 1, 200), tckp)
                # find min of second derivative 
                dlnew, dtenew = splev(np.linspace(0, 1, 200), tckp, der=2)
                # find equiv point on track grid
                almost_ind, = np.nonzero(np.diff(np.sign(dtenew)) == 2)
                almost_ind = almost_ind[np.argmin(tenew[almost_ind])]
            ind, diff = math_utils.closest_match2d(almost_ind,
                                                   self.data.LOG_TE[inds],
                                                   self.data.LOG_L[inds],
                                                   tenew, lnew)
            min_l = inds[ind]
            '''
        #print 'my idea now is to use interps dl/dt to find some of these points.'
        peak_dict = math_utils.find_peaks(intp_col)
        #print peak_dict
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
                    print 'not ready yet...'

            else:
                if mess_err is not None:
                    print mess_err
                return -1

        else:
            if peak_dict['minima_number'] > 0:
                if more_than_one == 'first':
                    almost_ind = np.min(peak_dict['minima_locations'])
                elif more_than_one == 'min of min':
                    mins = peak_dict['minima_locations']
                    almost_ind = mins[np.argmin(intp_col[mins])]
            else:
                if mess_err is not None:
                    print mess_err
                return -1

        if parametric_interp is True:
            # closest point in interpolation to data
            ind, dif = math_utils.closest_match2d(almost_ind,
                                                  self.data[col][inds],
                                                  np.log10(self.data.AGE[inds]),
                                                  intp_col, agenew)
        else:
            # closest point in interpolation to data
            ind, dif = math_utils.closest_match2d(almost_ind, xdata, ydata,
                                                  xnew, ynew)
        
        if ind == -1:
            return ind

        if dif > dif_tol:
            if diff_err is not None:
                print diff_err
            else:
                print 'bad match %s-%s M=%.3f' % (eep1, eep2, self.mass) 
            return -1
        return inds[ind]

        
    def load_critical_points(self, filename=None, ptcri=None, eep_obj=None):
        '''
        iptcri is the critical point index rel to self.data
        mptcri is the model number of the critical point
        '''
        assert filename is not None or ptcri is not None, \
            'Must supply either a ptcri file or object'

        if ptcri is None:
            ptcri = critical_point(filename, eep_obj=eep_obj)
        
        self.ptcri = ptcri

        # Sandro's definitions.
        mptcri = ptcri.data_dict['M%.3f' % self.mass]
        self.ptcri.iptcri = np.concatenate([np.nonzero(self.data.MODE == m)[0]
                                            for m in mptcri])
        # sometimes, the track has been cut...
        #self.ptcri.iptcri = [p for p in self.ptcri.iptcri if p < len(self.data.LOG_L)]
        # sandro's points, just for comparison.
        self.ptcri.sptcri = self.ptcri.iptcri

        if len(ptcri.please_define) > 0:
            if hasattr(self.ptcri, 'eep'):
                # already loaded eep
                eep_obj = self.ptcri.eep                
                space_for_new = np.zeros(len(eep_obj.eep_list), dtype='int')
                self.ptcri.iptcri = space_for_new
                
            # Get the values that we won't be replacing.
            for eep_name in self.ptcri.sandro_eeps:
                if eep_name in self.ptcri.eep.eep_list:
                    # model number in track
                    mode_num = mptcri[self.ptcri.sandros_dict[eep_name]]
                    # which eep index would this replace
                    ieep =  self.ptcri.eep.eep_list.index(eep_name)
                    if mode_num == 0 or mode_num > len(self.data.LOG_L):
                        # this track doesn't have this eep either because
                        # it just doesn't get there, or it was too long, and
                        # cut before I got the track set.
                        iorig = 0
                    else:
                        # new ptcri
                        iorig, = np.nonzero(self.data.MODE == mode_num)
                        #print eep_name, iorig
                    self.ptcri.iptcri[ieep] = iorig

            #print self.ptcri.iptcri
            if self.mass < 2.:
                self.low_mass = True
            self.define_eep_stages()

        assert ptcri.Z == self.Z, 'Zs do not match between track and ptcri file'

        assert ptcri.Y == self.Y, 'Ys do not match between track and ptcri file'


    def write_culled_track(self, columns_list, **kwargs):
        '''
        write out only some columns of the PMS file.
        '''
        filename = kwargs.get('filename')
        if filename is None:
            name_dat = '%s.dat' % self.name
            filename = os.path.join(self.base, name_dat)
        to_write = [np.column_stack(self.data[c]) for c in columns_list]
        to_write = np.squeeze(np.array(to_write).T)
        with open(filename, 'w') as f:
            f.write('# %s \n' % ' '.join(columns_list))
            np.savetxt(f, to_write, fmt='%.6f')
        return filename


    def diagnostic_plots(self, inds=None, annotate=True, fig=None, axs=None):
        
        xcols = ['AGE', 'AGE', 'LOG_TE']
        xreverse = [False, False, True]
        
        ycols = ['LOG_L', 'LOG_TE', 'LOG_L']
        yreverse = [False, False, False]
        
        plt_kws = [{'lw': 2, 'color': 'black'}, 
                    {'marker': 'o', 'ls': '', 'color':'darkblue'}]
        
        if fig is None:
            fig = plt.figure(figsize=(10,10))
        if axs is None:
            axs = []
        for i, (x, y, xr, yr) in enumerate(zip(xcols, ycols, xreverse, yreverse)):
            
            axs.append(plt.subplot(2, 2, i + 1))
            
            if x == 'AGE':
                xdata = np.log10(self.data[x])
            else:
                xdata = self.data[x]

            if inds is not None:
                axs[i].plot(xdata[inds], self.data[y][inds], **plt_kws[1])
            else:
                inds, = np.nonzero(self.data.AGE > 0.2)
                axs[i].plot(xdata[inds], self.data[y][inds], **plt_kws[0])

            axs[i].set_xlabel('$%s$' % x.replace('_', '\ '))
            axs[i].set_ylabel('$%s$' % y.replace('_', '\ '))

            if annotate is True:
                self.annotate_plot(axs[i], xdata, y)

            if xr is True:
                axs[i].set_xlim(axs[i].get_xlim()[::-1])
            if yr is True:
                axs[i].set_ylim(axs[i].get_ylim()[::-1])
            axs[i].set_title('$%s$' % self.name.replace('_', '\ ').replace('.PMS', ''))
            
        return fig, axs

            
    def tracks_for_match(self, outfile='default'):        
        
        if outfile == 'default':
            outfile = os.path.join('%s' % self.base, 'match_%s.dat' % self.name.replace('.PMS',''))
            header = '# logAge Mass logTe Mbol logg C/O \n'
            #print 'writing %s' % outfile

        if hasattr(self.ptcri, 'eep') and self.ptcri.eep.nticks is not None:
            nticks = self.ptcri.eep.nticks
        else:
            nticks = np.repeat(200, len(self.ptcri.iptcri) - 1)


        logTe = np.array([])
        logL = np.array([])
        Age = np.array([])
        new_eep_dict = {}
        tot_pts = 0
        for i in range(len(np.nonzero(self.ptcri.iptcri > 0)[0]) - 1):
            mess = '%.3f %s=%i %s=%i' % (self.mass,
                                        self.ptcri.get_ptcri_name(i, sandro=False),
                                        self.ptcri.iptcri[i],
                                        self.ptcri.get_ptcri_name(i+1, sandro=False),
                                        self.ptcri.iptcri[i+1])

            if i != 0 and self.ptcri.iptcri[i+1] == 0:
                # except for PMS_BEG which == 0, skip if no iptcri.
                print mess
                print 'skipping %s-%s' % (self.ptcri.get_ptcri_name(i, sandro=False),
                                          self.ptcri.get_ptcri_name(i+1, sandro=False))
                print 'cause the second eep is zippo.'
                continue

            inds = np.arange(self.ptcri.iptcri[i], self.ptcri.iptcri[i+1])
            if len(inds) == 0:
                print mess
                print 'skipping %s-%s' % (self.ptcri.get_ptcri_name(i, sandro=False),
                                          self.ptcri.get_ptcri_name(i+1, sandro=False))
                
                print 'cause there are no inds between these crit pts.'
                continue
            
            if len(inds) == 1:
                # include the last ind.
                inds = np.arange(self.ptcri.iptcri[i], self.ptcri.iptcri[i+1] + 1)

            tckp = self.interpolate_te_l_age(inds)
            tenew, lnew, agenew = splev(np.linspace(0, 1, nticks[i]), tckp)
            new_eep_dict[self.ptcri.get_ptcri_name(i)] = tot_pts
            tot_pts += nticks[i]
            logTe = np.append(logTe, tenew)
            logL = np.append(logL, lnew)
            Age = np.append(Age, 10**agenew)
        #print new_eep_dict
        self.write_trilegal_isotrack_ptcri(Age, logL, logTe, new_eep_dict)
        line = ' '.join(map(str, np.sort(new_eep_dict.values())))
        line += '\t !M=%.6f' % self.mass
        print line
        
        Mbol = 4.77 - 2.5 * logL
        logg = -10.616 + np.log10(self.mass) + 4.0 * logTe - logL
        logAge = np.log10(Age)
        # CO place holder!
        CO = np.zeros(len(logL))
        mass_arr = np.repeat(self.mass, len(logL))
        
        to_write = np.column_stack((logAge, mass_arr, logTe, Mbol, logg, CO))


        with open(outfile, 'w') as f:
            f.write(header)
            np.savetxt(f, to_write, fmt='%.6f')

        self.match_data = to_write

    def write_trilegal_isotrack_ptcri(self, age, logl, logte, new_eep_dict, outfile=None):
        if self.mass >= 1.65:
            extra = 'INT2'
        else:
            return

        if outfile is None:
            outfile = 'ptcri_%s_Z%.4f_Y%.3f.dat.%s' % (os.path.split(self.base)[1],
                                                       self.Z, self.Y, extra)

        with open(outfile, 'a') as f:
            for i in range(len(age)):
                extra = ''
                if i in new_eep_dict.values():
                    key, = [k for (k,v) in new_eep_dict.items() if v == i]
                    if key == 'PMS_BEG':
                        start = True
                        age[i] = 0.
                        extra = ' M=%.6f PMS_BEG 0 \n' % self.mass
                    else:
                        extra += ' %s %i \n' % (key, i)
                else:
                    extra += ' %i \n' % i
                isoc = '%.12e %.5f %.5f' % (age[i], logl[i], logte[i])

                #print isoc
                f.write(isoc+extra)
        

    def interpolate_vs_age(self, col, inds, k=3, nest=-1, s=0.):
        '''
        a caller for scipy.optimize.splprep. Will also rid the array
        of duplicate values. Returns tckp, an input to splev.
        '''
        non_dupes = self.remove_dupes(self.data[col][inds],
                                      self.data.AGE[inds], 'lixo', just_two=True)

        if len(non_dupes) <= k:
            k = len(non_dupes) - 1
            print 'only %i indices to fit...' % len(non_dupes)
            print 'new spline_level %i' % k

        tckp, u = splprep([self.data[col][inds][non_dupes],
                           np.log10(self.data.AGE[inds][non_dupes])],
                           s=s, k=k, nest=nest)
        return tckp
    

    def interpolate_te_l_age(self, inds, k=3, nest=-1, s=0.):
        '''
        a caller for scipy.optimize.splprep. Will also rid the array
        of duplicate values. Returns tckp, an input to splev.
        '''
        non_dupes = self.remove_dupes(self.data.LOG_TE[inds],
                                      self.data.LOG_L[inds],
                                      self.data.AGE[inds])

        if len(non_dupes) <= k:
            k = len(non_dupes) - 1
            print 'only %i indices to fit...' % len(non_dupes)
            print 'new spline_level %i' % k

        tckp, u = splprep([self.data.LOG_TE[inds][non_dupes],
                           self.data.LOG_L[inds][non_dupes],
                           np.log10(self.data.AGE[inds][non_dupes])],
                           s=s, k=k, nest=nest)
        return tckp


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


    def plot_track(self, xcol, ycol, reverse_x=False, reverse_y=False, ax=None, 
                   inds=None, plt_kw={}, annotate=False, clean=True, ainds=None,
                   sandro=False, cmd=False):
        '''
        ainds is passed to annotate plot, and is to only plot a subset of crit
        points.
        sandro = True will plot sandro's ptcris.
        '''
        if ax is None:
            fig = plt.figure()
            ax = plt.axes()
        
        if len(plt_kw) != 0:
            # not sure why, but every time I send marker='o' it also sets
            # linestyle = '-' ...
            if 'marker' in plt_kw.keys():
                if not 'ls' in plt_kw.keys() or not 'linestyle' in plt_kw.keys():
                    plt_kw['ls'] = ''

        if clean is True and inds is None:
            # Non physical inds go away.
            inds, = np.nonzero(self.data.AGE > 0.2)

        if cmd is True:
            xdata = self.data[xcol] - self.data[ycol]
        else:
            xdata = self.data[xcol]
        if inds is not None:
            inds = [i for i in inds if i > 0]
            ax.plot(xdata[inds], self.data[ycol][inds], **plt_kw)
        else:
            ax.plot(xdata, self.data[ycol], **plt_kw)

        if reverse_x:
            ax.set_xlim(ax.get_xlim()[::-1])
        
        if reverse_y:
            ax.set_ylim(ax.get_ylim()[::-1])

        if annotate:
            ax = self.annotate_plot(ax, xcol, ycol, inds=ainds, sandro=sandro,
                                    cmd=cmd)
        
        return ax

    
    def annotate_plot(self, ax, xcol, ycol, inds=None, sandro=False):
        '''
        if a subset of ptcri inds are used, set them in inds. If you want
        sandro's ptcri's sandro=True, will also change the face color of the
        label bounding box so you can have both on the same plot.
        '''
        if sandro is False:
            ptcri = self.ptcri.iptcri
            fc = 'blue'
        else:
            ptcri = self.ptcri.sptcri
            fc = 'red'
        
        if inds is None:
            #inds = np.array([p for p in ptcri if p > 0])
            inds = ptcri
            labels = ['$%s$' % self.ptcri.get_ptcri_name(i, sandro=sandro).replace('_','\ ')
                      for i in range(len(inds))]
        else:
            iplace = np.array([np.nonzero(ptcri == i)[0][0] for i in inds])
            labels = ['$%s$' % self.ptcri.get_ptcri_name(int(i), sandro=sandro).replace('_','\ ')
                      for i in iplace]
        
        if type(xcol) == str:
            xdata = self.data[xcol]
        else:
            xdata = xcol
        
        if cmd is True:
            xdata = xdata - self.data[ycol]
        # label stylings                  
        bbox = dict(boxstyle='round, pad=0.5', fc=fc, alpha=0.5)
        arrowprops = dict(arrowstyle='->', connectionstyle='arc3, rad=0')

        for i, (label, x, y) in enumerate(zip(labels, xdata[inds],
                                       self.data[ycol][inds])):
            # varies the labels placement... default is 20, 20
            xytext = ((-1.) ** (i - 1.) * 20, (-1.) ** (i + 1.) * 20)
            ax.annotate(label, xy=(x, y), xytext=xytext, fontsize=10,
                        textcoords='offset points', ha='right', va='bottom',
                        bbox=bbox, arrowprops=arrowprops)
        return ax


    def check_ptcris(self):
        '''
        plot of the track, the interpolation, with each eep labeled
        '''
        all_inds, = np.nonzero(self.data.AGE > 0.2)
        
        iptcri, = np.nonzero(self.ptcri.iptcri > 0)
        
        last = self.ptcri.get_ptcri_name(int(iptcri[-1]),sandro=False)

        plots = [['PMS_BEG', 'PMS_MIN', 'PMS_END', 'MS_BEG'],
                 ['MS_BEG', 'MS_TMIN', 'MS_TO', 'SG_MAXL', 'RG_MINL'],
                 ['RG_MINL', 'RG_BMP1', 'RG_BMP2', 'RG_TIP'],
                 ['YCEN_0.55', 'YCEN_0.50', 'YCEN_0.40', 'YCEN_0.20', 
                 'YCEN_0.10', 'YCEN_0.00'],
                 ['YCEN_0.10', 'YCEN_0.00', 'C_BUR']]

        for i, plot in enumerate(plots):
            if last in plot:
                nplots = i + 1

        line_pltkw = {'color': 'black'}
        point_pltkw = {'marker': 'o', 'ls': ''}
        (fig, axs) = rspg.setup_multiplot(nplots, **{'figsize':(12,8)})

        for i, ax in enumerate(np.ravel(axs)):
            if i == len(plots):
                continue
            inds = [self.ptcri.get_ptcri_name(cp, sandro=False) for cp in plots[i]]
            inds = self.ptcri.iptcri[inds][np.nonzero(self.ptcri.iptcri[inds])[0]]
            if np.sum(inds) == 0:
                continue

            ax = self.plot_track('LOG_TE', 'LOG_L', ax=ax, inds=all_inds,
                                 reverse_x=True, plt_kw=line_pltkw)
            goodlimx = ax.get_xlim()
            goodlimy = ax.get_ylim()
            ax = self.plot_track('LOG_TE', 'LOG_L', ax=ax, inds=inds,
                                 plt_kw=point_pltkw, annotate=True, ainds=inds)
            
            if hasattr(self, 'match_data'):
                logl = (4.77 - self.match_data.T[3]) / 2.5
                ax.plot(self.match_data.T[2], logl, lw=2, color='green')
            
            tmax, tmin = self.maxmin('LOG_TE', inds=inds)
            lmax, lmin = self.maxmin('LOG_L', inds=inds)
            
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
            ax.set_xlabel('$LOG\ TE$',fontsize=20)
            ax.set_ylabel('$LOG\ L$', fontsize=20)
        
        
        fig.suptitle('M = %.3f Z = %.4f Y = %.4f' % (self.mass, self.Z, self.Y), fontsize=20)
        plt.savefig('ptcri_Z%g_Y%g_M%.3f.png' % (self.Z, self.Y, self.mass))
        self.plot_sandro_ptcri()

    def plot_sandro_ptcri(self):
        fig = plt.figure()
        ax = plt.axes()
        ax = self.plot_track('LOG_TE', 'LOG_L', reverse_x=1, inds=np.nonzero(self.data.AGE>0.2)[0])
        #ax = self.plot_track('LOG_TE', 'LOG_L', ax=ax, annotate=True)
        ax = self.plot_track('LOG_TE', 'LOG_L', ax=ax, annotate=True, sandro=True)
        fig.suptitle('M = %.3f Z = %.4f Y = %.4f' % (self.mass, self.Z, self.Y), fontsize=20)
        plt.savefig('sandro_ptcri_Z%g_Y%g_M%.3f.png' % (self.Z, self.Y, self.mass))
        return (fig, ax)

    def maxmin(self, col, inds=None):
        '''
        returns the max and min of a column in Track.data. use inds to index.
        '''
        arr = self.data[col]
        if inds is not None:
            arr = arr[inds]
        ma = np.max(arr)
        mi = np.min(arr)
        return (ma, mi)

class DefineEeps(object):
    def __init__(self):
        pass
    # almost everything...

class TrackDiag(object):
    def __init__(self, track):
        pass
    
    # plot_track
    # annotate_plot
    # check_ptcris
    # convective test
    # diagnostic plots
        
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
        ystr = filename.replace('.dat','').split('_Y')[-1].split('_')[0]
        if ystr.endswith('.'):
            ystr = ystr[:-1]
        self.Y = float(ystr)

    def get_ptcri_name(self, val, sandro=True):
    
        if sandro is True:      
            search_dict = self.sandros_dict
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

        please_define = []        
        if eep_obj is not None:
            please_define = [c for c in eep_obj.eep_list if c not in col_keys]

        self.data_dict = data_dict

        if eep_obj is not None:
            self.eep = eep_obj
            self.key_dict = dict(zip(eep_obj.eep_list,
                                     range(len(eep_obj.eep_list))))
            self.please_define = please_define
        else:
            self.please_define = []
            self.key_dict = self.sandros_dict


class eep(object):
    '''
    a simple class to hold eep data. Gets added as an attribute to ptcri class.
    '''
    def __init__(self, eep_list, eep_lengths=None):
        self.eep_list = eep_list
        self.nticks = eep_lengths



class TrackSet(object):
    def __init__(self, tracks_dir=None, prefix=None, ptcrifile_loc=None,
                 eep_list=None, eep_lengths=None, hb=False,
                 track_search_term='*F7_*PMS', hbtrack_search_term='*F7_*HB'):
    
        self.tracks_base = os.path.join(tracks_dir, prefix)

        self.prefix = prefix
        search_term = '*%s*dat' % prefix

        if ptcrifile_loc is not None:
            self.ptcri_file, = fileIO.get_files(ptcrifile_loc, search_term)
            if eep_list is not None:
                self.eep = eep(eep_list, eep_lengths=eep_lengths)
            else:
                self.eep = None
            self.ptcri = critical_point(self.ptcri_file, eep_obj=self.eep)
        else:
            self.ptcri = None
            self.eep = None

        self.track_names = fileIO.get_files(self.tracks_base, track_search_term)
        self.tracks = [Track(track, ptcri=self.ptcri, min_lage=0., cut_long=0)
                       for track in self.track_names]
        self.masses = np.round([t.mass for t in self.tracks], 3)

        if hb is True:
            self.hbtrack_names = fileIO.get_files(self.tracks_base, hbtrack_search_term)
            self.hbtracks = [Track(track, ptcri=self.ptcri, min_lage=0., cut_long=0)
                             for track in self.hbtrack_names]
            self.hbmasses = np.round([t.mass for t in self.hbtracks], 3)


    def plot_all_tracks(self, xcol, ycol, annotate=True, ax=None, hb=False,
                        both=False, reverse_x=False, sandro=True, cmd=False):
        didit = 0
        line_pltkw = {'color': 'black', 'alpha': 0.3}
        # would be nice to use color brewer here.
        nptcris = len(self.ptcri.key_dict)
        cols = rspg.discrete_colors(nptcris, colormap='spectral')
        point_pltkw = {'marker': 'o', 'ls': '', 'alpha': 0.5}
        
        xlims = np.array([])
        ylims = np.array([])
        if both is True:
            tracks = np.concatenate((self.tracks, self.hbtracks))
        elif hb is True:
            tracks = self.hbtracks
        else:
            tracks = self.tracks
        for t in tracks:
            #for t in ts:
            all_inds, = np.nonzero(t.data.AGE > 0.2)    
            
            ax = t.plot_track(xcol, ycol, ax=ax, inds=all_inds,
                              plt_kw=line_pltkw, cmd=cmd)
            
            xlims = np.append(xlims, np.array(ax.get_xlim()))
            ylims = np.append(ylims, np.array(ax.get_ylim()))       
            if annotate is True:
                iptcri, = np.nonzero(t.ptcri.iptcri > 0)
                inds = t.ptcri.iptcri[np.nonzero(t.ptcri.iptcri)[0]]
                if len(inds) == nptcris:
                    labs = [t.ptcri.get_ptcri_name(i, sandro=sandro).replace('_','\_') for i in range(nptcris)]
                    didit += 1
                if np.sum(inds) == 0:
                    continue
                if didit == 1:
                    didit += 1
                    if cmd is True:
                        xdata = t.data[xcol] - t.data[ycol]
                    else:
                        xdata = t.data[xcol]
                    [ax.plot(xdata[inds[i]], t.data[ycol][inds[i]],
                             color=cols[i], label='$%s$' % labs[i], **point_pltkw)
                             for i in range(len(inds))]
                else:
                    [ax.plot(xdata[inds[i]], t.data[ycol][inds[i]],
                             color=cols[i], **point_pltkw)
                             for i in range(len(inds))]
                

        if reverse_x is True:
            ax.set_xlim(np.max(xlims), np.min(xlims))
        else:
            ax.set_xlim(np.min(xlims), np.max(xlims))
        if annotate is True:
            ax.legend(loc=0, numpoints=1)
        ax.set_xlim(np.min(ylims), np.max(ylims))
        ylab = ycol.replace('_', '\ ')
        xlab = xcol.replace('_', '\ ')
        figname = '%s_%s_%s.png' % (self.prefix, xcol, ycol)

        if cmd is True:
            xlab = '%s-%s' % (xlab, ylab)
        ax.set_xlabel('$%s$' % xlab)
        ax.set_ylabel('$%s$' % ylab)
        
        plt.savefig(figname)
        return ax


    def track_set_for_match(self):
        for t in self.tracks:
            t.load_critical_points(self.ptcri_file, eep_obj=self.eep)
            t.tracks_for_match()
            t.check_ptcris()
        self.plot_all_tracks('LOG_TE', 'LOG_L')


if __name__ == '__main__':
    import pdb
    input_obj = fileIO.load_input(sys.argv[1])
    pdb.set_trace()
    ts = TrackSet(**input_obj)
    #ts.track_set_for_match()