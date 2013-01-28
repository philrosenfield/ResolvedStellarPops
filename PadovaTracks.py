import traceback
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep, splprep
import fileIO
import math_utils
    

class Track(object):
    def __init__(self, filename, ptcri=None, min_lage=0.2, cut_long=True,
                 eep=None):
        (self.base, self.name) = os.path.split(filename)
        self.load_track(filename, min_lage=min_lage, cut_long=cut_long)
        self.filename_info()
        self.mass = self.data.MASS[0]
        if ptcri is not None:
            if type(ptcri) == str:
                self.load_critical_points(ptcri_file=ptcri, eep_obj=eep)
            else:
                self.load_critical_points(ptcri=ptcri, eep_obj=eep)
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
        col_keys = lines[begin_track + 1].strip().split()
        dtype = [(c, '<f8') for c in col_keys]
        try:
            data = np.genfromtxt(filename,
                                 skiprows=begin_track + 2,
                                 names=col_keys)
        except ValueError:
            data = np.genfromtxt(filename,
                                 skiprows=begin_track + 2,
                                 skip_footer = 2,
                                 names=col_keys)

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

    def define_eep_stages(self):
        '''
        must define the stages here if not using Sandro's defaults.
        * denotes stages defined here.

        1 MS_BEG  Starting of the central H-burning phase
        2 MS_TMIN* First Minimum in Teff for high-mass or Xc=0.30 for low-mass stars
        3 MS_TO*   Maximum in Teff along the Main Sequence - TURN OFF POINT
        4 MAX_L*   Maximum in logL for high-mass or Xc=0.0 for low-mass stars
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

        ptcri = self.ptcri
        default_list = ['MS_TMIN', 'MS_TO', 'MAX_L', 'YCEN_0.55', 'YCEN_0.50',
                        'YCEN_0.40', 'YCEN_0.20', 'YCEN_0.10', 'YCEN_0.00']

        eep_list = ptcri.please_define
        assert default_list == eep_list, \
            'Can not define all EEPs. Please check lists'

        self.add_ms_to_eep()
        # even though ms_tmin comes first, need to bracket with ms_to

        if self.ptcri.iptcri[self.ptcri.get_ptcri_name('MS_TO', sandro=False)] != 0:
            if self.ptcri.iptcri[self.ptcri.get_ptcri_name('MS_TO', sandro=False)] < self.ptcri.iptcri[self.ptcri.get_ptcri_name('MS_TMIN', sandro=False)]:
                plt.figure()
                ax = plt.axes()
                ax.plot(self.data.LOG_TE)
                ax.plot(ms_tmin, self.data.LOG_TE[ms_tmin], 'o')
                ax.plot(ms_to, self.data.LOG_TE[ms_to], 'o')
                #  the tmin can't come after ms_to, this is low mass, use xcen
                print 'MS_TO comes before MS_TMIN, should be low mass M=%.3f' % self.mass
        else:
            self.add_ms_tmin_eep()    
            self.add_max_l_eep()        
            self.add_ycen_eeps()
            self.add_cburn_eep()

    def add_ms_eep(self):
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

        if self.data.MODE[ms_beg] - self.ptcri.sandro_dict['MS_BEG'] != 0:
            print self.data.MODE[ms_beg], self.ptcri.sandro_dict['MS_BEG'], self.mass
            ms_beg, diff = math_utils.closest_match(1., self.data.LX)
            ms_beg -= 2
            print 'OR', self.data.MODE[ms_beg], self.ptcri.sandro_dict['MS_BEG'], self.mass

        
    def add_cburn_eep(self):
        '''
        Just takes Sandro's, will need to expand here for TPAGB...
        '''
        eep_name = 'C_BUR'
        c_bur = self.ptcri.sandro_dict[eep_name]
        self.add_eep(eep_name, c_bur)


    def add_ycen_eeps(self):
        '''
        Add YCEN_[fraction] eeps, if YCEN=fraction found to 0.01, will add 0 as
        the iptrcri. (0.01 is hard coded)
        '''
        iheb, = np.nonzero((self.data.LY > 0.) & (self.data.XCEN == 0.))
        ycens = [y for y in self.ptcri.please_define if y.startswith('YCEN')]
        for ycen in ycens:
            if len(iheb) >= 10:    
                # e.g., YCEN_0.50
                frac = float(ycen.split('_')[-1])
                ind, dif = math_utils.closest_match(frac, self.data.YCEN[iheb])
                # some tolerance for a good match.
                if dif > 0.01:
                    #print '%s is found, but %.2f off: %.3f M=%.3f' % (eep_name, dif, self.data.YCEN[iheb][ind], self.mass)
                    ind = 0
            else:
                ind = 0
            self.add_eep(ycen, ind)

    def add_ms_to_eep(self):
        '''
        Adds  MS_TMIN and MS_TO. 
        Doing both in one method because both take same call to find_peaks.
        Note: MS_TMIN could be XCEN = 0.3 if no actual MS_TMIN (low masses)
              (0.3 is hard coded)        
        '''
        inds = self.ptcri.inds_between_ptcris('MS_BEG', 'RG_BASE')
        if len(inds) == 0:
            if self.mass > 0.5:
                print 'Woah, there is nothing between MS_BEG and RG_BASE.'
            ms_to = 0
        else:
            tckp, u = splprep([self.data.LOG_TE[inds], self.data.LOG_L[inds]],
                             s=0., k=3)
            tenew, lnew = splev(np.linspace(0, 1, 100), tckp)

            peak_dict = math_utils.find_peaks(tenew)
            if peak_dict['maxima_number'] > 0:
                itemax = peak_dict['maxima_locations']
                almost_ind = itemax[np.argmax(tenew[itemax])]
                ind, diff = math_utils.closest_match(tenew[almost_ind], self.data.LOG_TE[inds])
                ms_to = ind + inds[0]
            else:
                print 'MS_TO not found. M=%.3f' % self.mass

        self.add_eep('MS_TO', ms_to)
        return

    def add_ms_tmin_eep(self):
        inds = self.ptcri.inds_between_ptcris('MS_BEG', 'MS_TO', sandro=False)
        if self.ptcri.iptcri[self.ptcri.get_ptcri_name('MS_TO', sandro=False)] == 0:
            if self.mass > 0.5:
                print 'there is no MS_TO! M=%.3f' % self.mass
            ms_tmin = 0          
        else:
            peak_dict = math_utils.find_peaks(self.data.LOG_TE[inds])

            if peak_dict['minima_number'] == 1:
                # Fount MS_TMIN
                ind, = peak_dict['minima_locations']
                ms_tmin = ind + inds[0]
            elif peak_dict['minima_number'] == 0:
                # low mass, no ms_tmin, use xcen = 0.3
                ind, dif = math_utils.closest_match(0.3, self.data.XCEN[inds])
                ms_tmin = ind + inds[0]
            elif peak_dict['minima_number'] > 1:
                # take the first min
                min_inds = peak_dict['minima_locations']
                ind = min_inds[0]
                ms_tmin = ind + inds[0]
        self.add_eep('MS_TMIN', ms_tmin)

    
    def add_max_l_eep(self):
        '''
        Adds MAX_L between MS_TO and RG_BASE. For low mass, there will be no MAX_L
        and will add XCEN = 0.0 (0.0 is hard coded)
        '''
        inds = self.ptcri.inds_between_ptcris('MS_TO', 'RG_BASE', sandro=False)
        if len(inds) < 10:
            max_l = 0
        else:    
            peak_dict = math_utils.find_peaks(self.data.LOG_L[inds])
            if peak_dict['maxima_number'] < 1:
                print 'no MAX L, doing XCEN. M=%.3f' % self.mass
                ind, dif = math_utils.closest_match(0.0, self.data.XCEN[inds])
                max_l = ind + inds[0]
                assert dif < 0.01, \
                    'MAX_L error: no XCEN value near 0.0 M=%.3f' % self.mass
            elif peak_dict['maxima_number'] == 1:
                ind, = peak_dict['maxima_locations']
                max_l = ind + inds[0]
            elif peak_dict['maxima_number'] > 1:
                # take the max of the max!
                max_inds = peak_dict['maxima_locations']
                ind = max_inds[np.argmax(self.data.LOG_L[inds][max_inds])]
                max_l = ind + inds[0]
        self.add_eep('MAX_L', max_l)


    def add_eep(self, eep_name, ind):
        '''
        Will add the index of Track.data to self.ptcri.iptcri
        and will add the value of Track.data.MODE to self.ptcri.mptcri
        if no eep (ind == 0), mptcri will get -1. 
        '''
        self.ptcri.iptcri[self.ptcri.key_dict[eep_name]] = ind
        if ind == 0:
            mind = -1
            #print 'No %s found M=%.3f' % (eep_name, self.mass)
        else:
            mind, = np.nonzero(self.data.MODE == ind)
            self.ptcri.mptcri[self.ptcri.key_dict[eep_name]] = mind


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
        self.ptcri.mptcri = ptcri.data_dict['M%.3f' % self.mass]
        self.ptcri.iptcri = np.concatenate([np.nonzero(self.data.MODE == m)[0] for m in self.ptcri.mptcri])
        self.ptcri.sandro_dict = dict(zip(ptcri.sandro_eeps, ptcri.data_dict['M%.3f' % self.mass]))

        if len(ptcri.please_define) > 0:
            if hasattr(self.ptcri, 'eep'):
                # already loaded eep
                eep_obj = self.ptcri.eep
            if len(self.ptcri.iptcri) != len(eep_obj.eep_list):
                eep_diff = len(eep_obj.eep_list) - len(self.ptcri.iptcri)
                space_for_new = np.zeros(eep_diff, dtype='int') - 1
                self.ptcri.iptcri = np.concatenate((self.ptcri.iptcri, space_for_new))
                self.ptcri.mptcri = np.concatenate((self.ptcri.mptcri, space_for_new))
            self.define_eep_stages()

        assert self.data.MODE[0] <= self.ptcri.mptcri[0], \
            'First critical point not contained in Track.data.MODE.'

        mptcris, = self.ptcri.mptcri.nonzero()

        assert self.data.MODE[-1] >= self.ptcri.mptcri[mptcris][-1], \
            'Last critical point not contained in Track.data.MODE.'

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
                axs[i].plot(xdata, self.data[y], **plt_kws[0])

            axs[i].set_xlabel('$%s$' % x.replace('_', '\ '))
            axs[i].set_ylabel('$%s$' % y.replace('_', '\ '))

            if annotate is True:
                self.annotate_plot(ax, xdata, y)

            if xr is True:
                axs[i].set_xlim(axs[i].get_xlim()[::-1])
            if yr is True:
                axs[i].set_ylim(axs[i].get_ylim()[::-1])
            axs[i].set_title('$%s$' % self.name.replace('_', '\ ').replace('.PMS', ''))
            
        return fig, axs

            
    def tracks_for_match(self, outfile='default', ptcrifile=None, nticklist=None):        
        assert ptcrifile is not None or hasattr(self, 'ptcri'), \
            'Need either a critical point file or critical_point object'
        
        if outfile == 'default':
            outfile = os.path.join('%s' % self.base, 'match_%s.dat' % self.name.replace('.PMS',''))
            print 'writing %s' % outfile

        if ptcrifile is not None:
            self.load_critical_points(filename=ptcrifile)

        # add the final point of the track to the crit point array.
        #self.iptcri = np.append(self.iptcri, len(self.data)-1)

        nknots = -1
        spline_level = 3
        smooth = 0.

        if nticklist is None:
            nticks = np.repeat(200, len(self.ptcri.iptcri)-1)
        else:
            nticks = nticklist
        logTe = np.array([])
        logL = np.array([])
        Age = np.array([])
        for i in range(len(self.ptcri.iptcri)-1):
            inds = np.arange(self.ptcri.iptcri[i], self.ptcri.iptcri[i+1])
            print self.mass, self.ptcri.get_ptcri_name(i, sandro=False), self.ptcri.get_ptcri_name(i+1, sandro=False)
            '''
            if self.mptcri[i+1] - self.mptcri[i] < 35: 
                print self.mptcri[i+1], self.mptcri[i], self.mptcri[i+1] - self.mptcri[i]
                print 'skipping! m %s' % self.mass
                continue
            if len(inds) < 20:
                print 'skipping! i %s' % self.mass
                continue
            '''
            non_dupes = self.remove_dupes(self.data.LOG_TE[inds],
                                          self.data.LOG_L[inds],
                                          self.data.AGE[inds])
            
            if len(non_dupes) <= spline_level:
                spline_level = len(non_dupes) - 1
                print 'only %i indices to fit...' % len(non_dupes)

            try:
                tckp, u = splprep([self.data.LOG_TE[inds][non_dupes],
                                   self.data.LOG_L[inds][non_dupes],
                                   np.log10(self.data.AGE[inds][non_dupes])],
                                   s=smooth, k=spline_level,
                                   nest=nknots)
            except TypeError:
                traceback.print_exc(file=sys.stdout)
                tenew = np.repeat(self.data.LOG_TE[inds][non_dupes], nticks[i])
                lnew = np.repeat(self.data.LOG_TE[inds][non_dupes], nticks[i])
                agenew = np.repeat(self.data.LOG_TE[inds][non_dupes], nticks[i])
                
            tenew, lnew, agenew = splev(np.linspace(0, 1, nticks[i]), tckp)
            #plt.plot(tenew,lnew, '.', color='red')
            logTe = np.append(logTe, tenew)
            logL = np.append(logL, lnew)
            Age = np.append(Age, 10**agenew)
        
        Mbol = 4.77 - 2.5 * logL
        logg = -10.616 + np.log10(self.mass) + 4.0 * logTe - logL
        logAge = np.log10(Age)
        # place holder!
        CO = np.zeros(len(logL))
        mass_arr = np.repeat(self.mass, len(logL))
        
        header = '# logAge Mass logTe Mbol logg C/O \n'
        to_write = np.column_stack((logAge, mass_arr, logTe, Mbol, logg, CO))
        with open(outfile, 'w') as f:
            f.write(header)
            np.savetxt(f, to_write, fmt='%.6f')

        self.match_data = to_write

    
    def remove_dupes(self, x, y, z):
        '''
        Duplicates will make the interpolation fail, and thus delay graduation
        dates. Here is where they die.
        '''
        inds = np.arange(len(x))
        mask, = np.nonzero(((np.diff(x) == 0) & (np.diff(y) == 0) & (np.diff(z) == 0)))
        non_dupes = [i for i in inds if i not in mask]
        
        return non_dupes


    def plot_track(self, xcol, ycol, reverse_x=False, reverse_y=False, ax=None, 
                   inds=None, plt_kw={}, annotate=False, clean=True):
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

        if inds is not None:
            inds = [i for i in inds if i > 0]
            ax.plot(self.data[xcol][inds], self.data[ycol][inds], **plt_kw)
        else:
            ax.plot(self.data[xcol], self.data[ycol], **plt_kw)

        if reverse_x:
            ax.set_xlim(ax.get_xlim()[::-1])
        
        if reverse_y:
            ax.set_ylim(ax.get_ylim()[::-1])

        if annotate:
            ax = self.annotate_plot(ax, xcol, ycol)
        return ax

    
    def annotate_plot(self, ax, xcol, ycol):
        inds = np.array([p for p in self.ptcri.iptcri if p > 0])

        if type(xcol) == str:
            xdata = self.data[xcol]
        else:
            xdata = xcol
            
        labels = ['$%s$' % self.ptcri.get_ptcri_name(i, sandro=False).replace('_','\ ')
                  for i in range(len(inds))]

        # label stylings                  
        bbox = dict(boxstyle='round, pad=0.5', fc='blue', alpha=0.5)
        arrowprops = dict(arrowstyle='->', connectionstyle='arc3, rad=0')

        for label, x, y in zip(labels, xdata[inds],
                                       self.data[ycol][inds]):
            ax.annotate(label, xy=(x, y), xytext=(-20, 20), fontsize=10,
                        textcoords='offset points', ha='right', va='bottom',
                        bbox=bbox, arrowprops=arrowprops)
        return ax


    def check_ptcris(self):
        '''
        plot of the track, the interpolation, with each eep labeled
        '''
        all_inds, = np.nonzero(self.data.AGE > 0.2)
        ax = self.plot_track('LOG_TE', 'LOG_L', inds=all_inds, reverse_x=True,
                             plt_kw={'color': 'black'})

        ax = self.plot_track('LOG_TE', 'LOG_L', ax=ax, inds=self.ptcri.iptcri,
                             plt_kw={'marker': 'o', 'ls': ''}, annotate=True)
                
        if hasattr(self, 'match_data'):
            logl = (4.77 - self.match_data.T[3]) / 2.5
            ax.plot(self.match_data.T[2], logl, lw=2, color='green')

        ax.set_title('M = %.3f' % self.mass)
        plt.savefig('ptcri_Z%g_Y%g_M%.3f.png' % (self.Z, self.Y, self.mass))


class TrackDiag(object):
    def __init__(self, track):
        pass
    
    #finished phases
    #very low mass or not        
        
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
        self.Z = float(zstr[:5].split('_')[0])
        ystr = filename.split('_Y')[-1][:5].split('_')[0]
        if ystr.endswith('.'):
            ystr = ystr[:-1]
        self.Y = float(ystr)

    def get_ptcri_name(self, val, sandro=True):
    
        if sandro == True:      
            sandros_dict = dict(zip(self.sandro_eeps,
                                    range(len(self.sandro_eeps))))
            if type(val) == int:
                return [name for name, pval in sandros_dict.items()
                        if pval == val][0]
            elif type(val) == str:
                return [pval for name, pval in sandros_dict.items()
                        if name == val][0]
        else:        
            if type(val) == int:
                return [name for name, pval in self.key_dict.items()
                        if pval == val][0]
            elif type(val) == str:
                return [pval for name, pval in self.key_dict.items()
                        if name == val][0]

    def inds_between_ptcris(self, name1, name2, sandro=True):
        '''
        returns the indices from [name1, name2)
        this is iptcri, not mptcri
        they will be the same inds that can be used in Track.data
        '''
        first = self.iptcri[self.get_ptcri_name(name1, sandro=sandro)]
        second = self.iptcri[self.get_ptcri_name(name2, sandro=sandro)]
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
        
        begin, = [i for i in range(len(lines)) if lines[i].startswith('#') and 'F7' in lines[i]]
        # the final column is a filename.
        col_keys = lines[begin + 1].replace('#', '').strip().split()[3:-1]

        # useful to save what Sandro defined
        self.sandro_eeps = col_keys
        
        # ptcri file has filename as col #19 so skip the last column
        usecols = range(0, 18)
        please_define = []
        if eep_obj is not None:
            #skip_cols = [c for c in col_keys if c not in eep_obj.eep_list]
            #iskips = [col_keys.index(i) for i in skip_cols]
            #usecols = [u for u in usecols if u not in iskips]
            please_define = [c for c in eep_obj.eep_list if c not in col_keys]

        data = np.genfromtxt(filename, usecols=usecols, skip_header=begin + 2)
        self.data = data
        self.masses = data[:, 1]

        # ptcri has all track data, but this instance only cares about one mass.
        data_dict = {}
        for i in range(len(data)):
            str_mass = 'M%.3f' % self.masses[i]
            data_dict[str_mass] = data[i][3:].astype('int')
        
        self.eep = eep_obj
        self.data_dict = data_dict
        self.key_dict = dict(zip(eep_obj.eep_list, range(len(eep_obj.eep_list))))
        self.please_define = please_define
  
class eep(object):
    '''
    a simply class to hold eep data. Gets added as an attribute to ptcri class.
    '''
    def __init__(self, inputobj):
        self.eep_list = inputobj.eep_list
        self.nticks = inputobj.eep_lengths
    

def track_set_for_match(input_obj):
    
    track_names = fileIO.get_files(os.path.join(input_obj.tracks_dir, input_obj.prefix), '*PMS')
    ptcri_file, = fileIO.get_files(os.path.join(input_obj.ptcrifile_loc), '*%s*dat' % input_obj.prefix)

    ptcri = critical_point(ptcri_file, eep_obj=eep(input_obj))

    for track in track_names:
        t = Track(track, ptcri=ptcri, min_lage=0., cut_long=0)
        #t.load_critical_points(ptcri_file, eep_obj=eep(input_obj))
        t.tracks_for_match()
        t.check_ptcris()

if __name__ == '__main__':
    import pdb
    input_obj = fileIO.input_file('/Users/phil/Desktop/OmegaCenTracks/trilegal_runs/galaxy_input/parsec2match.inp')
    pdb.set_trace()
    track_set_for_match(input_obj)