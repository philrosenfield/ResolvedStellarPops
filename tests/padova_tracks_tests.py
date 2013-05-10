import ResolvedStellarPops.PadovaTracks as pc
import matplotlib.pyplot as plt

global prefixs
prefixs = ['S12D_NS_Z0.0001_Y0.249',
          'S12D_NS_Z0.0002_Y0.249',
          'S12D_NS_Z0.0005_Y0.249',
          'S12D_NS_Z0.001_Y0.25',
          'S12D_NS_Z0.002_Y0.252',
          'S12D_NS_Z0.004_Y0.256',
          'S12D_NS_Z0.006_Y0.259',
          'S12D_NS_Z0.008_Y0.263',
          'S12D_NS_Z0.014_Y0.273',
          'S12D_NS_Z0.017_Y0.279',
          'S12D_NS_Z0.01_Y0.267',
          'S12D_NS_Z0.02_Y0.284',
          'S12D_NS_Z0.03_Y0.302',
          'S12D_NS_Z0.04_Y0.321',
          'S12D_NS_Z0.05_Y0.339',
          'S12D_NS_Z0.06_Y0.356']

# find them by eye. sorry sucker.
global ms_tmin_xcen
ms_tmin_xcen = {'S12D_NS_Z0.0001_Y0.249': 1.2,
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
                'S12D_NS_Z0.06_Y0.356': 1.10}

ms_tmin_byhand = {'S12D_NS_Z0.0001_Y0.249': {},
                'S12D_NS_Z0.0002_Y0.249':   {},
                'S12D_NS_Z0.0005_Y0.249':   {},
                'S12D_NS_Z0.001_Y0.25':     {},
                'S12D_NS_Z0.002_Y0.252':    {},
                'S12D_NS_Z0.004_Y0.256':    {},
                'S12D_NS_Z0.006_Y0.259':    {1.1: 1436},
                'S12D_NS_Z0.008_Y0.263':    {1.1: 1452, 1.15: 1413},
                'S12D_NS_Z0.014_Y0.273':    {1.1: 1408, 1.15: 1412},
                'S12D_NS_Z0.017_Y0.279':    {1.1: 1443, 1.15: 1365},
                'S12D_NS_Z0.01_Y0.267':     {1.1: 1380, 1.15: 1421},
                'S12D_NS_Z0.02_Y0.284':     {1.1: 1544, 1.15: 1570},
                'S12D_NS_Z0.03_Y0.302':     {1.1: 1525, 1.15: 1460},
                'S12D_NS_Z0.04_Y0.321':     {1.1: 1570, 1.15: 1458},
                'S12D_NS_Z0.05_Y0.339':     {},
                'S12D_NS_Z0.06_Y0.356':     {1.05: 1436}}


class ExamineTracks(pc.TrackSet, pc.DefineEeps, pc.TrackDiag):
    def __init__(self, trackset_kw={}, masses=None):
        trackset_kw.update({'masses': masses})
        pc.TrackSet.__init__(self, **trackset_kw)
        pc.DefineEeps.__init__(self)

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
        try_ax_adjust = False
        self.ptcri_inds(eep, hb=hb, sandro=sandro)
        inds = self.__getattribute__('%s_inds' % eep.lower())

        if ax is None:
            fig, ax = plt.subplots()
            try_ax_adjust = True

        ax = self.plot_all_tracks(self.tracks, xcol, ycol, annotate=False,
                                  ax=ax, sandro=sandro, hb=hb, plot_dir=None,
                                  one_plot=True)

        for i, track in enumerate(self.tracks):
            if inds[i] == 0:
                # track is too short (probably too low mass) to have this eep.
                print 'no eep for M=%.3f' % track.mass
                continue
            xdata = track.data[xcol]
            ydata = track.data[ycol]
            if color is not None:
                ax.plot(xdata[inds[i]], ydata[inds[i]], 'o', color=color)
            else:
                ax.plot(xdata[inds[i]], ydata[inds[i]], 'o')

            if write_mass is True:
                ax.text(xdata[inds[i]], ydata[inds[i]], '%.3f' % track.mass)
        

        ax.set_xlabel('$%s$' % xcol.replace('_', '\ '))
        ax.set_ylabel('$%s$' % ycol.replace('_', '\ '))

        if xcol == 'LOG_TE' and try_ax_adjust is True:
            ax.set_xlim(ax.get_xlim()[::-1])
        return ax


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


def clean_output_files():
    '''
    kills the match output files, careful with this sort of thing.
    '''
    tracks_dir = '/Users/phil/research/parsec2match/S12_set/CAF09_S12D_NS/'
    mix_dirs = [os.path.join(tracks_dir, l) for l in os.listdir(tracks_dir) if os.path.isdir(os.path.join(tracks_dir, l))]
    for mix in mix_dirs:
        tokills = [os.path.join(mix, 'match', l) for l in os.listdir(os.path.join(mix, 'match'))]
        _ = [os.remove(tokill) for tokill in tokills]


def load_ets(prefixs, sandro=False, hb=False, masses=None):
    ets = []
    for prefix in prefixs:
        print prefix
        basic_kw = {'tracks_dir': '/Users/phil/research/parsec2match/S12_set/CAF09_S12D_NS/',
                    'ptcrifile_loc': '/Users/phil/research/parsec2match/S12_set/iso_s12/data/',
                    'prefix': prefix,
                    'hb': hb,
                    'masses': masses}

        trackset_kw = pc.default_params(basic_kw)
        et = ExamineTracks(trackset_kw=trackset_kw)
        [et.load_critical_points(track, ptcri=et.ptcri) for track in et.tracks]
        ets.append(et)
    return ets


def load_low_mass_ets():
    masses = [.50, .55, .60, .65, .70, .75, .80, .85, .90, .95, 1.00, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50, 1.55, 1.60]
    ets = load_ets(prefixs, sandro=False, hb=False, masses=masses)
    return ets

 
def low_mass_debug(test_sg_maxl=False):
    ets = load_low_mass_ets()
    eeps = ['MS_TMIN', 'MS_TO', 'SG_MAXL', 'RG_MINL']
    ptcri_inds_kw = {'sandro': False, 'hb': False}
    pct.add_eep_inds(ets, *eeps, **ptcri_inds_kw)
    masses = [0.500, 0.550, 0.600, 0.650, 0.700, 0.750, 0.800, 0.850, 0.900,
              0.950, 1.000, 1.050, 1.100, 1.150, 1.200, 1.250]
    #masses = [.70, .75, .80, .85, .95, 1.00, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50, 1.55, 1.60]
    masses =  np.unique(np.concatenate([et.masses for et in ets]))
    #masses = masses[masses <= 1.15]
    for mass in masses:
        tracks = []
        for ett in ets:
            try:
                tracks.append(ett.select_track(mass))
            except ValueError:
                pass

        eep1 = 'MS_BEG'
        eep2 = 'RG_BMP1'
        
        fig, ax = plt.subplots()
        for t in tracks:
            eind1 = t.ptcri.key_dict[eep1]
            eind2 = t.ptcri.key_dict[eep2]
            inds = t.ptcri.inds_between_ptcris(eep1, eep2, sandro=False)
            et.plot_track(t, 'LOG_TE', 'LOG_L', inds=inds, ax=ax)
            pts = t.ptcri.iptcri[eind1: eind2]
            cols = rspg.discrete_colors(len(pts))
            [ax.plot(t.data.LOG_TE[pts[i]], t.data.LOG_L[pts[i]], 'o', c=cols[i])
                for i in range(len(pts))]
            ax.text(t.data.LOG_TE[pts[-1]], t.data.LOG_L[pts[-1]], '%.4f' % t.Z)
            ax.set_title('%.4f' % t.mass)
            #if test_ms_tmin is True:
            byhand_dict = et.eep_info['ms_tmin_byhand']
            if len(byhand_dict[et.prefix]) != 0 and byhand_dict[self.prefix].has_key(t.mass):
                print 'ms_tmin by hand. %.4f %.3f' % (t.Z, t.mass) 
                ms_tmin = byhand_dict[self.prefix][t.mass]
            else:
                inds = t.ptcri.inds_between_ptcris('MS_BEG', 'POINT_C', sandro=True)
                if len(inds) == 0:
                    ms_tmin = 0
                else:
                    xdata = t.data.LOG_TE[inds]
                    tmin_ind = np.argmin(xdata)
                    ms_tmin = inds[tmin_ind]
                    col = 'red'
                    if t.mass < et.eep_info['ms_tmin_xcen'][et.prefix]:
                        # use XCEN == 0.3
                        tmin_ind = np.argmin(np.abs(t.data.XCEN[inds] - 0.3))
                        # not used... but a QC:
                        dif = np.abs(t.data.XCEN[inds[tmin_ind]] - 0.3)
                        col = 'blue'
                    elif np.abs(np.diff((t.data.LOG_L[ms_tmin], t.data.LOG_L[inds[0]]))) < .1:
                        print np.abs(np.diff((t.data.LOG_L[ms_tmin], t.data.LOG_L[inds[0]])))
                        # find the arg min of teff between these points and get
                        # something very close to MS_BEG probably means the MS_BEG
                        # is at a lower Teff than Tmin.
                        mode = inds
                        tckp, u = splprep([mode, xdata], s=0, k=3, nest=-1)
                        # if not using something like model number instead of log l,
                        # the tmin will get hidden by data with t < tmin but different
                        # log l, this is only a problem for very low Z.
                        arb_arr = np.arange(0, 1, 1e-2)
                        xnew, ynew = splev(arb_arr, tckp)
                        # second derivative, bitches.
                        ddxnew, ddynew = splev(arb_arr, tckp, der=2)
                        ddyddx = ddynew/ddxnew
                        fig1, (ax1, ax2) = plt.subplots(nrows=2)
                        ax1.plot(xnew, ynew)
                        ax1.plot(xnew, ynew, ',')
                        ax2.plot(xnew, ddyddx)
                        ax2.plot(xnew, ddyddx, ',')
                        # diff displaces the index by one. 
                        aind = np.argmin(np.diff(ddynew/ddxnew)) + 1
                        # not just argmin, but must be actual min...
                        aind = [a for a in np.argsort(ddyddx) if ddyddx[a-1] > 0][0]
                        ax1.plot(xnew[aind], ynew[aind], 'o')
                        ax2.plot(xnew[aind], ddyddx[aind], 'o')
                        ax1.set_title('%.4f %.3f' % (t.Z, t.mass))
                        #tmin_ind, dif = rsp.math_utils.closest_match(ynew[aind], xdata)
                        tmin_ind, dif = rsp.math_utils.closest_match2d(aind, mode, xdata, xnew, ynew)
                        col = 'black'
            ms_tmin = inds[tmin_ind]
            ax.plot(t.data.LOG_TE[ms_tmin], t.data.LOG_L[ms_tmin], 'o', color=col)

            if test_sg_maxl is True:
                inds = t.ptcri.inds_between_ptcris('MS_TO', 'RG_MINL', sandro=False)
                ydata = t.data.LOG_L[inds]
                xdata = t.data.LOG_TE[inds]
                tckp, u = splprep([xdata, ydata], s=0, k=3, nest=-1)
                arb_arr = np.arange(0, 1, 1e-2)
                xnew, ynew = splev(arb_arr, tckp)
                # linear fit
                p = np.polyfit(xnew, ynew, 1)
                # subtract linear fit, find peaks
                peak_dict = rsp.math_utils.find_peaks(ynew - (p[0] * xnew + p[1]))
                ax.plot(xnew[peak_dict['maxima_locations'][0]], ynew[peak_dict['maxima_locations'][0]], 'o', ms=10, color='red')
                if len(peak_dict['maxima_locations']) == 0:
                    print 'still a problem with max_l %.3f' % t.mass
                # it's the later peak.
                print len(peak_dict['maxima_locations']), t.mass, t.Z
                ymax = ynew[peak_dict['maxima_locations'][0]]
                #max_l = inds[rsp.math_utils.closest_match(ymax, xdata)[0]]
                max_l = inds[rsp.math_utils.closest_match2d(peak_dict['maxima_locations'][0], 
                                                            xdata, ydata, xnew, ynew)[0]]
                ax.plot(t.data.LOG_TE[max_l], t.data.LOG_L[max_l], 'o', color='black')
                dist = np.sqrt((xdata - xnew[peak_dict['maxima_locations'][0]]) ** 2 + (ydata - ynew[peak_dict['maxima_locations'][0]]) ** 2)
                max_l = inds[np.argmin(dist)]
                ax.plot(t.data.LOG_TE[max_l], t.data.LOG_L[max_l], 'o', color='green')
    
def add_eep_inds(ets, *eeps, **ptcri_inds_kw):
    [[et.ptcri_inds(eep, **ptcri_inds_kw) for eep in eeps] for et in ets]

def test_ycen(et):
    import ResolvedStellarPops.graphics.GraphicsUtils as rspg
    eeps = ['YCEN_0.000', 'YCEN_0.100', 'YCEN_0.200', 'YCEN_0.400',
            'YCEN_0.500', 'YCEN_0.550']
    cols = rspg.discrete_colors(len(eeps))
    ptcri_inds_kw = {'hb': False, 'sandro': False}
    add_eep_inds([et], *eeps, **ptcri_inds_kw)
    fig, ax = plt.subplots(figsize=(8, 8))
    for i, track in enumerate(et.tracks):
        if track.mass == 0.5 or track.mass == 0.55:
            continue
        tinds =  track.ptcri.inds_between_ptcris('YCEN_0.550', 'YCEN_0.000',
                                                 sandro=False)
        inds = np.array([et.__getattribute__(s) for s in et.__dict__.keys() if s.startswith('ycen')]).T
        if np.sum(inds) == 0:
            continue
        #ax.plot(track.data.LOG_TE, track.data.LOG_L, color='black', alpha=0.1)
        ax.plot(track.data.LOG_TE[tinds], track.data.LOG_L[tinds], color='red')
        [ax.plot(track.data.LOG_TE[inds[i][j]], track.data.LOG_L[inds[i][j]], 'o', color=cols[j]) for j in range(len(eeps))]
        ax.text(track.data.LOG_TE[inds[i][0]], track.data.LOG_L[inds[i][0]], '%.3g' % track.mass)
    ax.set_title('Z=%.4f' % track.Z)


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


def test_rgminl(ets, Zsubset=None):
    fig, ax = plt.subplots()
    for track in et.tracks:
        if track.mass < 0.5:
            continue
        rg_minl = et.peak_finder(track, 'LOG_L', 'MS_TO', 'RG_BMP1', sandro=False,
                                 more_than_one='last')
        print track.mass, rg_minl
        inds = track.ptcri.inds_between_ptcris('MS_TO', 'RG_BMP1', sandro=False)
        xdata = track.data.LOG_TE[inds]
        ydata = track.data.LOG_L[inds]
        non_dupes = et.remove_dupes(xdata, ydata, 0, just_two=True)
        tckp, u = splprep([xdata[non_dupes], ydata[non_dupes]], s=0, k=3, nest=-1)
        arb_arr = np.arange(0, 1, 1e-2)
        xnew, ynew = splev(arb_arr, tckp)
        # linear fit
        p = np.polyfit(xnew, ynew, 1)
        # subtract linear fit, find peaks
        peak_dict = rsp.math_utils.find_peaks(ynew - (p[0] * xnew + p[1]))

        ax.plot(xnew[peak_dict['minima_locations']], ynew[peak_dict['minima_locations']], 'o', ms=10, color='red')
        if len(peak_dict['minima_locations']) != 0:
            # if more than one max is found, take the max of the maxes.
            almost_ind = peak_dict['minima_locations'][np.argmin(peak_dict['minima_locations'])]
        else:
            # sometimes (very low mass) there is not much a discernable
            # minimum, this forces a discontinuity, and finds it.
            print 'poop'
            #almost_ind = np.argmin(lnew / (slope * tenew + intercept))

        # find equiv point on track grid
        ind, diff = rsp.math_utils.closest_match2d(almost_ind,
                                               track.data.LOG_TE[inds[non_dupes]],
                                               track.data.LOG_L[inds[non_dupes]], 
                                               xnew, ynew)
        min_l = inds[ind]
        ax.plot(t.data.LOG_TE[min_l], t.data.LOG_L[min_l], 'o', color='green')

        
def test_maxl(ets, Zsubset=None, col1='LOG_TE', col2='LOG_L'):
    # not tested.
    eeps = ['MS_TO', 'RG_MINL']
    ptcri_inds_kw = {'hb': False, 'sandro': False}
    add_eep_inds(ets, *eeps, **ptcri_inds_kw)
    pind = ets[0].ptcri.get_ptcri_name('SG_MAXL', **ptcri_inds_kw)
    for et in ets:
        if Zsubset is not None:
            if et.tracks[0].Z not in Zsubset:
                continue
        fig, ax = plt.subplots(figsize=(8, 8))
        for i, track in enumerate(et.tracks):
            if track.mass == 0.5 or track.mass == 0.55:
                continue
            inds = np.arange(et.ms_to_inds[i], et.rg_minl_inds[i]+1)
            if len(inds) < 2:
                print 'no maxl Z=%.3f, M=%.3f' % (track.Z, track.mass)
                continue
            xdata = track.data[col1][inds]
            ydata = track.data[col2][inds]
            ax.plot(xdata, ydata)
            ind = track.ptcri.iptcri[pind]
            ax.plot(track.data[col1][ind], track.data[col2][ind], 'o')
        ax.set_title('Z=%.4f' % track.Z)

def test_ms_to(ets, Zsubset=None):
    eeps = ['MS_TMIN', 'RG_BMP1', 'MS_BEG']
    add_eep_inds(ets, *eeps, **{'hb': False, 'sandro': False})
    eeps = ['POINT_B', 'POINT_C']
    add_eep_inds(ets, *eeps, **{'hb': False, 'sandro': True})

    for et in ets:
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(8, 8))
        #fig, ax = plt.subplots(figsize=(8, 8))
        if Zsubset is not None:
            if et.tracks[0].Z not in Zsubset:
                continue
        byhand_dict = ms_tmin_byhand[et.prefix]
        for i, track in enumerate(et.tracks):
            #for i, track in enumerate(et.tracks):
            if track.mass == 0.5 or track.mass == 0.55:
                continue
            inds = np.arange(et.ms_beg_inds[i], et.point_c_inds[i])
            if len(inds) < 2:
                #print 'nothing Z=%.4f, M=%.3f' % (track.Z, track.mass)
                #print '%s: %i, %s %i' % (eeps[0], et.ms_tmin_inds[i],
                #                         eeps[1], et.rg_bmp1_inds[i])
                continue
            if len(byhand_dict) != 0 and byhand_dict.has_key(track.mass):
                pind = byhand_dict[track.mass]
                cols = 'red'
            else:
                pind = track.ptcri.iptcri[5]
                cols = 'green'
            xdata = track.data.LOG_TE
            #ydata = track.data.LOG_L
            #ax.plot(xdata, ydata, color='black', alpha=0.2)
            #ax.plot(xdata[inds], ydata[inds])
            #ax.plot(xdata[pind], ydata[pind], 'o', color=cols)
            #ax.text(xdata[inds[0]], ydata[inds[0]], '%g' % track.mass, fontsize=10)
            #ax.set_title('Z=%.4f' % track.Z)
            #if pind - inds[0] < 500:
            if track.mass < ms_tmin_xcen[et.prefix]:
                print 'to very close to tmin. Z=%.4f, M=%.3f' % (track.Z, track.mass)
                for ax, ydata in zip((ax1, ax2), (np.arange(len(xdata)), track.data.LOG_L)):
                    ax.plot(xdata[inds], ydata[inds], ',')
                    ax.plot(xdata[pind], ydata[pind], 'o')
                    ax.plot(xdata[track.ptcri.iptcri[4]], ydata[track.ptcri.iptcri[4]], 'o', color='black')
                    ax.text(xdata[inds[0]], ydata[inds[0]], '%g' % track.mass, fontsize=10)
                    ax.set_title('Z=%.4f' % track.Z)
            pf_kw = {'max': True, 'sandro': False, 'more_than_one': 'max of max', 
                     'parametric_interp': False}
            ms_to = self.peak_finder(track, 'LOG_TE', 'MS_TMIN', 'RG_BMP1',
                                     **pf_kw)            

            ax.set_title('Z=%.4f' % track.Z)
            to_ind = np.argmax(xdata[inds])
            cols = 'red'

            if tmin_ind < 10:
                cols='black'
                # this is kinda a code draft that's in padovatracks.
                tckp, u = splprep([np.arange(len(xdata[inds])), xdata[inds]], s=0, k=k, nest=-1)
                xnew, ynew = splev(np.arange(0, 1, 1e-2), tckp)
                dxnew, dynew = splev(np.arange(0, 1, 1e-2), tckp, der=1)
                ddxnew, ddynew = splev(np.arange(0, 1, 1e-2), tckp, der=2)
                dydx = dynew / dxnew
                aind = np.argmin(np.diff(ddynew/ddxnew)) + 1
                tmin_ind, dif = rsp.math_utils.closest_match(ynew[aind], xdata[inds])
                if dif > 1e-3:
                    print dif
                    print 'bad match Z=%.4f, M=%.3f' % (track.Z, track.mass)
                else:
                   print 'not bad.'
            ax.plot(xdata[inds[tmin_ind]], ydata[inds[tmin_ind]], 'o', color=cols)
            ax.text(xdata[inds[0]], ydata[inds[0]], '%g' % track.mass, fontsize=10)
        ax.set_title('Z=%.4f' % track.Z)

def eep_by_hand(ets, Zsubset=None):
    eeps = ['MS_BEG', 'POINT_C']
    add_eep_inds(ets, *eeps, **{'hb': False, 'sandro': True})
    add_eep_inds(ets, 'MS_TMIN', **{'hb': False, 'sandro': False})
    fig, ax = plt.subplots()
    masses =  np.unique(np.concatenate([et.masses for et in ets]))
    masses = masses[masses <= 1.15]
    for mass in masses:
        tracks = []
        for ett in ets:
            try:
                tracks.append(ett.select_track(mass))
            except ValueError:
                pass
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(8, 8))
        for ax, col in zip((ax1, ax2), ('MODE', 'LOG_L')):
            for i, track in enumerate(tracks):
                inds = track.ptcri.inds_between_ptcris('MS_BEG', 'POINT_C', sandro=True)
                if len(inds) == 0:
                    continue
                # Check for MS_TMIN by hand.
                byhand_dict = et.eep_info['ms_tmin_byhand']
                if len(byhand_dict[et.prefix]) != 0 and byhand_dict[et.prefix].has_key(track.mass):
                    print 'ms_tmin by hand. %.4f %.3f' % (track.Z, track.mass) 
                    ms_tmin = byhand_dict[et.prefix][track.mass]
                else:
                    xdata = track.data.LOG_TE[inds]
                    ydata = track.data[col][inds]
                    tmin_ind = np.argmin(xdata)
                    ms_tmin = inds[tmin_ind]
                    cols='red'
                    delta_te = np.abs(np.diff((track.data.LOG_L[ms_tmin],
                                               track.data.LOG_L[inds[0]])))
                    delta_l2 = delta_te = np.abs(np.diff((track.data.LOG_L[ms_tmin],
                                               track.data.LOG_L[inds[-1]])))
                    if track.mass < et.eep_info['ms_tmin_xcen'][et.prefix]:
                        # use XCEN == 0.3
                        dte = np.abs(track.data.XCEN[inds] - 0.3)
                        tmin_ind = np.argmin(dte)
                        # not used... but a quality control:
                        dif = dte[tmin_ind]
                        cols = 'grey'
                    elif  delta_te < .1 or delta_l2 < .1:
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
                        ax1.plot(xnew[aind], ynew[aind], '*', color = 'purple')
                        cols ='black'
                    ms_tmin = inds[tmin_ind]
                    
                ax.plot(xdata, ydata)
                ax.plot(xdata[tmin_ind], ydata[tmin_ind], 'o', color=cols)
                ax.text(xdata[0], ydata[0], '%g' % track.Z, fontsize=10)
            ax.set_title('Z=%.4f' % track.mass)


def test_ms_tmin(ets, Zsubset=None):
    '''
    red: normal just min te
    grey: zcen = 0.3
    black: interpolated and found minimum
    '''
    
    eeps = ['MS_BEG', 'POINT_B']
    add_eep_inds(ets, *eeps, **{'hb': False, 'sandro': True})
    for et in ets:
        if Zsubset is not None:
            if et.tracks[0].Z not in Zsubset:
                continue
        fig, ax = plt.subplots(figsize=(8, 8))
        for i, track in enumerate(et.tracks):
            inds = np.arange(et.ms_beg_inds[i], et.point_b_inds[i])
            if len(inds) < 2:
                if track.mass > 0.5:
                    print 'nothing Z=%.4f, M=%.3f' % (track.Z, track.mass)
                    print 'ms_beg: %i, ms_to %i' % (et.ms_beg_inds[i], et.point_c_inds[i])
                    ax.text(xdata[0], ydata[0], '%g' % track.mass, fontsize=10)
                continue

            xdata = track.data.LOG_TE[inds]
            ydata = track.data.LOG_L[inds]
            ax.plot(xdata, ydata, color='black', alpha=0.3)

            tmin_ind = np.argmin(xdata)
            cols = 'red'

            delta_te = np.abs(np.diff((ydata[tmin_ind], ydata[0])))
            print track.mass, delta_te
            if track.mass < ms_tmin_xcen[et.prefix]:
                tmin_ind = np.argmin(np.abs(track.data.XCEN[inds] - 0.3))
                dif =  np.abs(track.data.XCEN[inds[tmin_ind]] - 0.3)
                cols = 'grey'
                if dif > 0.001:
                    print 'no xcen=0.3 Z=%.4f, M=%.3f' % (track.Z, track.mass)
                    continue
            elif delta_te < .1:
                cols='black'
                mode = inds
                tckp, u = splprep([mode, xdata[inds]], s=0, k=3, nest=-1)
                arb_arr = np.arange(0, 1, 1e-2)
                xnew, ynew = splev(arb_arr, tckp)
                # second derivative, bitches.
                ddxnew, ddynew = splev(arb_arr, tckp, der=2)
                ddyddx = ddynew/ddxnew
                # not just argmin, but must be actual min...
                aind = [a for a in np.argsort(ddyddx) if ddyddx[a-1] > 0]
                if len(aind) == 0:
                    print 'no min!! %.3f %.4f' % (track.mass, track.Z)
                    continue
                else:
                    aind = aind[0]
                tmin_ind, dif = math_utils.closest_match2d(aind, mode,
                                                           xdata[inds], xnew, ynew)
                if dif > 1e-3:
                    print dif
                    print 'bad match Z=%.4f, M=%.3f' % (track.Z, track.mass)

            ax.plot(xdata[inds[tmin_ind]], ydata[inds[tmin_ind]], 'o', color=cols)
            ax.text(xdata[inds[0]], ydata[inds[0]], '%g' % track.mass, fontsize=10)
        ax.set_title('Z=%.4f' % track.Z)


def test_heb(ets, col1='AGE', norm=True, Zsubset=None):
    '''
    can call this twice, col1 is AGE norm True, col1 is LOG_TE, norm False
    '''
    if not hasattr(ets[0], 'et.rg_tip_inds'):
        add_eep_inds(ets, ['RG_TIP', 'YCEN_0.550'], **{'hb': False, 'sandro': False})
    for et in ets:
        if Zsubset is not None:
            if et.tracks[0].Z not in Zsubset:
                continue
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(8, 8))
        for ax, col in zip((ax1, ax2), ('LY', 'LOG_L')):
            for i, track in enumerate(et.tracks):
                if track.mass == 0.5 or track.mass == 0.55:
                    continue
                inds = np.arange(et.rg_tip_inds[i], et.ycen_0_550_inds[i]+1)
                if len(inds) < 2:
                    print 'no heb Z=%.3f, M=%.3f' % (track.Z, track.mass)
                    continue
                xdata = track.data[col1][inds]
                if norm is True:
                    xdata /= np.max(xdata)
                ax.plot(xdata, track.data[col][inds])
                min = np.argmin(track.data.LY[inds])
                npts = inds[-1] - inds[0] + 1
                half_of_em = npts/3
                max = np.argmax(track.data.LY[inds[:half_of_em]])
                rat = track.data.LY[inds[max]]/track.data.LY[inds[0]]
                if min == 0 or rat > 10:
                    amin = np.argmin(track.data.LY[inds[max+1:]])
                    min = max + 1 + amin
                
                ax.plot(xdata[min], track.data[col][inds[min]], 'o')
                ax.set_title('Z=%.4f' % track.Z)
            ax1.set_ylim(-0.05, 0.4)