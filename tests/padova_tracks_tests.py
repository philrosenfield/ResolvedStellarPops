import ResolvedStellarPops.PadovaTracks as pc
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





def load_ets(prefixs, sandro=False, hb=False):
    ets = []
    for prefix in prefixs:
        print prefix
        basic_kw = {'tracks_dir': '/Users/phil/research/parsec2match/S12_set/CAF09_S12D_NS/',
                    'ptcrifile_loc': '/Users/phil/research/parsec2match/S12_set/iso_s12/data/',
                    'prefix': prefix,
                    'hb': hb}

        trackset_kw = pc.default_params(basic_kw)
        et = pc.ExamineTracks(trackset_kw=trackset_kw)
        [et.load_critical_points(track, ptcri=et.ptcri) for track in et.tracks]
        ets.append(et)
    return ets

def add_eep_inds(ets, *eeps, **ptcri_inds_kw):
    [[et.ptcri_inds(eep, **ptcri_inds_kw) for eep in eeps] for et in ets]

def test_rgminl(ets, Zsubset=None):
    for et in ets:
        if Zsubset is not None:
            if et.tracks[0].Z not in Zsubset:
                continue
        et.ptcri_inds('RG_MINL')
        inds = et.rg_minl_inds
        et.ptcri_inds('SG_MAXL')
        inds = et.sg_maxl_inds

        fig, ax = plt.subplots(figsize=(8, 8))
        for i, track in enumerate(et.tracks):
            if track.mass == 0.5 or track.mass == 0.55:
                continue
            tinds =  track.ptcri.inds_between_ptcris('MS_TO', 'RG_TIP', sandro=False)
            ax.plot(track.data.LOG_TE, track.data.LOG_L, color='black', alpha=0.2)
            ax.plot(track.data.LOG_TE[tinds], track.data.LOG_L[tinds], color='red')
            ax.plot(track.data.LOG_TE[inds[i]], track.data.LOG_L[inds[i]], 'o')
            ax.text(track.data.LOG_TE[inds[i]], track.data.LOG_L[inds[i]], '%.3g' % track.mass)
        ax.set_title('Z=%.4f' % track.Z)

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
    for et in ets:
        if Zsubset is not None:
            if et.tracks[0].Z not in Zsubset:
                continue
        fig, ax = plt.subplots(figsize=(8, 8))
        for i, track in enumerate(et.tracks):
            if track.mass == 0.5 or track.mass == 0.55:
                continue
            if track.mass > ms_tmin_xcen[et.prefix]:
                continue
            inds = np.arange(et.ms_beg_inds[i], et.point_c_inds[i])
            ydata = track.data.LOG_TE
            xdata = np.arange(len(ydata))
            if len(inds) < 2:
                #print 'nothing Z=%.4f, M=%.3f' % (track.Z, track.mass)
                #print 'ms_beg: %i, ms_to %i' % (et.ms_beg_inds[i], et.ms_to_inds[i])
                #ax.plot(xdata, ydata)
                #ax.text(xdata[0], ydata[0], '%g' % track.mass, fontsize=10)
                continue
            ax.plot(xdata, ydata, color='black', alpha=0.2)
            ax.plot(xdata[inds], ydata[inds])
            ax.text(xdata[inds[0]], ydata[inds[0]], '%g' % track.mass, fontsize=10)

            tmin_ind = np.argmin(xdata[inds])
            cols = 'red'

            if track.mass < ms_tmin_xcen[et.prefix]:
                tmin_ind = np.argmin(np.abs(track.data.XCEN[inds] - 0.3))
                dif =  np.abs(track.data.XCEN[inds[tmin_ind]] - 0.3)
                cols = 'grey'
                if dif > 0.001:
                    print 'no xcen=0.3 Z=%.4f, M=%.3f' % (track.Z, track.mass)
                    continue
            elif tmin_ind < 10:
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


def test_ms_tmin(ets, Zsubset=None):
    '''
    red: normal just min te
    grey: zcen = 0.3
    black: interpolated and found minimum
    '''
    
    eeps = ['MS_BEG', 'POINT_B']
    add_eep_inds([et], *eeps, **{'hb': False, 'sandro': True})
    for et in ets:
        if Zsubset is not None:
            if et.tracks[0].Z not in Zsubset:
                continue
        fig, ax = plt.subplots(figsize=(8, 8))
        for i, track in enumerate(et.tracks):
            if track.mass == 0.5 or track.mass == 0.55:
                continue
            inds = np.arange(et.ms_beg_inds[i], et.point_b_inds[i])
            xdata = track.data.LOG_TE
            ydata = track.data.LOG_L
            ax.plot(xdata, ydata, color='black', alpha=0.3)
            if len(inds) < 2:
                print 'nothing Z=%.4f, M=%.3f' % (track.Z, track.mass)
                print 'ms_beg: %i, ms_to %i' % (et.ms_beg_inds[i], et.point_c_inds[i])
                ax.text(xdata[0], ydata[0], '%g' % track.mass, fontsize=10)
                continue
            ax.plot(xdata[inds], ydata[inds])

            tmin_ind = np.argmin(xdata[inds])
            cols = 'red'

            if track.mass < ms_tmin_xcen[et.prefix]:
                tmin_ind = np.argmin(np.abs(track.data.XCEN[inds] - 0.3))
                dif =  np.abs(track.data.XCEN[inds[tmin_ind]] - 0.3)
                cols = 'grey'
                if dif > 0.001:
                    print 'no xcen=0.3 Z=%.4f, M=%.3f' % (track.Z, track.mass)
                    continue
            elif tmin_ind > 1000000:
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


def test_heb(ets, col1='AGE', norm=True, Zsubset=None):
    '''
    can call this twice, col1 is AGE norm True, col1 is LOG_TE, norm False
    '''
    if not hasattr(ets[0], 'et.rg_tip_inds'):
        add_eep_inds(ets, ['RG_TIP', 'YCEN_0.550'], **{'hb': False, 'sandro': False}):
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