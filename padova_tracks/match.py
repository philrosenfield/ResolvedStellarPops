from __future__ import print_function
import numpy as np
import matplotlib.pylab as plt
from ..graphics.GraphicsUtils import discrete_colors
import os
from eep import DefineEeps
from eep import critical_point
from tracks import TrackSet
from scipy.interpolate import splev
from tracks import TrackDiag
from tracks import Track
import pprint
max_mass = 120.

class MatchTracks(critical_point.Eep, TrackSet, TrackDiag):
    '''
    a simple check of the output from TracksForMatch. I want it to run on the
    same input file as TracksForMatch.
    '''
    def __init__(self, inputs):
        critical_point.Eep.__init__(self)
        inputs.match = True
        TrackSet.__init__(self, inputs=inputs)
        self.flag_dict = inputs.flag_dict
        self.match_info = {}

    def check_tracks(self):
        '''
        check the tracks for identical ages and monotontically increasing ages
        test results go into self.match_info dictionary with keys set by
        M%.3f % track.mass and filled with a list of strings with the info.
        
        if the track has already been flagged, not test occurs.
        '''
        for i, t in enumerate(self.tracks):
            key = 'M%.3f' % t.mass
            #match_info = self.match_info['M%.3f' % t.mass]
            if self.flag_dict['M%.3f' % t.mass] is not None:
                continue
            test = np.diff(t.data.logAge) > 0
            if False in test:
                bads, = np.nonzero(np.diff(t.data.logAge) < 0)
                if len(bads) != 0:
                    if not key in self.match_info:
                        self.match_info[key] = []
                        match_info = self.match_info[key]
                    match_info.append('Age not monotonicly increasing near')
                    bad_inds = \
                        np.unique([np.nonzero(j - np.cumsum(self.nticks) < 0)[0][0]
                                   for j in bads])
                    match_info.append([np.array(self.eep_list)[bad_inds],
                                      t.data.logAge[bads]])
                    self.flag_dict['M%.3f' % t.mass] = 'age decreases on track'
                bads1, = np.nonzero(np.diff(t.data.logAge) == 0)                
                if len(bads1) != 0:
                    if not key in self.match_info:
                        self.match_info[key] = []
                        match_info = self.match_info[key]
                    match_info.append(['%i identical age values' % (len(bads1))])
                    bad_inds = \
                        np.unique([np.nonzero(j - np.cumsum(self.nticks) < 0)[0][0]
                                   for j in bads1])
                    match_info.append(['near', np.array(self.eep_list)[bad_inds]])
                    match_info.append(['log ages:', t.data.logAge[bads1]])
                    match_info.append(['inds:', bads1])


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
        self.debug = inputs.debug

    def match_interpolation(self, inputs):
        # to pass the flags to another class
        flag_dict = {}
        info_dict = {}
        trackss = []
        hbswitch = []
        eep = critical_point.Eep()
        if inputs.hb_only is False:
            trackss = [self.tracks]
            hbswitch = [False]

        if inputs.hb is True:
            trackss.append(self.hbtracks)
            hbswitch.append(True)

        for i, tracks in enumerate(trackss):
            for track in tracks:
                flag_dict['M%.3f' % track.mass] = track.flag

                if track.flag is not None:
                    print('skipping track because of flag:', track.flag)
                    continue

                # interpolate tracks for match
                self.prepare_track(track, inputs.ptcri, outfile_dir=inputs.outfile_dir,
                                   diag_plot=inputs.diag_plot, hb=hbswitch[i])

                info_dict['M%.3f' % track.mass] = track.info

                if inputs.diag_plot is True:
                    # make diagnostic plots
                    for xcol in ['LOG_TE', 'AGE']:
                        self.check_ptcris(track, plot_dir=inputs.plot_dir,
                                          xcol=xcol, hb=hbswitch[i],
                                          ptcri=inputs.ptcri)
            if inputs.diag_plot is True:
                # make summary diagnostic plots
                if hbswitch[i] is False:
                    # split into three plots
                    # PMS_BEG to MS_BEG, MS_BEG to RG_TIP, and RG_TIP to TPAGB.
                    eep_lists = [eep.eep_list[0:11], eep.eep_list[10:14],
                                 eep.eep_list[13:]]
                    extras = ['pms', 'ms', 'rg', 'ycen']
                    eep_lengthss = [eep.nticks[0:11], eep.nticks[10:14],
                                    eep.nticks[13:]]
                else:
                    eep_lists = [eep.eep_list_hb]
                    extras = ['hb']
                    eep_lengthss = [eep.nticks_hb]
    
                for i in range(len(eep_lengthss)):
                    pat_kw = {'eep_lengths': eep_lengthss[i],
                              'eep_list': eep_lists[i],
                              'extra': extras[i],
                              'plot_dir': inputs.plot_dir}
                    self.diag_plots(pat_kw=pat_kw, xcols=['LOG_TE', 'AGE'])

        logfile = os.path.join(self.tracks_base, 'logfile_%s.dat' % self.prefix.lower())
        with open(logfile, 'w') as out:
            for m, d in info_dict.items():
                out.write('# %s\n' % m)
                for k, v in d.items():
                    out.write('%s: %s\n' % (k, v))
        
        return flag_dict

    def prepare_track(self, track, ptcri, outfile='default', hb=False,
                      outfile_dir=None, diag_plot=False):
        broken = False
        if outfile == 'default':
            if outfile_dir is None or outfile_dir is 'default':
                outfile_dir = track.base
        outfile = os.path.join(outfile_dir,
                               'match_%s.dat' % track.name.replace('.PMS', ''))
        header = '# logAge Mass logTe Mbol logg C/O \n'

        if hb is True:
            nticks = ptcri.eep.nticks_hb
        else:
            nticks = ptcri.eep.nticks

        logTe = np.array([])
        logL = np.array([])
        Age = np.array([])
        tot_pts = 0
        ptcri_kw = {'sandro': False, 'hb': hb}
        ndefined_ptcri = len(np.nonzero(track.iptcri >= 0)[0]) 
        for i in range(ndefined_ptcri - 1):
            this_eep = ptcri.get_ptcri_name(i, **ptcri_kw)
            next_eep = ptcri.get_ptcri_name(i+1, **ptcri_kw)
            ithis_eep = track.iptcri[i]
            inext_eep = track.iptcri[i+1]
            mess = '%.3f %s=%i %s=%i' % (track.mass,
                                         this_eep, ithis_eep,
                                         next_eep, inext_eep)

            if i != 0 and track.iptcri[i+1] == 0:
                # except for PMS_BEG which == 0, skip if no iptcri.
                # this is not an error, just the end of the track.
                continue

            if ithis_eep == -1:
                print(mess)
                continue
            tot_pts += nticks[i]
            inds = np.arange(ithis_eep, inext_eep + 1)

            if len(inds) <= 1:
                print(mess)
                print('skipping %s-%s: there are %i inds between these eeps.'
                    % (this_eep, next_eep, len(inds)))
                print(track.base, track.name)
                continue
            
            agenew, lnew, tenew = self.interpolate_along_track(track, inds,
                                                          nticks[i], mess=mess,
                                                          diag_plot=diag_plot)
            if type(agenew) is int:
                broken = True
                break
            logTe = np.append(logTe, tenew)
            logL = np.append(logL, lnew)
            Age = np.append(Age, agenew)

        if broken is True:
            print('problem during match interpolation. Skipping track.', mess)
            return

        #if diag_plot is True and self.debug is True:
            #pprint.pprint(track.info)
            #plt.show()

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
        #print('wrote %s' % outfile)
        self.match_data = to_write

    def interpolate_along_track(self, track, inds, nticks, diag_plot=False,
                                mess=None):
        '''
        interpolate along the track, check for age increasing.
        '''
        tckp = self._interpolate(track, inds, s=0, parafunc='np.log10')[0]
        if tckp == -1:        
            return -1, -1, -1

        arb_arr = np.linspace(0, 1, nticks + 1)
        lagenew, tenew, lnew = splev(arb_arr, tckp)
        test = np.diff(lagenew) > 0
        bads, = np.nonzero(test==False)
        # try again with lower spline level
        # xxx there should be a better way to do this!!!
        if False in test:
            track.info[mess] = 'Match interpolation by interp1d'
            from scipy.interpolate import interp1d
            lage = np.log10(track.data.AGE[inds])
            fage_l = interp1d(lage, track.data.LOG_L[inds],
                              bounds_error=0)
            fage_te = interp1d(lage, track.data.LOG_TE[inds],
                               bounds_error=0)
            lagenew = np.linspace(lage[0], lage[-1], nticks + 1)
            lnew = fage_l(lagenew)
            tenew = fage_te(lagenew)
            '''
            if False in test:
                track.flag = 'Age not monotonically increasing'
                print(track.base, track.name)
                print('Age not monotonically increasing', track.mass)
                print(mess)
                if len(inds) * 2 < nticks:
                    print('there are only %i inds this part of the track, probably overfitting with %i' % (len(inds), nticks))
                else:
                    #print('%i inds on the track, %i requested for MATCH' % (len(inds), nticks))
                    print('track ends with Sandro\'s %s' % self.ptcri.get_ptcri_name(len(track.sptcri)-1, sandro=True))
                    agediff = track.data.AGE[inds[-1]] - track.data.AGE[inds[0]]
                    tediff = track.data.LOG_TE[inds[-1]] - track.data.LOG_TE[inds[0]]
                    logldiff = track.data.LOG_L[inds[-1]] - track.data.LOG_L[inds[0]]
                    print('dAge %g dlog Te %g dlog L %g' % (agediff, tediff, logldiff))
                    import pdb; pdb.set_trace()
                    print(track.info)
                #print('%g: %g' % (10**agenew[bads[0]], 10**agenew[bads[-1]]))
                if diag_plot is True:
                    fig, (axs) = plt.subplots(ncols=2, figsize=(16, 10), sharey=True)
                    iptcri = track.iptcri[track.iptcri > 0]
                    for ax, xcol in zip(axs, ['AGE', 'LOG_TE']):
                        ax.plot(track.data[xcol], track.data.LOG_L, color='k')
                        ax.plot(track.data[xcol], track.data.LOG_L, ',', color='k')
                        if hasattr(track, 'data_orig'):
                            ax.plot(track.data_orig[xcol], track.data_orig.LOG_L,
                                    color='k', alpha=0.3, label='original track')
                            ax.plot(track.data_orig[xcol], track.data_orig.LOG_L,
                                    ',', color='k', alpha=0.3)
                        xlim = ax.get_xlim()
                        ylim = ax.get_ylim()

                        ax.set_xlabel('$%s$' % xcol.replace('_', r'\! '),
                                      fontsize=20)
                        ax.set_ylabel('$LOG\! L$', fontsize=20)
                        ax.scatter(track.data[xcol][iptcri],
                                   track.data.LOG_L[iptcri],
                                   s=60, c='k', label='crit pts')
                        [ax.annotate('%i' % i, (track.data[xcol][i],
                                                track.data.LOG_L[i]))
                         for i in iptcri]
                        ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    for ax, xcol in zip(axs, [10 ** agenew, tenew]):
                        ax.plot(xcol, lnew, lw=2, alpha=0.4, color='r')
                        ax.plot(xcol, lnew, '.', color='r', label='match intp')
                        ax.plot(xcol[bads], lnew[bads], 'o', color='b', label='bads')
                        [ax.annotate('%i' % i, (xcol[i], lnew[i])) for i in bads]

                    #axs[0].set_xscale('log')
                    fig.suptitle('$%s$' % track.name.replace('_', r'\! '))
                    plt.legend(loc=0)
                    plt.savefig(os.path.join(os.getcwd(), track.name + '_bad.png'))
                    plt.close('all')
            '''
        return 10 ** lagenew[:-1], lnew[:-1], tenew[:-1]