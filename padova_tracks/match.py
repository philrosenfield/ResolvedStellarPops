from fileio import fileIO
import numpy as np
import matplotlib.pylab as plt
from graphics.GraphicsUtils import discrete_colors
import os
from eep.define_eep import DefineEeps
from eep.define_eep import eep
from tracks.track_set import TrackSet
from scipy.interpolate import splev
from tracks.track_diag import TrackDiag


class MatchTracks(eep):
    '''
    a simple check of the output from TracksForMatch. I want it to run on the
    same input file as TracksForMatch.
    '''
    def __init__(self, inputs):
        eep.__init__(self)
        self.tracks_base = inputs.outfile_dir
        self.prefix = inputs.prefix
        # hard coding where the match files are kept tracks_base/match/
        all_track_names = fileIO.get_files(self.tracks_base,
                                           inputs.track_search_term.replace('PMS', ''))

        self.hbtrack_names = [t for t in all_track_names if 'HB' in t]
        self.track_names = [t for t in all_track_names
                            if t not in self.hbtrack_names]

        self.load_tracks()

    def load_tracks(self):
        self.tracks = [self._load_track(t) for t in self.track_names]
        self.hbtracks = [self._load_track(t) for t in self.hbtrack_names]

    def check_tracks(self):
        for i, t in enumerate(self.tracks):
            test = np.diff(t['logAge']) > 0
            if False in test:
                bads, = np.nonzero(np.diff(t['logAge']) < 0)
                bads1, = np.nonzero(np.diff(t['logAge']) == 0)
                if len(bads) != 0:
                    print 'Age not monotonicly increasing!'
                    print 'Track Name, in phase or before), Lage of bad inds'
                    #print self.track_names[i], bads, t['logAge'][bads]
                    bad_inds = np.unique([np.nonzero(j - np.cumsum(self.eep_lengths) < 0)[0][0]
                                          for j in bads])
                    print self.track_names[i], self.eep_list[bad_inds], t['logAge'][bads]

                if len(bads1) != 0:
                    print '%i identical age values in %s' % (len(bads1),
                                                             self.track_names[i])
                    bad_inds = np.unique([np.nonzero(j - np.cumsum(self.eep_lengths) < 0)[0][0]
                                          for j in bads1])
                    print 'check out %s and the one before.' % self.eep_list[bad_inds]
            else:
                print 'no issues found.'

    def diag_plots(self):
        pat_kw = {'eep_list': self.eep_list,
                  'eep_lengths': self.eep_lengths,
                  'plot_dir': self.outfile_dir}

        self._plot_all_tracks(self.tracks, **pat_kw)

        pat_kw['xcol'] = 'logAge'
        self._plot_all_tracks(self.tracks, **pat_kw)

        if self.eep_list_hb is not None:
            pat_kw['extra'] = '_HB'
            self._plot_all_tracks(self.hbtracks, **pat_kw)
            del pat_kw['xcol']
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
        cols = discrete_colors(len(eep_list), colormap='spectral')
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
                if len(np.nonzero(track.iptcri>0)[0]) < 3:
                    print 'skipping track because there is no ms_beg.', track.base, track.name
                    continue
                # make match output files.
                self.prepare_track(track, outfile_dir=inputs.outfile_dir)

                if inputs.diag_plot is True:
                    # make diagnostic plots
                    self.check_ptcris(track, plot_dir=inputs.plot_dir)
                    self.check_ptcris(track, plot_dir=inputs.plot_dir,
                                      xcol='AGE')

            if inputs.diag_plot is True:
                # make summary diagnostic plots
                self.plot_all_tracks(self.tracks, 'LOG_TE', 'LOG_L',
                                     sandro=False, reverse_x=True,
                                     plot_dir=inputs.plot_dir)

        else:
            print('Only doing HB.')

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
            if inputs.diag_plot is True:
                self.plot_all_tracks(self.hbtracks, 'LOG_TE', 'LOG_L',
                                     hb=inputs.hb, reverse_x=True,
                                     plot_dir=inputs.plot_dir, sandro=False)
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
            print('using default spacing between eeps')
            nticks = np.repeat(200, len(track.iptcri) - 1)

        assert nticks is not None, 'invalid eep_lengths, check eep list.'

        logTe = np.array([])
        logL = np.array([])
        Age = np.array([])
        new_eep_dict = {}
        tot_pts = 0
        ptcri_kw = {'sandro': False, 'hb': hb}
        for i in range(len(np.nonzero(track.iptcri >= 0)[0]) - 1):
            this_eep = track.ptcri.get_ptcri_name(i, **ptcri_kw)
            next_eep = track.ptcri.get_ptcri_name(i+1, **ptcri_kw)
            ithis_eep = track.iptcri[i]
            inext_eep = track.iptcri[i+1]
            mess = '%.3f %s=%i %s=%i' % (track.mass,
                                         this_eep, ithis_eep,
                                         next_eep, inext_eep)

            if i != 0 and track.iptcri[i+1] == 0:
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
                print(mess)
                print(
                    'skipping %s-%s cause there are no inds between these crit pts.'
                    % (this_eep, next_eep))
                continue

            if len(inds) == 1:
                # include the last ind.
                #inds = np.arange(ithis_eep, inext_eep + 1)
                print mess
                print 'skipping %s-%s cause there are 1 ind between these crit pts.' \
                      % (this_eep, next_eep)
                print track.base, track.name
                continue

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
                print track.base, track.name
                print '\n AGE NOT MONOTONICALLY INCREASING', track.mass
                print 10**agenew[bads]
                print mess
                fig, (axs) = plt.subplots(ncols=2, figsize=(16, 10))
                for ax, xcol in zip(axs, ['AGE', 'LOG_TE']):
                    ax.scatter(track.data[xcol][track.iptcri],
                               track.data.LOG_L[track.iptcri],
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
        print('wrote %s' % outfile)
        self.match_data = to_write
