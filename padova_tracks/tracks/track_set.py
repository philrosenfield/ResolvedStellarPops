'''
a container of track
'''
from __future__ import print_function
import os
import matplotlib.pylab as plt
import numpy as np

from ResolvedStellarPops.graphics.GraphicsUtils import discrete_colors
from ResolvedStellarPops.fileio import fileIO

from .track import Track
from .track_diag import TrackDiag
from ..eep.critical_point import critical_point

max_mass = 120.
td = TrackDiag()

class TrackSet(object):
    '''
    A class to load multiple track instances at once.
    '''
    def __init__(self, inputs=None):
        if inputs is None:
            self.prefix = ''
            return
        self.prefix = inputs.prefix

        if inputs.match is False:
            self.tracks_base = os.path.join(inputs.tracks_dir, self.prefix)
        else:
            self.tracks_base = inputs.outfile_dir
            inputs.track_search_term = \
                                inputs.track_search_term.replace('PMS', 'dat')

        if inputs.hb_only is False:
            self.find_tracks(track_search_term=inputs.track_search_term,
                             masses=inputs.masses, match=inputs.match)

        if inputs.hb is True:
            self.find_tracks(track_search_term=inputs.hbtrack_search_term,
                             masses=inputs.masses, match=inputs.match,
                             hb=True)
        else:
            self.hbtrack_names = []
            self.hbtracks = []
            self.hbmasses = []
            
    def find_tracks(self, track_search_term='*F7_*PMS', masses=None, hb=False,
                    match=False):
        '''
        loads tracks or hb tracks and their masses as attributes
        can load subset if masses (list, float, or string) is set.
        If masses is string, it must be have format like:
        '%f < 40' and it will use masses that are less 40.
        '''
        track_names = np.array(fileIO.get_files(self.tracks_base,
                               track_search_term))

        assert len(track_names) != 0, \
            'No tracks found: %s/%s' % (self.tracks_base, track_search_term)
        ext = '.' + fileIO.split_file_extention(track_names[0])[1]
        mass = np.array([os.path.split(t)[1].split('_M')[1].split(ext)[0]
                         for t in track_names], dtype=float)
        cut_mass, = np.nonzero(mass <= max_mass)
        track_names = track_names[cut_mass][np.argsort(mass[cut_mass])]
        mass = mass[cut_mass][np.argsort(mass[cut_mass])]

        # only do a subset of masses
        if masses is not None:
            if type(masses) == float:
                inds = [masses]
            elif type(masses) == str:
                inds = [i for i in range(len(mass)) if eval(masses % mass[i])]
            if type(masses) == list:
                inds = np.array([])
                for m in masses:
                    try:
                        inds = np.append(inds, list(mass).index(m))
                    except ValueError:
                        # this mass is missing
                        pass
        else:
            inds = np.argsort(mass)
        
        track_str = 'track'
        mass_str = 'masses'

        if hb is True:
            track_str = 'hb%s' % track_str
            mass_str = 'hb%s' % mass_str

        track_attr = '%ss' % track_str
        self.__setattr__('%s_names' % track_str, track_names[inds])
        self.__setattr__(track_attr, [Track(track, match=match)
                                      for track in track_names[inds]])
        self.__setattr__('%s' % mass_str,
                         ['%.3f' % t.mass for t in self.__getattribute__(track_attr)])
        return


    def plot_all_tracks(self, tracks, xcol, ycol, annotate=True, ax=None,
                        reverse_x=False, sandro=True, cmd=False,
                        convert_mag_kw={}, hb=False, plot_dir=None,
                        zoomin=True, one_plot=False, cols=None,
                        td_kw={}):
        '''
        should go in TrackDiag??
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

        if one_plot is True:
            fig_extra = 'one'
            plots = [list(np.unique(np.concatenate(np.array(plots))))]

        xlims = np.array([])
        ylims = np.array([])
        for j in range(len(plots)):
            fig, ax = plt.subplots()
            if annotate is True:
                point_pltkw = {'marker': '.', 'ls': '', 'alpha': 0.5}
                cols = discrete_colors(len(plots[j]), colormap='spectral')
                labs = ['$%s$' % p.replace('_', '\_') for p in plots[j]]

            didit = 0
            xlimi = np.array([])
            ylimi = np.array([])
            for t in tracks:
                if t.flag is not None:
                    continue

                if sandro is False:
                    ptcri = t.iptcri
                else:
                    ptcri = t.sptcri

                ainds = [self.ptcri.get_ptcri_name(cp, **ptcri_kw)
                         for cp in plots[j]]

                ainds = [i for i in ainds if i < len(ptcri)]

                inds = ptcri[ainds]
                inds = np.squeeze(inds)

                if np.sum(inds) == 0:
                    noplot = True
                    continue
                else:
                    noplot = False

                some_inds = np.arange(inds[0], inds[-1])

                ax = td.plot_track(t, xcol, ycol, ax=ax, inds=some_inds,
                                   plt_kw=line_pltkw, cmd=cmd, clean=False,
                                   convert_mag_kw=convert_mag_kw, **td_kw)

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
                    if one_plot is True:
                        plines = pls

            if zoomin is True and noplot is False:
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
            #print('wrote %s' % figname)
        return

    def all_inds_of_eep(self, eep_name):
        '''
        get all the ind for all tracks of some eep name, for example
        want ms_to of the track set? set eep_name = point_c if sandro==True.
        '''
        inds = []
        for track in self.tracks:
            check = self.ptcri.load_sandro_eeps(track)
            if check == -1:
                inds.append(-1)
                continue
            eep_ind = self.ptcri.get_ptcri_name(eep_name)
            if len(track.sptcri) <= eep_ind:
                inds.append(-1)
                continue
            data_ind = track.sptcri[eep_ind]
            inds.append(data_ind)
        return inds
