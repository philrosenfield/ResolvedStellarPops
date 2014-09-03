'''
plotting and diagnostics track is always track object.
'''
from __future__ import print_function
import matplotlib.pylab as plt
from matplotlib.ticker import MaxNLocator, NullFormatter
import numpy as np
from copy import deepcopy
import os
from ResolvedStellarPops.graphics.GraphicsUtils import arrow_on_line
from ResolvedStellarPops.graphics.GraphicsUtils import setup_multiplot
from ResolvedStellarPops.graphics.GraphicsUtils import discrete_colors
from ResolvedStellarPops import utils

from ..eep.critical_point import Eep

def offset_axlims(track, xcol, ycol, ax, inds=None):
    xmax, xmin = track.maxmin(xcol, inds=inds)
    ymax, ymin = track.maxmin(ycol, inds=inds)

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
    return ax


def plot_match(track, xcol, ycol, ax=None):
    '''plot match track'''
    if ax is None:
        fig, ax = plt.subplots()
    for col in [xcol, ycol]:
        if col == 'LOG_L':
            tmp = (4.77 - track.match_data.T[3]) / 2.5
        if 'age' in col.lower():
            tmp = track.match_data.T[2]
            if not 'log' in col.lower():
                tmp = 10 ** tmp
        if col == 'LOG_TE':
            tmp = track.match_data.T[2]
        if col == xcol:
            x = tmp
        if col == ycol:
            y = tmp
    ax.plot(x, y, lw=4, color='green', alpha=0.3)
    return ax
    
def quick_hrd(track, ax=None, inds=None, reverse='x'):
    '''
    make an hrd.
    written for interactive use (usually in pdb)
    '''
    plt.ion()
    if ax is None:
        plt.figure()
        ax = plt.axes()
        reverse = 'x'
        
    ax.plot(track.data.LOG_TE, track.data.LOG_L, color='k')
    if inds is not None:
        ax.plot(track.data.LOG_TE[inds], track.data.LOG_L[inds], 'o')
    
    if 'x' in reverse:
        ax.set_xlim(ax.get_xlim()[::-1])
    return ax

def check_eep_hrd(tracks, ptcri_loc, between_ptcris=[0, -2], sandro=True):
    from .track_set import TrackSet
    if type(tracks[0]) is str:
        from .track import Track
        tracks = [Track(t) for t in tracks]
    td = TrackDiag()
    ts = TrackSet()
    ts.tracks = tracks
    if not hasattr(tracks[0], 'sptcri'):
        ts._load_ptcri(ptcri_loc, sandro=True)
    if not hasattr(tracks[0], 'iptcri'):
        ts._load_ptcri(ptcri_loc, sandro=False)

    zs = np.unique([t.Z for t in tracks])
    axs = [plt.subplots()[1] for i in range(len(zs))]
    [axs[list(zs).index(t.Z)].set_title(t.Z) for t in tracks]
    
    for t in tracks:
        ax = axs[list(zs).index(t.Z)]
        plot_track(t, 'LOG_TE', 'LOG_L', sandro=sandro, ax=ax,
                   between_ptcris=between_ptcris, add_ptcris=True,
                   add_mass=True)

        ptcri_names = Eep().eep_list[between_ptcris[0]: between_ptcris[1] + 1]
        td.annotate_plot(t, ax, 'LOG_TE', 'LOG_L', ptcri_names=ptcri_names) 
        
    [ax.set_xlim(ax.get_xlim()[::-1]) for ax in axs]
    return ts, axs

def column_to_data(track, xcol, ycol, xdata=None, ydata=None, cmd=False,
                   convert_mag_kw={}, norm=''):
    '''
    convert a string column name to data
    
    returns xdata, ydata
    
    norm 'xy', 'x', 'y' for which or both axis to normalize
    can also pass xdata, ydata to normalize or if its a cmd
    '''
    if ydata is None:
        ydata = track.data[ycol]

    if xdata is None:
        if cmd:
            if len(convert_mag_kw) != 0:
                import astronomy_utils
                photsys = convert_mag_kw['photsys']
                dmod = convert_mag_kw.get('dmod', 0.)
                Av = convert_mag_kw.get('Av', 0.)
                Mag1 = track.data[xcol]
                Mag2 = track.data[ycol]
                avdmod = {'Av': Av, 'dmod': dmod}
                mag1 = astronomy_utils.Mag2mag(Mag1, xcol, photsys, **avdmod)
                mag2 = astronomy_utils.Mag2mag(Mag2, ycol, photsys, **avdmod)
                xdata = mag1 - mag2
                ydata = mag2
            else:
                xdata = track.data[xcol] - track.data[ycol]
        else:
            xdata = track.data[xcol]

    if 'x' in norm:
        xdata /= np.max(xdata)

    if 'y' in norm:
        ydata /= np.max(ydata)
        
    return xdata, ydata


def plot_track(track, xcol, ycol, reverse='',
               ax=None, inds=None, plt_kw={}, clean=False,
               sandro=False, cmd=False, convert_mag_kw={},
               xdata=None, ydata=None, norm='',
               arrows=False, yscale='linear', xscale='linear',
               ptcri_inds=False, add_ptcris=False, between_ptcris=[0, -1],
               add_mass=False):
    '''
    ainds is passed to annotate plot, and is to only plot a subset of crit
    points.
    sandro = True will plot sandro's ptcris.
    
    plot helpers:
    reverse 'xy', 'x', or 'y' will flip that axis
    ptcri_inds bool will annotate ptcri numbers
    add_ptcris will mark plot using track.iptcri or track.sptcri
    
    '''
    if type(track) == str:
        from .track import Track
        track = Track(track)

    if ax is None:
        plt.figure()
        ax = plt.axes()

    if len(plt_kw) != 0:
        # not sure why, but every time I send marker='o' it also sets
        # linestyle = '-' ...
        if 'marker' in plt_kw.keys():
            if not 'ls' in plt_kw.keys() or not 'linestyle' in plt_kw.keys():
                plt_kw['ls'] = ''

    if clean and inds is None:
        # non-physical inds go away.
        inds, = np.nonzero(track.data.AGE > 0.2)

    xdata, ydata = column_to_data(track, xcol, ycol, cmd=cmd, norm=norm,
                                  convert_mag_kw=convert_mag_kw)

    if inds is not None:
        ax.plot(xdata[inds], ydata[inds], **plt_kw)
    else:
        if not hasattr(track, 'sptcri') or not hasattr(track, 'iptcri'):
            ax.plot(xdata, ydata, **plt_kw)
        else:
            if sandro:
                iptcri = track.sptcri
            else:
                iptcri = track.iptcri
            pinds = np.arange(iptcri[between_ptcris[0]], iptcri[between_ptcris[1]])
            ax.plot(xdata[pinds], ydata[pinds], **plt_kw)

    if 'x' in reverse:
        ax.set_xlim(ax.get_xlim()[::-1])

    if 'y' in reverse:
        ax.set_ylim(ax.get_ylim()[::-1])

    if add_ptcris:
        # very simple ... use annotate for the fancy boxes
        if sandro:
            iptcri = track.sptcri
        else:
            iptcri = track.iptcri
        pinds = iptcri[between_ptcris[0]: between_ptcris[1] + 1]
        ax.plot(xdata[pinds], ydata[pinds], 'o', color='k')
        if ptcri_inds:
            [ax.annotate('%i' % i, (xdata[i], ydata[i])) for i in pinds]
    
    if add_mass:
        ax.annotate(r'$%g$' % track.mass, (xdata[iptcri[5]], ydata[iptcri[5]]),
                    fontsize=10)

    if arrows:
        # hard coded to be 10 equally spaced points...
        inds, = np.nonzero(track.data.AGE > 0.2)
        ages = np.linspace(np.min(track.data.AGE[inds]),
                           np.max(track.data.AGE[inds]), 10)
        indz, _ = zip(*[utils.closest_match(i, track.data.AGE[inds])
                        for i in ages])
        # I LOVE IT arrow on line... AOL BUHSHAHAHAHAHA
        aol_kw = deepcopy(plt_kw)
        if 'color' in aol_kw:
            aol_kw['fc'] = aol_kw['color']
            del aol_kw['color']
        indz = indz[indz > 0]
        print(track.data.LOG_L[inds][np.array([indz])])
        arrow_on_line(ax, xdata, ydata, indz, plt_kw=plt_kw)

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    return ax


class TrackDiag(object):
    '''a class for plotting tracks'''
    def __init__(self):
        pass

    def quick_hrd(self, *args):
        return quick_hrd(*args)
    
    def plot_track(self, *args, **kwargs):
        return plot_track(*args, **kwargs)
    
    def diag_plots(self, tracks, pat_kw=None, xcols=['LOG_TE', 'logAge'],
                   extra='', hb=False, mass_split='default', mextras=None,
                   plot_dir='.'):
                   
        '''
        pat_kw go to plot all tracks default:
            'eep_list': self.eep_list,
            'eep_lengths': self.nticks,
            'plot_dir': self.tracks_base
        xcols are the xcolumns to make individual plots
        mass_split is a list to split masses length == 3 (I'm lazy)
        extras is the filename extra associated with each mass split
           length == mass_split + 1
        '''

        if hasattr(self, 'prefix'):
            prefix = self.prefix
        else:
            prefix = os.path.split(tracks[0].base)[1]

        eep = Eep()
        default = {'eep_list': eep.eep_list, 'eep_lengths': eep.nticks}
        pat_kw = pat_kw or {}
        pat_kw = dict(default.items() + pat_kw.items())
        
        if hb:
            extra += '_hb'
            pat_kw['eep_lengths'] = eep.nticks_hb
        
        
        if mass_split == 'default':
            mass_split = [1, 1.4, 3, 12, 50]
            mextras = ['_lowest', '_vlow', '_low', '_inte', '_high', '_vhigh']
            tracks_split = [[t for t in tracks if t.mass <= mass_split[0]],
                            [t for t in tracks if t.mass >= mass_split[0]
                             and t.mass <= mass_split[1]],
                            [t for t in tracks if t.mass >= mass_split[1]
                             and t.mass <= mass_split[2]],
                            [t for t in tracks if t.mass >= mass_split[2]
                             and t.mass <= mass_split[3]],
                            [t for t in tracks if t.mass >= mass_split[3]
                             and t.mass <= mass_split[4]],
                            [t for t in tracks if t.mass >= mass_split[4]]]
        else:
            tracks_split = [tracks]
            mextras = ['']

        for i, ts in enumerate(tracks_split):
            if len(ts) == 0:
                continue

            for xcol in xcols:
                pat_kw['xcol'] = xcol
                ax = self.plot_all_tracks(ts, **pat_kw)
                ax.set_title('$%s$' % prefix.replace('_', '\ '))
                figname = 'diag_plot_%s_%s%s%s.png' % (prefix, xcol,
                                                        extra, mextras[i])
                figname = os.path.join(plot_dir, figname)
                plt.savefig(figname, dpi=300)
                plt.close('all')

    def plot_all_tracks(self, tracks, eep_list=None, eep_lengths=None,
                        xcol='LOG_TE', ycol='LOG_L', ax=None, ptcri=None):
        '''plot all tracks and annotate eeps'''
        if eep_lengths is not None:
            inds = np.insert(np.cumsum(eep_lengths), 0, 1)
        line_pltkw = {'color': 'black', 'alpha': 0.3}
        point_pltkw = {'marker': 'o', 'ls': '', 'alpha': 0.5}
        labs = [p.replace('_', '\_') for p in eep_list]
        if ax is None:
            fig, ax = plt.subplots(figsize=(16, 9))
        
        # plot colors
        cols = discrete_colors(len(eep_list) + 1, colormap='spectral')

        xlims = np.array([])
        ylims = np.array([])
        for t in tracks:
            if t.flag is not None:
                continue

            if ptcri is not None:
                try:
                    inds = ptcri.data_dict['M%.3f' % t.mass]
                except KeyError:
                    # mass not found
                    continue
                # skip undefined eeps
                inds = inds[inds > 0]

            xdata = t.data[xcol]
            ydata = t.data[ycol]
            
            # only eeps that are in the track
            inds = [eep for eep in inds if eep < len(xdata)]
            finds = np.arange(inds[0], inds[-1])
            ax.plot(xdata[finds], ydata[finds], **line_pltkw)
            for i in range(len(inds)):
                x = xdata[inds[i]]
                y = ydata[inds[i]]
                ax.plot(x, y, color=cols[i], **point_pltkw)
                xlims = np.append(xlims, (np.min(x), np.max(x)))
                ylims = np.append(ylims, (np.min(y), np.max(y)))
                if i == 5:
                    ax.annotate('%g' % t.mass, (x, y), fontsize=8)

        ax.set_xlim(np.max(xlims), np.min(xlims))
        ax.set_ylim(np.min(ylims), np.max(ylims))
        ax.set_xlabel('$%s$' % xcol.replace('_', '\! '), fontsize=20)
        ax.set_ylabel('$%s$' % ycol.replace('_', '\! '), fontsize=20)
        #ax.legend(loc=0, numpoints=1, frameon=0)
        return ax

    def annotate_plot(self, track, ax, xcol, ycol, ptcri_names=[],
                      sandro=False, hb=False, box=True, khd=False):
        '''
        if a subset of ptcri inds are used, set them in inds. If you want
        sandro's ptcri's sandro=True, will also change the face color of the
        label bounding box so you can have both on the same plot.
        '''
        eep = Eep()
        eep_list = eep.eep_list
            
        if hb:
            eep_list = eep.eep_list_hb

        if not sandro:
            fc = 'blue'
            iptcri = track.iptcri
        else:
            fc = 'red'
            iptcri = track.sptcri

        ptcri_kw = {'sandro': sandro, 'hb': hb}
        pts = [eep_list.index(i) for i in ptcri_names]
        inds = iptcri[pts]

        labs = ['$%s$' % p.replace('_', r'\ ') for p in ptcri_names]

        xdata, ydata = column_to_data(track, xcol, ycol)
        
        if box:
            # label stylings
            bbox = dict(boxstyle='round, pad=0.2', fc=fc, alpha=0.2)
            arrowprops = dict(arrowstyle='->', connectionstyle='arc3, rad=0')

        for i, (lab, x, y) in enumerate(zip(labs, xdata[inds], ydata[inds])):
            # varies the labels placement... default is 20, 20
            if khd:
                ax.vlines(x, 0, 1, label=lab)
                y = 0.75
            if box:
                xytext = ((-1.) ** (i - 1.) * 20, (-1.) ** (i + 1.) * 20)
                ax.annotate(lab, xy=(x, y), xytext=xytext, fontsize=10,
                            textcoords='offset points', ha='right', va='bottom',
                            bbox=bbox, arrowprops=arrowprops)
        return ax

    def check_ptcris(self, track, ptcri, hb=False, plot_dir=None,
                     sandro_plot=False, xcol='LOG_TE', ycol='LOG_L'):
        '''
        plot of the track, the interpolation, with each eep labeled
        '''
        if track.flag is not None:
            return

        all_inds, = np.nonzero(track.data.AGE > 0.2)
        iptcri = track.iptcri
        defined, = np.nonzero(iptcri > 0)
        ptcri_kw = {'sandro': False, 'hb': hb}
        last = ptcri.get_ptcri_name(len(defined) - 1, **ptcri_kw)
        if not hb:
            plots = [['PMS_BEG', 'PMS_MIN', 'PMS_END', 'MS_BEG'],
                     ['MS_BEG', 'MS_TMIN', 'MS_TO', 'SG_MAXL', 'RG_MINL'],
                     ['RG_MINL', 'RG_BMP1', 'RG_BMP2', 'RG_TIP'],
                     ['RG_TIP', 'HE_BEG', 'YCEN_0.550', 'YCEN_0.500',
                      'YCEN_0.400'],
                     ['YCEN_0.400', 'YCEN_0.200', 'YCEN_0.100'],
                     ['YCEN_0.100', 'YCEN_0.005', 'YCEN_0.000', 'TPAGB']]
        else:
            plots = [['HB_BEG', 'YCEN_0.500', 'YCEN_0.400', 'YCEN_0.200',
                     'YCEN_0.100', 'YCEN_0.005'],
                     ['YCEN_0.005', 'TPAGB']]

        for i, plot in enumerate(plots):
            if last in plot:
                nplots = i + 2

        line_pltkw = {'color': 'black'}
        point_pltkw = {'marker': 'o', 'ls': ''}
        fig, axs = setup_multiplot(nplots, subplots_kwargs={'figsize': (12, 8)})
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, wspace=0.1)
        for i, ax in enumerate(axs.ravel()[:nplots]):
            self.plot_track(track, xcol, ycol, ax=ax, inds=all_inds,
                            reverse='x', plt_kw=line_pltkw)
            self.plot_track(track, xcol, ycol, ax=ax, inds=iptcri[iptcri>0],
                            plt_kw=point_pltkw)

            if hasattr(track, 'match_data'):
                # overplot the match interpolation
                ax = plot_match(track, xcol, ycol, ax=ax)
            
            if i < nplots - 1:
                ainds = [ptcri.get_ptcri_name(cp, **ptcri_kw) for cp in plots[i]]
                inds = iptcri[ainds][np.nonzero(iptcri[ainds])[0]]
                #ainds = track.iptcri[track.iptcri>0]
                if np.sum(inds) == 0:
                    continue
                self.annotate_plot(track, ax, xcol, ycol, ptcri_names=plots[i])
                ax = offset_axlims(track, xcol, ycol, ax, inds=inds)
            else:
                ax = offset_axlims(track, xcol, ycol, ax)           

            if 'age' in xcol:
                ax.set_xscale('log')

        
        [axs[-1][i].set_xlabel('$%s$' % xcol.replace('_', r'\ '), fontsize=16)
         for i in range(np.shape(axs)[1])]
        [axs[i][0].set_ylabel('$%s$' % ycol.replace('_', r'\ '), fontsize=16)
         for i in range(np.shape(axs)[0])]
        
        title = 'M = %.3f Z = %.4f Y = %.4f' % (track.mass, track.Z, track.Y)
        fig.suptitle(title, fontsize=20)
        extra = ''
        if hb:
            extra += '_HB'
        extra += '_%s' % xcol

        figname = 'ptcri_Z%g_Y%g_M%.3f%s.png' % (track.Z, track.Y, track.mass,
                                                 extra)
        if plot_dir is not None:
            figname = os.path.join(plot_dir, figname)
        plt.savefig(figname)
        plt.close()

        if not hb and sandro_plot:
            self.plot_sandro_ptcri(track, plot_dir=plot_dir)
        return axs
    
    def plot_sandro_ptcri(self, track, plot_dir=None, ptcri=None):
        ax = self.plot_track(track, 'LOG_TE', 'LOG_L', reverse='x',
                             inds=np.nonzero(track.data.AGE > 0.2)[0])

        ax = self.plot_track(track, 'LOG_TE', 'LOG_L', ax=ax, annotate=True,
                             sandro=True, ptcri=ptrci)
        title = 'M = %.3f Z = %.4f Y = %.4f' % (track.mass, track.Z, track.Y)
        ax.set_title(title, fontsize=20)
        figname = 'sandro_ptcri_Z%g_Y%g_M%.3f.png' % (track.Z, track.Y,
                                                      track.mass)
        if plot_dir is not None:
            figname = os.path.join(plot_dir, figname)
        plt.savefig(figname)
        plt.close()
        return

    def kippenhahn(self, track, col_keys=None, heb_only=True, ptcri=None):
        track.calc_core_mu()

        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(8, 8))
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(8, 2)
        sm_axs = [plt.subplot(gs[i, 0:]) for i in range(4)]
        ax2 = plt.subplot(gs[4:, 0:])

        #ax3 = plt.subplot(gs[0:, -1])
        if heb_only:
            # Core HeB:
            inds, = np.nonzero((track.data.LY > 0) & (track.data.QHE1 == 0))
        else:
            inds = np.arange(len(track.data.LY))
        # AGE IN Myr
        xdata = track.data.AGE[inds]/1e6

        ycols = ['LOG_TE', '', 'LOG_RHc', 'LOG_Pc']
        ycolls = ['$\log T_{eff}$', '$\mu_c$', '$\\rho_c$', '$\log P_c$']

        for smax, ycol, ycoll in zip(sm_axs, ycols, ycolls):
            if len(ycol) == 0:
                ydata = track.muc[inds]
            else:
                ydata = track.data[ycol][inds]
                smax.plot(xdata, ydata, lw=3, color='black', label=ycoll)

            smax.plot(xdata, ydata, lw=3, color='black', label=ycoll)
            smax.set_ylabel('$%s$' % ycoll)
            smax.set_ylim(np.min(ydata), np.max(ydata))
            smax.yaxis.set_major_locator(MaxNLocator(4))
            smax.xaxis.set_major_formatter(NullFormatter())

        # discontinuities in conv...
        p1 = np.argmin((np.diff(track.data.CF1[inds])))
        p2 = np.argmax(np.diff(track.data.CF1[inds]))
        zo = 1
        ax2.fill_between(xdata[:p1], track.data.CI1[inds[:p1]],
                         track.data.CF1[inds[:p1]],
                         where=track.data.CF1[inds[:p1]]>0.2,
                         color='grey', alpha=0.4, zorder=zo)
        ax2.fill_between(xdata[p2:], track.data.CI1[inds[p2:]],
                         track.data.CF1[inds[p2:]],
                         where=track.data.CF1[inds[p2:]]<0.2,
                         color='grey', alpha=0.4, zorder=zo)
        ax2.fill_between(xdata, track.data.CI2[inds], track.data.CF2[inds],
                         color='grey', alpha=0.4)
        ax2.fill_between(xdata, track.data.QH1[inds], track.data.QH2[inds],
                         color='navy', label='$H$', zorder=zo)
        ax2.fill_between(xdata, track.data.QHE1[inds], track.data.QHE2[inds],
                         color='darkred', label='$^4He$', zorder=zo)

        zo = 100
        rel_cols = ['XC_cen', 'XO_cen', 'YCEN', 'LX', 'LY', 'CONV']
        rel_collss = [[''] * 6, ['$^{12}C$', '$^{16}O$', '$Y_c$','$L_X$', '$L_Y$', '$core$']]
        rcolss = [['white'] * 6, ['green', 'purple', 'darkred', 'navy', 'darkred', 'black']]
        lws = [5, 2]
        rlsss = [['-'] * 6, ['-', '-', '-', '--', '--', '-']]
        for rcols, rlss, rel_colls, lw in zip(rcolss, rlsss, rel_collss, lws):
            zo += 10
            for rel_col, rel_coll, rcol, rls in zip(rel_cols, rel_colls, rcols, rlss):
                ax2.plot(xdata, track.data[rel_col][inds], ls=rls, lw=3, color=rcol,
                         label=rel_coll, zorder = zo)

        itmax = p1 + np.argmax(track.data.LOG_TE[inds[p1:]])
        for ax in np.concatenate([[ax2], sm_axs]):
            ax.set_xlim(xdata[0], xdata[-1])
            ylim = ax.get_ylim()
            [ax.vlines(xdata[i], *ylim, color='grey', lw=2)
                       for i in [p1, itmax]]
            ax.set_ylim(ylim)
            #ax.set_xlim(track.data.AGE[track.ptcri.sptcri[11]], track.data.AGE[track.ptcri.sptcri[13]])


        ax2.legend(frameon=False, loc=6)
        ax2.set_ylim(0, 1)
        #ax3.set_xlim(np.max(track.data.LOG_TE[inds]), np.min(track.data.LOG_TE[inds]))
        #ax3.set_ylim(np.min(track.data.LOG_L[inds]), np.max(track.data.LOG_L[inds]))
        ax2.set_xlabel('$Age\ (Myr)$')
        #ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        #ax2.ticklabel_format(style='sci', scilimits=(0,0), axis='x')
        #sm_axs[0].set_title(os.path.split(track.base)[1].replace('_', '\ ').replace('PH',''))
        #ax1.set_ylabel('H Shell Abundances')
        ax2.set_ylabel('$m/M\ or\ f/f_{tot}$')
        self.annotate_plot(track, ax2, xdata, xdata, ptcri=ptcri, sandro=True,
                           khd=True)
        #ax3.set_xlabel('LOG TE')
        #ax3.set_ylabel('LOG L')
        #ax3.set_title('$M=%.4f$' % track.mass)
        #plt.savefig('%s_%s_khd.png' % (track_set, track.name), dpi=300)
        #fig.subplots_adjust(hspace=0)
        #ax.plot(track.data.AGE[inds], track.data['LOG_TE'][inds]/np.max(track.data['LOG_TE'][inds]), lw=3, color='black')
        #ax.plot(track.data.AGE[inds], track.data['CONV'][inds], lw=3, color='grey')
        #ax.plot(track.data.AGE[inds], track.data['MU'][inds], lw=3, color='brown')
        return fig, (ax1, ax2, ax3)
