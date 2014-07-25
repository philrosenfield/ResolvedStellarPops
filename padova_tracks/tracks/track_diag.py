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


class TrackDiag(object):
    '''a class for plotting tracks'''
    def __init__(self):
        pass

    def plot_track(self, track, xcol, ycol, reverse_x=False, reverse_y=False,
                   ax=None, inds=None, plt_kw={}, annotate=False, clean=False,
                   ainds=None, sandro=False, cmd=False, convert_mag_kw={},
                   xdata=None, ydata=None, hb=False, xnorm=False, ynorm=False,
                   arrows=False, yscale='linear', xscale='linear',
                   ptcri_inds=False, ptcri=None, add_ptcris=False):
        '''
        ainds is passed to annotate plot, and is to only plot a subset of crit
        points.
        sandro = True will plot sandro's ptcris.
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
            # Non physical inds go away.
            inds, = np.nonzero(track.data.AGE > 0.2)

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

        if xnorm:
            xdata /= np.max(xdata)

        if ynorm:
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

        if add_ptcris:
            iptcri = track.iptcri
            if sandro:
                iptcri = track.sptcri
            ax.plot(xdata[iptcri], ydata[iptcri], 'o', color='k')
            if ptcri_inds:
                [ax.annotate('%i' % i, (xdata[i], ydata[i])) for i in iptcri]
            
        if annotate:
            ax = self.annotate_plot(track, ax, xdata, ydata, ptcri=ptcri,
                                    inds=ainds, sandro=sandro, hb=hb, cmd=cmd)
        if arrows:
            # hard coded to be 10 equally spaced points...
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

    def diag_plots(self, pat_kw=None, xcols=['LOG_TE', 'logAge'], hb=False,
                   mass_split='default', mextras=None):
                   
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
        mextras = ['', '', '', '', '', '']
        if mass_split == 'default':
            mass_split = [1, 1.4, 3, 12, 50]
            if mextras is None:
                mextras = ['lowest', 'vlow', 'low', 'inte', 'high', 'vhigh']
            
        pat_kw = pat_kw or {}
        eep = Eep()
        default = {'eep_list': eep.eep_list,
                   'eep_lengths': eep.nticks,
                   'plot_dir': self.tracks_base,
                   'extra': ''}
        
        pat_kw = dict(default.items() + pat_kw.items())
        orig_extra = pat_kw['extra']
        if hb:
            tracks = self.hbtracks
            orig_extra += '_hb'
            pat_kw['eep_lengths'] = eep.nticks_hb
        else:
            tracks = self.tracks
        
        if mass_split is None:
            tracks_split = [tracks]
            mextras = ['']
        else:
            # could be done a lot better and faster:
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

        for i, ts in enumerate(tracks_split):
            if len(ts) == 0:
                continue
            pat_kw['extra'] = '_' + '_'.join([orig_extra, mextras[i]])
            for xcol in xcols:
                pat_kw['xcol'] = xcol
                self.plot_all_tracks(ts, **pat_kw)

        plt.close('all')

    def plot_all_tracks(self, tracks, eep_list=None, eep_lengths=None,
                         plot_dir=None, extra='', xcol='LOG_TE', ycol='LOG_L',
                         ax=None, ptcri=None):
        '''plot all tracks and annotate eeps'''
        extra += '_%s' % xcol
        if eep_lengths is not None:
            inds = np.insert(np.cumsum(eep_lengths), 0, 1)
        line_pltkw = {'color': 'black', 'alpha': 0.3}
        point_pltkw = {'marker': 'o', 'ls': '', 'alpha': 0.5}
        labs = [p.replace('_', '\_') for p in eep_list]
        if ax is None:
            fig, ax = plt.subplots(figsize=(16, 9))
        # fake lengend
        cols = discrete_colors(len(eep_list) + 1, colormap='spectral')
        #[ax.plot(9999, 9999, color=cols[i], label=labs[i], **point_pltkw)
        # for i in range(len(eep_list))]

        # instead, plot the tracks with alternating alpha for clarity
        #[ax.plot(t[xcol], t[ycol], **line_pltkw) for t in tracks[::2]]
        #line_pltkw['alpha'] = 0.8
        #[ax.plot(t[xcol], t[ycol], **line_pltkw) for t in tracks[1::2]]
        xlims = np.array([])
        ylims = np.array([])

        for t in tracks:
            if t.flag is not None:
                continue

            xdata = t.data[xcol]
            ydata = t.data[ycol]

            if ptcri is not None:
                try:
                    inds = ptcri.data_dict['M%.3f' % t.mass]
                except KeyError:
                    print('no %.3f in ptcri data' % t.mass)
                    continue
                inds = inds[inds > 0]

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
        
        if hasattr(self, 'prefix'):
            ax.set_title('$%s$' % self.prefix.replace('_', '\ '))
            figname = '%s%s.png' % (self.prefix, extra)
        else:
            figname = 'diag_plot%s.png' % extra
        
        ax.set_xlim(np.max(xlims), np.min(xlims))
        ax.set_ylim(np.min(ylims), np.max(ylims))
        ax.set_xlabel('$%s$' % xcol.replace('_', '\! '), fontsize=20)
        ax.set_ylabel('$%s$' % ycol.replace('_', '\! '), fontsize=20)
        #ax.legend(loc=0, numpoints=1, frameon=0)
        
        if plot_dir is not None:
            figname = os.path.join(plot_dir, figname)
        plt.savefig(figname, dpi=300)

    def annotate_plot(self, track, ax, xcol, ycol, ptcri=None, inds=None,
                      sandro=False, cmd=False, hb=False, box=True, khd=False):
        '''
        if a subset of ptcri inds are used, set them in inds. If you want
        sandro's ptcri's sandro=True, will also change the face color of the
        label bounding box so you can have both on the same plot.
        '''
        if not sandro:
            fc = 'blue'
            iptcri = track.iptcri
        else:
            fc = 'red'
            iptcri = track.sptcri
        
        ptcri_kw = {'sandro': sandro, 'hb': hb}
        if inds is None:
            inds = iptcri
            labels = ['$%s$' %
                      ptcri.get_ptcri_name(i, **ptcri_kw).replace('_', r'\ ')
                      for i in range(len(inds))]
        else:
            iplace = np.array([np.nonzero(iptcri == i)[0][0] for i in inds])
            labels = ['$%s$' %
                      ptcri.get_ptcri_name(int(i), **ptcri_kw).replace('_', r'\ ')
                      for i in iplace]

        if type(xcol) == str:
            xdata = track.data[xcol]
        else:
            xdata = xcol

        if type(ycol) == str:
            ydata = track.data[ycol]
        elif khd:
            ydata = xdata
        else:
            ydata = ycol

        if cmd:
            xdata = xdata - ydata
        
        if box:
            # label stylings
            bbox = dict(boxstyle='round, pad=0.5', fc=fc, alpha=0.5)
            arrowprops = dict(arrowstyle='->', connectionstyle='arc3, rad=0')

        for i, (label, x, y) in enumerate(zip(labels, xdata[inds],
                                              ydata[inds])):
            # varies the labels placement... default is 20, 20
            if khd:
                ax.vlines(x, 0, 1, label=label)
                y = 0.75
            if box:
                xytext = ((-1.) ** (i - 1.) * 20, (-1.) ** (i + 1.) * 20)
                ax.annotate(label, xy=(x, y), xytext=xytext, fontsize=10,
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
        defined, = np.nonzero(iptcri>0)
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
                nplots = i + 1
        nplots += 1

        line_pltkw = {'color': 'black'}
        point_pltkw = {'marker': 'o', 'ls': ''}
        fig, axs = setup_multiplot(nplots, subplots_kwargs={'figsize': (12, 8)})

        for i, ax in enumerate(np.ravel(axs)):
            if i < len(plots):
                inds = [ptcri.get_ptcri_name(cp, **ptcri_kw) for cp in plots[i]]
                inds = iptcri[inds][np.nonzero(iptcri[inds])[0]]          
                if np.sum(inds) == 0:
                    continue
                
            ax = self.plot_track(track, xcol, ycol, ax=ax, inds=all_inds,
                                 reverse_x=True, plt_kw=line_pltkw)
            if i < len(plots):
                ax = self.plot_track(track, xcol, ycol, ax=ax, inds=inds,
                                     plt_kw=point_pltkw, annotate=True,
                                     ainds=inds, hb=hb, ptcri=ptcri)

            if hasattr(track, 'match_data'):
                # overplot the match interpolation
                ax = plot_match(track, xcol, ycol, ax=ax)

            if i < len(plots):
                ax = offset_axlims(track, xcol, ycol, ax, inds=inds)
            else:
                ax = offset_axlims(track, xcol, ycol, ax)

            ax.set_xlabel('$%s$' % xcol.replace('_', r'\! '), fontsize=20)
            ax.set_ylabel('$%s$' % ycol.replace('_', r'\! '), fontsize=20)
            if 'age' in xcol:
                ax.set_xscale('log')

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

    def plot_sandro_ptcri(self, track, plot_dir=None, ptcri=None):
        ax = self.plot_track(track, 'LOG_TE', 'LOG_L', reverse_x=1,
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
