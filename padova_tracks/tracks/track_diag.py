'''
plotting and diagnostics track is always track object.
'''
from __future__ import print_function
import matplotlib.pylab as plt
import numpy as np
import utils
from copy import deepcopy
import os
from graphics.GraphicsUtils import arrow_on_line
from graphics.GraphicsUtils import setup_multiplot


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

            axs[i].set_xlabel('$%s$' % x.replace('_', r'\ '))
            axs[i].set_ylabel('$%s$' % y.replace('_', r'\ '))

            if annotate is True:
                self.annotate_plot(track, axs[i], xdata, y)

            if xr is True:
                axs[i].set_xlim(axs[i].get_xlim()[::-1])
            if yr is True:
                axs[i].set_ylim(axs[i].get_ylim()[::-1])
            axs[i].set_title('$%s$' %
                             track.name.replace('_', r'\ ').replace('.PMS', ''))

        return fig, axs

    def plot_track(self, track, xcol, ycol, reverse_x=False, reverse_y=False,
                   ax=None, inds=None, plt_kw={}, annotate=False, clean=False,
                   ainds=None, sandro=False, cmd=False, convert_mag_kw={},
                   xdata=None, ydata=None, hb=False, xnorm=False, ynorm=False,
                   arrow_on_line=False, yscale='linear', xscale='linear'):
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

        if xnorm is True:
            xdata /= np.max(xdata)

        if ynorm is True:
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

        if annotate:
            ax = self.annotate_plot(track, ax, xdata, ydata, inds=ainds,
                                    sandro=sandro, hb=hb, cmd=cmd)
        if arrow_on_line is True:
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

    def annotate_plot(self, track, ax, xcol, ycol, inds=None, sandro=False,
                      cmd=False, hb=False):
        '''
        if a subset of ptcri inds are used, set them in inds. If you want
        sandro's ptcri's sandro=True, will also change the face color of the
        label bounding box so you can have both on the same plot.
        '''
        if sandro is False:
            ptcri = track.iptcri
            fc = 'blue'
        else:
            ptcri = track.sptcri
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

    def check_ptcris(self, track, hb=False, plot_dir=None, sandro_plot=False,
                    xcol='LOG_TE', ycol='LOG_L'):
        '''
        plot of the track, the interpolation, with each eep labeled
        '''
        all_inds, = np.nonzero(track.data.AGE > 0.2)

        iptcri, = np.nonzero(track.iptcri > 0)
        ptcri_kw = {'sandro': False, 'hb': hb}
        last = track.ptcri.get_ptcri_name(int(iptcri[-1]), **ptcri_kw)

        if hb is False:
            plots = [['PMS_BEG', 'PMS_MIN', 'PMS_END', 'MS_BEG'],
                     ['MS_BEG', 'MS_TMIN', 'MS_TO', 'SG_MAXL', 'RG_MINL'],
                     ['RG_MINL', 'RG_BMP1', 'RG_BMP2', 'RG_TIP'],
                     ['RG_TIP', 'HE_BEG', 'YCEN_0.550', 'YCEN_0.500',
                      'YCEN_0.400'],
                     ['YCEN_0.400', 'YCEN_0.200', 'YCEN_0.100'],
                     ['YCEN_0.100', 'YCEN_0.000', 'TPAGB']]
        else:
            plots = [['HB_BEG', 'YCEN_0.500', 'YCEN_0.400', 'YCEN_0.200',
                     'YCEN_0.100', 'YCEN_0.005'],
                     ['YCEN_0.005', 'AGB_LY1', 'AGB_LY2']]

        for i, plot in enumerate(plots):
            if last in plot:
                nplots = i + 1

        line_pltkw = {'color': 'black'}
        point_pltkw = {'marker': 'o', 'ls': ''}
        (fig, axs) = setup_multiplot(nplots,
                                     subplots_kwargs={'figsize': (12, 8)})

        for i, ax in enumerate(np.ravel(axs)):
            if i == len(plots):
                continue
            inds = [track.ptcri.get_ptcri_name(cp, **ptcri_kw)
                    for cp in plots[i]]
            inds = track.iptcri[inds][np.nonzero(track.iptcri[inds])[0]]
            if np.sum(inds) == 0:
                continue

            ax = self.plot_track(track, xcol, ycol, ax=ax, inds=all_inds,
                                 reverse_x=True, plt_kw=line_pltkw)
            ax = self.plot_track(track, xcol, ycol, ax=ax, inds=inds,
                                 plt_kw=point_pltkw, annotate=True, ainds=inds,
                                 hb=hb)

            if hasattr(self, 'match_data'):
                # over plot the match interpolation
                for col in [xcol, ycol]:
                    if col == 'LOG_L':
                        tmp = (4.77 - self.match_data.T[3]) / 2.5
                    if 'age' in col.lower():
                        tmp = self.match_data.T[2]
                        #if not 'log' in col.lower():
                        #    tmp = 10 ** tmp
                    if col == 'LOG_TE':
                        tmp = self.match_data.T[2]
                    if col == xcol:
                        x = tmp
                    if col == ycol:
                        y = tmp

                ax.plot(x, y, lw=2, color='green')

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
            #ax.set_xlim(goodlimx)
            #ax.set_ylim(goodlimy)
            ax.set_xlabel('$%s$' % xcol.replace('_', r'\! '), fontsize=20)
            ax.set_ylabel('$%s$' % ycol.replace('_', r'\! '), fontsize=20)
            if 'age' in xcol:
                ax.set_xscale('log')
        title = 'M = %.3f Z = %.4f Y = %.4f' % (track.mass, track.Z, track.Y)
        fig.suptitle(title, fontsize=20)
        if hb is True:
            extra = '_HB'
        else:
            extra = ''
        
        extra += '_%s' % xcol

        figname = 'ptcri_Z%g_Y%g_M%.3f%s.png' % (track.Z, track.Y, track.mass,
                                                 extra)
        if plot_dir is not None:
            figname = os.path.join(plot_dir, figname)
        plt.savefig(figname)
        plt.close()
        print('wrote %s' % figname)

        if hb is False and sandro_plot is True:
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
