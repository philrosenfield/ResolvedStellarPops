import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import ResolvedStellarPops.match_utils as match_utils

def add_inner_title(ax, title, loc, size=None, **kwargs):
    from matplotlib.offsetbox import AnchoredText
    from matplotlib.patheffects import withStroke
    if size is None:
        size = dict(size=plt.rcParams['legend.fontsize'])
    at = AnchoredText(title, loc=loc, prop=size,
                      pad=0., borderpad=0.5,
                      frameon=False, **kwargs)
    ax.add_artist(at)
    at.txt._text.set_path_effects([withStroke(foreground="w", linewidth=3)])
    return at

    
def match_plot(ZS, extent, labels=["Data", "Model", "Diff","Sig"], **kwargs):
    '''
    ex ZS = [h[2],sh[2],diff_cmd,resid]
    '''
    fig = plt.figure(figsize = (9,9))
    grid = ImageGrid(fig,111, # similar to subplot(111)
                    nrows_ncols = (2, 2),
                    direction="row",
                    axes_pad = 0.23,
                    add_all=True,
                    label_mode = "1",
                    share_all = True,
                    cbar_location="top",
                    cbar_mode="each",
                    cbar_size="7%",
                    cbar_pad="2%",
                    aspect=0
                    )
    #PuBu
    #1. find max abs value of diff array (and also sig...)
    #2. use normalize instance and pass vmin=-abs to vmax=abs 
    #normed_floats = norm_instance(float_array)
    #colors = matplotlib.cm.RdBu(normed_floats) # or any cmap really.

    for i, (ax, z) in enumerate(zip(grid, ZS)):
        if i > 1:
            # second row: make 0 on the color bar white
            vmin =  - np.max(np.abs(z))
            vmax = np.max(np.abs(z))
            colors = cm.RdBu
        else:
            # first row: make white 0, but will be on the left of color bar
            vmin = None
            vmax = None
            colors = cm.bone_r
        im = ax.imshow(z, extent=extent, interpolation="nearest",
                       cmap=colors, vmin=vmin, vmax=vmax)
        ax.cax.colorbar(im)
        forceAspect(ax,aspect=1)

    ylabel = kwargs.get('ylabel')
    xlabel = kwargs.get('xlabel')
    if ylabel:
        [ax.set_ylabel(ylabel, fontsize=20) for ax in grid[0::2]]
    if xlabel:
        [ax.set_xlabel(xlabel, fontsize=20) for ax in grid[2:]]
    
    for ax, im_title in zip(grid, labels):
        t = add_inner_title(ax, im_title, loc=1)
        t.patch.set_alpha(0.5)

    return grid

def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

def pgcmd(filename, labels=None, saveplot=False, out_dir=None, axis_labels='default'):
    '''
    produces the image that pgcmd.pro makes
    '''
    cmd = match_utils.read_match_cmd(filename)
    if axis_labels.lower() == 'default':
        import re
        print filename
        t = re.search('_(.*)/(.*)_(.*)_(.{5})_(.{5}).(.*)', filename)
        (__, propidtarget, filter1, filter2, __, __) = t.groups()
        kwargs = {'xlabel': r'$%s-%s$' % (filter1, filter2),
                  'ylabel': r'$%s$' % filter2}
        labels = [r'$%s$' % propidtarget.replace('_', '\ '),
                  'Model', 'Diff', 'Sig']

    nmagbin = len(np.unique(cmd['mag']))
    ncolbin = len(np.unique(cmd['color']))
    h = cmd['Nobs'].reshape(nmagbin, ncolbin)
    s = cmd['Nsim'].reshape(nmagbin, ncolbin)
    diff = cmd['diff'].reshape(nmagbin, ncolbin)
    sig = cmd['sig'].reshape(nmagbin, ncolbin)

    ZS = [h, s, diff, sig]
    extent = [cmd['color'][0], cmd['color'][-1], cmd['mag'][-1], cmd['mag'][0]]
    if labels is None:
        grid = match_plot(ZS, extent, **kwargs)
    else:
        grid = match_plot(ZS, extent, labels=labels, **kwargs)

    if saveplot:
        figname = rsp.fileIO.replace_ext(filename, '.png')
        if out_dir is not None:
            figname = os.path.join(out_dir, os.path.split(figname)[1])
        plt.savefig(figname)
        logger.info(' % s wrote %s' % (pgcmd.__name__, figname))
    return grid

def plot_sfh(filename):  
        """
        Plots the star formation history (default at input (logarithmic) time resolution).
        Optionally plots the cumulative energy added from star formation, if
        include_energy = True. If this option is set, the default time resolution is
        the 5 Myr sampled linear resolution.
        """
        
        if include_energy and sampled is None:
            sampled = True
        
        if time_array is None:
            if sampled:
                time = np.dstack( (self.t0_sampled, self.t1_sampled ) ).flatten()
                sfh_array = self.get_sfh( t0 = self.t0_sampled,
                                          t1 = self.t1_sampled )
                sfh_array_lo = self.get_sfh(t0 = self.t0_sampled, t1 = self.t1_sampled,
                                            error = "low")
                sfh_array_hi = self.get_sfh(t0 = self.t0_sampled, t1 = self.t1_sampled,
                                            error = "high")
            else:
                time = np.dstack( ( self.t0_in, self.t1_in ) ).flatten()
                sfh_array = np.copy( self.sfr_in )
                sfh_array_lo = np.copy( self.sfr_in_err_lo)
                sfh_array_hi = np.copy( self.sfr_in_err_hi)
        else:
            time = np.dstack( ( time_array[:-1], time_array[1:] ) ).flatten()
            sfh_array = self.get_sfh( t0 = time_array[:-1], t1 = time_array[1:] )
            sfh_array_lo = self.get_sfh( t0 = time_array[:-1], t1 = time_array[1:],
                                         error = "low" )
            sfh_array_hi = self.get_sfh( t0 = time_array[:-1], t1 = time_array[1:],
                                         error = "high")
        
        sfh = np.dstack( (sfh_array, sfh_array) ).flatten()
        sfh_lo = np.dstack( (sfh_array_lo, sfh_array_lo) ).flatten()
        sfh_hi = np.dstack( (sfh_array_hi, sfh_array_hi) ).flatten()
        
            
        time = np.append([time[0]], time )
        sfh = np.append([0], sfh)
        sfh_lo = np.append([0], sfh_lo)
        sfh_hi = np.append([0], sfh_hi)
        
        if include_energy:
            fig, axes = plt.subplots(nrows = 2, ncols = 1, sharex = True,
                                     figsize = (6,8))
            ax_sfr = axes[1]
            ax_energy = axes[0]
        else:
            if ax is not None:
                print 'using given axis'
                ax_sfr = ax
            else:
                fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (6, 4))
                ax_sfr = axes
        
        
        plt.subplots_adjust(hspace = 0, left = 0.2, right = 0.9,
                            bottom = 0.15, top = 0.9)
        

        ax_sfr.set_xlim(0, tmax / 1e6)
        
        line = ax_sfr.plot(time / 1e6, sfh, '-')

        print line[0].get_color()
        # plot the average times and errors.
        yerr = np.vstack( (sfh_array_lo, sfh_array_hi) )
        ax_sfr.errorbar( (time[1::2] + time[2::2]) / 2. / 1e6, sfh[1::2],
            fmt = None,
            capsize = 0,
            ecolor = line[0].get_color(),
            xerr = 0,
            yerr = yerr)
        #ax_sfr.plot( (time[1::2] + time[2::2]) / 2. / 1e6, sfh[1::2], 'o',
        #    color = line[0].get_color())
        
        
        ax_sfr.set_xlabel("Time [Myr]")
        ax_sfr.set_ylabel("SFH [M$_\mathrm{\odot}$ yr${-1}$]")
        
        ax_sfr.yaxis.set_major_locator(mticker.MaxNLocator(5, prune = 'upper'))
        
            
        # if energies are requested..
        if include_energy:
            plt.setp(ax_energy.get_xticklabels(), visible=False)

            ax_energy.plot(self.t1_sampled / 1e6, self.e_sf_cumul / 1e52, 'ko-')
            ax_energy.fill_between(self.t1_sampled / 1e6,
                                   (self.e_sf_cumul + self.e_sf_cumul_err_hi) / 1e52,
                                   y2 = (self.e_sf_cumul - self.e_sf_cumul_err_lo) / 1e52,
                                   color = 'k', alpha = 0.3)
            ax_energy.set_ylabel("E$_\mathrm{SF}$ [10$^{52}$ erg]")
            ax_energy.yaxis.set_major_locator(mticker.MaxNLocator(5))
            ax_energy.text(0.1, 0.85, self.string,
                         transform = ax_energy.transAxes,
                         size = 'large')
        else:
            ax_sfr.text(0.1, 0.85, self.string,
                         transform = ax_sfr.transAxes,
                         size = 'large')
