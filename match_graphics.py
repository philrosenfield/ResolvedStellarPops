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
    fig = plt.figure(figsize=(9,9))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(2, 2),
                     direction="row",
                     axes_pad=0.23,
                     add_all=True,
                     label_mode="1",
                     share_all=True,
                     cbar_location="top",
                     cbar_mode="each",
                     cbar_size="7%",
                     cbar_pad="2%",
                     aspect=0)

    for i, (ax, z) in enumerate(zip(grid, ZS)):
        if i > 1:
            # second row: make 0 on the color bar white
            vmin =  -1. * np.max(np.abs(z))
            vmax = np.max(np.abs(z))
            colors = cm.RdBu
        else:
            # first row: make white 0, but will be on the left of color bar
            vmin = None
            vmax = None
            if i == 0:
                colors = cm.Blues
            if i == 1:
                colors = cm.Reds
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

def pgcmd(filename, labels=None, saveplot=False, out_dir=None,
          axis_labels='default', **kwargs):
    '''
    produces the image that pgcmd.pro makes
    '''
    cmd = match_utils.read_match_cmd(filename)
    if axis_labels.lower() == 'default':
        filter1 = kwargs.get('filter1')
        filter2 = kwargs.get('filter2')
        if filter1 is None or filter2 is None:
            import re
            t = re.search('_(.*)/(.*)_(.*)_(.{5})_(.{5}).(.*)', filename)
            (__, propidtarget, filter1, filter2, __, __) = t.groups()
        kwargs = {'xlabel': r'$%s-%s$' % (filter1, filter2),
                  'ylabel': r'$%s$' % filter2}
        if labels is None:
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
