import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.offsetbox import AnchoredText
from matplotlib.patheffects import withStroke

__all__ = ['add_inner_title', 'forceAspect', 'match_plot', 'pgcmd',
           'read_match_cmd' ]


def add_inner_title(ax, title, loc, size=None, **kwargs):
    '''
    adds a title to an ax inside to a location loc, which follows plt.legends loc ints.
    '''
    if size is None:
        size = dict(size=plt.rcParams['legend.fontsize'])
    anct = AnchoredText(title, loc=loc, prop=size, pad=0., borderpad=0.5,
                        frameon=False, **kwargs)
    ax.add_artist(anct)
    anct.txt._text.set_path_effects([withStroke(foreground="w", linewidth=3)])
    return anct


def match_plot(ZS, extent, labels=["Data", "Model", "Diff", "Sig"],
               **kwargs):
    '''
    ex ZS = [h[2],sh[2],diff_cmd,resid]
    '''
    fig = plt.figure(figsize=(9, 9))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(2, 2),
                     direction="row",
                     axes_pad=.7,
                     add_all=True,
                     label_mode="all",
                     share_all=True,
                     cbar_location="top",
                     cbar_mode="each",
                     cbar_size="7%",
                     cbar_pad="2%",
                     aspect=0)

    # scale color bar data and model the same

    for i, (ax, z) in enumerate(zip(grid, ZS)):
        if i > 1:
            # second row: make 0 on the color bar white
            zz = z[np.isfinite(z)]
            vmin = -1. * np.max(np.abs(zz))
            vmax = np.max(np.abs(zz))
            colors = cm.RdBu
        else:
            # first row: make white 0, but will be on the left of color bar
            # scale color bar same for data and model.
            vmin = 0
            vmax = np.nanmax(ZS[0:2])
            if i == 0:
                colors = cm.Blues
            if i == 1:
                colors = cm.Reds
        im = ax.imshow(z, origin='upper', extent=extent, interpolation="nearest",
                       cmap=colors, vmin=vmin, vmax=vmax)
        ax.cax.colorbar(im)
        forceAspect(ax, aspect=1)

    # crop limits to possible data boundary
    ylim = (extent[2], extent[3])
    xlim = (extent[0], extent[1])
    for ax in grid:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.xaxis.label.set_visible(True)
        ax.yaxis.label.set_visible(True)

    for ax, im_title in zip(grid, labels):
        t = add_inner_title(ax, im_title, loc=1)
        t.patch.set_alpha(0.5)

    return grid


def forceAspect(ax, aspect=1):
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1] - extent[0]) /
                      (extent[3] - extent[2])) / aspect)


def pgcmd(filename, labels=None, figname=None, out_dir=None,
          axis_labels='default', filter1=None, filter2=None):
    '''
    produces the image that pgcmd.pro makes
    '''
    cmd = read_match_cmd(filename)
    if axis_labels.lower() == 'default':
        if filter1 is None or filter2 is None:
            filter1 = ''
            filter2 = ''
        kwargs = {'xlabel': r'$%s-%s$' % (filter1, filter2),
                  'ylabel': r'$%s$' % filter2}

    nmagbin = len(np.unique(cmd['mag']))
    ncolbin = len(np.unique(cmd['color']))
    data = cmd['Nobs'].reshape(nmagbin, ncolbin)
    model = cmd['Nsim'].reshape(nmagbin, ncolbin)
    diff = cmd['diff'].reshape(nmagbin, ncolbin)
    sig = cmd['sig'].reshape(nmagbin, ncolbin)

    hesses = [data, model, diff, sig]
    extent = [cmd['color'][0], cmd['color'][-1], cmd['mag'][-1], cmd['mag'][0]]
    if labels is None:
        grid = match_plot(hesses, extent, **kwargs)
    else:
        grid = match_plot(hesses, extent, labels=labels, **kwargs)

    [ax.set_xlabel('$%s-%s$' % (filter1, filter2), fontsize=20) for ax in grid.axes_all]
    [ax.set_ylabel('$%s$' % filter2, fontsize=20) for ax in grid.axes_all]
    grid.axes_all[0].xaxis.label.set_visible(True)

    if figname is not None:
        if out_dir is not None:
            figname = os.path.join(out_dir, os.path.split(figname)[1])
        plt.savefig(figname, dpi=300)
        print ' % s wrote %s' % (pgcmd.__name__, figname)
    return grid


def read_match_cmd(filename):
    '''
    reads MATCH .cmd file
    '''
    #mc = open(filename, 'r').readlines()
    # I don't know what the 7th column is, so I call it lixo.
    names = ['mag', 'color', 'Nobs', 'Nsim', 'diff', 'sig', 'lixo']
    cmd = np.genfromtxt(filename, skip_header=4, names=names, invalid_raise=False)
    return cmd

if __name__ == "__main__":
    '''
    calls pgcmd
    the labes for the plots are taken from the filename... this is hard coded
    labels[1] = '${\\rm %s}$' % filename.split('.')[0].replace('_', '\ ')
    I want to have the * usage in the command line, so the last two values
    are the filter names.
    usage eg:
    python graphics.py *cmd 'F555W' 'F814W\ (HRC)'
    '''
    import sys
    args = sys.argv
    filename = args[1:-2]
    filter1 = args[-2]
    filter2 = args[-1]
    labels = ['${\\rm %s}$' % i for i in ('data', 'model', 'diff', 'sig')]

    if type(filename) == str:
        filenames = [filename]
    else:
        filenames = filename

    olabels = labels
    for filename in filenames:
        figname = filename + '.png'
        labels[1] = '${\\rm %s}$' % filename.split('.')[0].replace('_', '\ ')
        pgcmd(filename, filter1=filter1, filter2=filter2, labels=labels,
              figname=figname)
