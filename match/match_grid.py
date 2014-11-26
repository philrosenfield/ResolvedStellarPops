import numpy as np
import os
from .. import fileio
from astroML.stats import binned_statistic_2d
import matplotlib.pylab as plt
from matplotlib import rc, rcParams

rcParams['text.usetex']=True
rcParams['text.latex.unicode']=False
rcParams['axes.linewidth'] = 2
rcParams['ytick.labelsize'] = 'large'
rcParams['xtick.labelsize'] = 'large'
rcParams['axes.edgecolor'] = 'grey'
rc('text', usetex=True)

__all__ = ['MatchGrid']

class MatchGrid(object):
    def __init__(self, filenames, covs=None, ssp=False):
        self.covs = covs
        if ssp:
            self.load_grid(filenames, ssp=ssp)
        else:
            self.data = fileio.readfile(filenames)

    def load_grid(self, filenames):
        """
        file names is a set of ssp screen outputs for various COV values.
        Will make an array with the data attaching a column of COV.
        """
        self.bases = []
        self.names = []
        liness = []
        for filename in filenames:
            base, name = os.path.split(filename)
            self.bases.append(base)
            self.names.append(name)
            with open(filename, 'r') as infile:
                lines = infile.readlines()
            liness.append(lines[:-2])

        col_keys = ['Av', 'IMF', 'dmod', 'logAge', 'mh', 'fit', 'bg1', 'COV']
        dtype = [(c, float) for c in col_keys]
        nrows = len(np.concatenate(liness))
        self.data = np.ndarray(shape=(nrows,), dtype=dtype)
        row = 0
        for i, lines in enumerate(liness):
            for j in range(len(lines)):
                datum = np.array(lines[j].strip().split(), dtype=float)
                datum = np.append(datum, self.covs[i])
                self.data[row] = datum
                row += 1

    def pdf_plot(self, xcol, ycol, zcol, stat='median', bins='uniq',
                 log=False, cbar=True):
        if bins == 'uniq':
            bins=[np.unique(self.data[xcol]), np.unique(self.data[ycol])]
        N, xe, ye = binned_statistic_2d(self.data[xcol], self.data[ycol],
                                        self.data[zcol], stat, bins=bins)
        if log is True:
            n = np.log10(N.T)
        else:
            n = N.T
        aspect = (xe[-1] - xe[0]) / (ye[-1] - ye[0])
        fig, ax = plt.subplots()
        im = ax.imshow(n, extent=[xe[0], xe[-1], ye[0], ye[-1]],
                       cmap=plt.cm.Blues_r, aspect=aspect, interpolation='nearest')
        ax.set_xlabel(key2label(xcol), fontsize=16)
        ax.set_ylabel(key2label(ycol), fontsize=16)
        if cbar:
            cb = plt.colorbar(im)
            if callable(stat):
                stat = stat.__name__
            cb.set_label(r'$\rm{%s}$\ %s' % (stat, key2label(zcol)))
            ax = (ax, cb)
        #import pdb; pdb.set_trace()
        return ax

def key2label(string):
    possible = ['Z', 'Av', 'IMF', 'dmod', 'logAge', 'mh', 'fit', 'bg1', 'COV']
    labels = [r'$Z$', r'$A_V$', r'$IMF$', r'$\mu$', r'$\log\ \rm{Age\ (yr)}$',
              r'$\rm{[M/H]}$', r'$\rm{Fit\ Parameter}$', r'$bg1$', r'$\Lambda_c$']
    return labels[possible.index(string)]
