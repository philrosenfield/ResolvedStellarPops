""" taken out of utils, these are Leo's tracks """

import os
import numpy as np

import matplotlib.pyplot as plt

class PadovaTrack(object):
    """leo's trilegal track"""
    def __init__(self, filename):
        self.base, self.name = os.path.split(filename)
        self.load_ptcri(filename)
        self.info_from_fname()

    def info_from_fname(self):
        """
        attributes from self.name
            ex: ptcri_CAF09_S12D_NS_S12D_NS_Z0.03_Y0.302.dat.INT2
            self.ext = 'INT2'
            self.Z = 0.03
            self.Y = 0.302
            self.pref = 'CAF09_S12D_NS_S12D_NS'
        """
        name, self.ext = self.name.replace('ptcri_', '').split('.dat.')
        self.pref, mets = name.split('_Z')
        self.Z, self.Y = np.array(mets.split('_Y'), dtype=float)

    def load_ptcri(self, fname):
        header = 6
        footer = 22
        all_data = np.genfromtxt(fname, usecols=(0,1,2), skip_header=header,
                                 skip_footer=footer,
                                 names=['age', 'LOG_L', 'LOG_TE'])
        lines = open(fname, 'r').readlines()

        inds, = np.nonzero(all_data['age'] == 0)
        
        mass_inx = int(lines[0]) + 3
        self.masses = np.array(lines[mass_inx].split(), dtype=float)

        inds = np.append(inds, -1)
        indss = [np.arange(inds[i], inds[i+1]) for i in range(len(inds)-1)]

        track_dict = {}
        for i, ind in enumerate(indss):
            track_dict['M%.4f' % self.masses[i]] = all_data[ind]

        self.all_data = all_data.view(np.recarray)
        self.track_dict = track_dict
        self.masses.sort()

    def plot_tracks(self, col1, col2, ax=None, plt_kw={},
                    masses=None, labels=False, title=False):
        if masses is None:
            masses = np.sort(self.track_dict.keys())
        elif type(masses[0]) == float:
            masses = ['M%.4f' % m for m in masses if m in self.masses]
        elif type(masses) == str:
            masses = eval(masses)
            masses = ['M%.4f' % m for m in masses if m in self.masses]

        if ax is None:
            fig, ax = plt.subplots()
        
        for key in masses:
            #key = 'M%.4f' % mass
            if labels:
                plt_kw['label'] = '$%sM_\odot$' % key.replace('M', 'M=')
            try:
                data = self.track_dict[key]
            except KeyError:
                print '{}'.format(self.track_dict.keys())
            ax.plot(data[col1], data[col2], **plt_kw)
        
        if title:
            mass_min = np.min(self.masses)
            mass_max = np.max(self.masses)
            ax.set_title('$Z={} M: {}-{}M_\odot$'.format(self.Z, mass_min,
                                                         mass_max))
        if labels:
            ax.legend(loc='best')
        ax.set_xlabel(r'$%s$' % col1.replace('_','\ '))
        ax.set_ylabel(r'$%s$' % col2.replace('_','\ '))
        return ax
