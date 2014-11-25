""" taken out of utils, these are Leo's tracks """

import os
import numpy as np


class PadovaTrack(object):
    """leo's trilegal track"""
    def __init__(self, filename):
        self.base, self.name = os.path.split(filename)
        self.data = self.load_ptcri(filename)
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
                                 names=['age', 'logl', 'logte'])
        lines = open(fname, 'r').readlines()

        inds, = np.nonzero(all_data['age'] == 0)
        self.masses = np.array([lines[i+header].split('M=')[1].split()[0]
                                for i in inds], dtype=float)
        inds = np.append(inds, -1)
        indss = [np.arange(inds[i], inds[i+1]) for i in range(len(inds)-1)]

        track_dict = {}
        for i, ind in enumerate(indss):
            track_dict['M%.4f' % self.masses[i]] = all_data[ind]

        self.all_data = all_data
        self.track_dict = track_dict
        self.masses.sort()