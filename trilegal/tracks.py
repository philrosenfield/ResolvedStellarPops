""" taken out of utils, these are Leo's tracks """

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

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

        with open(fname, 'r') as inp:
            lines = inp.readlines()

        nsplits = int(lines[0])
        header = nsplits + 4
        split_locs = nsplits + 2
        footer = np.sum(map(int, lines[split_locs].strip().split())) + header

        all_data = np.genfromtxt(fname, usecols=(0,1,2), skip_header=header,
                                 skip_footer=len(lines)-footer,
                                 names=['age', 'LOG_L', 'LOG_TE'])

        inds, = np.nonzero(all_data['age'] == 0)
        inds = np.append(inds, -1)
        indss = [np.arange(inds[i], inds[i+1]) for i in range(len(inds)-1)]

        self.masses = np.array(lines[header-1].split(), dtype=float)

        data = lines[header:footer]
        cri_lines = [l for l in data if 'cri' in l]
        if len(cri_lines) > 0:
            cri_data = np.array([l.split('cri')[-1].split()[-1] for l in cri_lines],
                                dtype=int)
            cinds, = np.nonzero(cri_data == 1)
        else:
            cri_lines = [l.strip() for l in data if len(l.strip().split()) > 4]
            cri_data = np.array([l.split()[-1] for l in cri_lines],
                                dtype=int)
            cinds, = np.nonzero(cri_data == 0)

        cinds = np.append(cinds, -1)
        cindss = [np.arange(cinds[i], cinds[i+1]) for i in range(len(cinds)-1)]

        track_dict = {}
        for i, ind in enumerate(indss):
            track_dict['M%.4f' % self.masses[i]] = all_data[ind]
            track_dict['criM%.4f' % self.masses[i]] = cri_data[cindss[i][:-1]]

        self.all_data = all_data.view(np.recarray)
        self.track_dict = track_dict
        self.masses.sort()

    def plot_tracks(self, col1, col2, ax=None, plt_kw={}, cri=False,
                    masses=None, labels=False, title=False):
        masses = masses or self.masses

        if type(masses) == str:
            masses = eval(masses)

        if ax is None:
            fig, ax = plt.subplots()

        for m in masses:
            key = 'M%.4f' % m
            if labels:
                plt_kw['label'] = '$%sM_\odot$' % key.replace('M', 'M=')
            try:
                data = self.track_dict[key]
            except KeyError:
                print '{}'.format(self.track_dict.keys())
            ax.plot(data[col1], data[col2], **plt_kw)
            if cri:
                inds = self.track_dict['cri%s' % key]
                ax.plot(data[col1][inds], data[col2][inds], 'o', **plt_kw)

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

def main(argv):
    parser = argparse.ArgumentParser(description="Plot ptcri files")

    parser.add_argument('-a', '--one_plot', action='store_true',
                        help='all files on one plot')

    parser.add_argument('-c', '--cri', action='store_true',
                        help='critical points')

    parser.add_argument('-x', '--col1', type=str, default='LOG_TE',
                        help='x axis column')

    parser.add_argument('-y', '--col2', type=str, default='LOG_L',
                        help='y axis column')

    parser.add_argument('-o', '--plotname', type=str, default='padovatracks.png',
                        help='output file name when making one big plot')

    parser.add_argument('name', nargs='*', type=str, help='ptcri file(s)')

    args = parser.parse_args(argv)

    pts = [PadovaTrack(f) for f in args.name]
    if args.one_plot:
        fig, ax = plt.subplots()
    else:
        ax = None

    for pt in pts:
        pt.plot_tracks(args.col1, args.col2, ax=ax)
        if not args.one_plot:
            outfile = os.path.join(pt.base, pt.name + '.png')
            plt.savefig(outfile)
            print('wrote %s' % outfile)
    if args.one_plot:
        plt.savefig(args.plotname)
        print('wrote %s' % args.plotname)
if __name__ == '__main__':
    main(sys.argv[1:])
