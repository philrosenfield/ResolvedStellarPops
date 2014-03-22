import fileIO
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
import sys

logger = logging.getLogger()

class Isochrone(object):
    def __init__(self):
        pass

    def fix_order(self):
        dif = np.sign(np.diff(self.data.M_ini))
        if -1 in dif:
            order = np.argsort(self.data.M_ini)
            self.data = self.data[order]

    def plot_isochrone(self, col1, col2, ax=None, fig=None, plt_kw={},
                       mag_convert_kw={}, photsys=None, clean=False, inds=None,
                       reverse_x=False, reverse_y=False, pms=False, xlim=None,
                       ylim=None, xdata=None, ydata=None):

        if ax is None:
            fig, ax = plt.subplots()

        if ydata is not None:
            y = ydata
        else:
            y = self.data[col2]
            if len(mag_convert_kw) != 0:
                import astronomy_utils
                y = astronomy_utils.Mag2mag(y, col2, photsys, **mag_convert_kw)
            ax.set_ylabel('$%s$' % col2.replace('_','\ '), fontsize=20)
        if xdata is not None:
            x = xdata
        else:
            if '-' in col1:
                col1a, col1b = col1.split('-')
                x1 = self.data[col1a]
                x2 = self.data[col1b]
                if len(mag_convert_kw) != 0:
                    x1 = astronomy_utils.Mag2mag(x1, col1a, photsys, **mag_convert_kw)
                    x2 = astronomy_utils.Mag2mag(x2, col1b, photsys, **mag_convert_kw)
                x = x1 - x2
            else:
                x = self.data[col1]

            ax.set_xlabel('$%s$' % col1.replace('_','\ '), fontsize=20)

        if pms is False and hasattr(self.data, 'stage'):
            nopms, = np.nonzero(self.data['stage'] != 0)
            if len(nopms) == 0:
                return ax
        else:
            nopms = np.arange(len(y))

        if inds is not None:
            inds = list(set(inds) & set(nopms))

        else:
            inds = nopms

        masses = self.data['M_ini']
        minds = np.argsort(masses[inds])
        x = x[inds]
        y = y[inds]

        if clean is True:
            x = x[minds]
            y = y[minds]
            isep = np.argmax(np.diff(x, 2))
            pl,  = ax.plot(x[:isep-1], y[:isep-1], **plt_kw)
            plt_kw['color'] = pl.get_color()
            plt_kw['label'] = ''
            pl,  = ax.plot(x[isep+1:], y[isep+1:], **plt_kw)
        else:
            ax.plot(x, y, **plt_kw)

        if reverse_x is True:
            ax.set_xlim(ax.get_xlim()[::-1])

        if reverse_y is True:
            ax.set_ylim(ax.get_ylim()[::-1])

        if xlim is not None:
            ax.set_xlim(xlim)

        if ylim is not None:
            ax.set_ylim(ylim)

        ax.tick_params(labelsize=16)

        return ax


class Isochrones(object):
    def __init__(self, filename):
        self.base, self.name = os.path.split(filename)
        meta_names = self.get_all_isochrones(filename)
        [self.__setattr__('%ss' % meta_names[i], [self.isos[j].__getattribute__(meta_names[i]) for j in range(len(self.isos))]) for i in range(len(meta_names))]

    def get_iso(self, attr, val):
        this_list = self.__getattribute__(attr)
        try:
            ind = this_list.index(val)
        except:
            print 'valid values of %s' % attr
            print this_list
            return
        return self.isos[ind]

    def plot_all_isochrones(self, col1, col2, plot_isochrone_kw={}):
        if not 'ax' in plot_isochrone_kw:
            fig, plot_isochrone_kw['ax'] = plt.subplots()

        for iso in self.isos:
            plot_isochrone_kw['ax'] = iso.plot_isochrone(col1, col2, **plot_isochrone_kw)

        return plot_isochrone_kw['ax']

    def get_all_isochrones(self, filename):

        with open(filename, 'r') as f:
            lines = np.array(f.readlines())

        header = [l for l in lines if l.startswith('#')]
        col_keys = header[-1].replace('#', '').replace('log', 'LOG_').replace('age/yr', 'AGE').replace('/', '_').replace('_Lo', '').replace('Te', 'TE').replace('(', '').replace(')', '').split()

        isoch_inds = [i for i, h in enumerate(lines) if h.replace('#', '').lstrip().startswith('Iso')]
        isoch_lines = lines[isoch_inds]
        start_inds = np.array(isoch_inds) + 2
        start_inds = np.concatenate((start_inds, [-1]))

        N_isochrones = len(isoch_inds)
        Nrows = len(lines) - len(header)
        Ncols = len(col_keys)

        #if 'PARSEC' in ' '.join(header):
        meta_data = np.array([il.replace('#', '').split()[3::3]
                              for il in isoch_lines], dtype=float).T
        meta_names = np.array(il.replace('#', '').replace('[M/H]', 'M_H').split()[1::3][:-1])

        # Padova models have that pesky string if tagged is true, so there's
        # a bit extra to the genfromtxt call.
        self.all_data = np.genfromtxt(filename, names=col_keys,
                                      usecols=range(len(col_keys)),
                                      invalid_raise=False)

        self.isos = []
        for i in range(N_isochrones):
            iso = Isochrone()
            [iso.__setattr__(meta_names[j], meta_data[j][i]) for j in range(len(meta_names))]
            iso.__setattr__('data', self.all_data[start_inds[i]:
                                                  start_inds[i+1]].view(np.recarray))
            iso.fix_order()
            self.isos.append(iso)
        return meta_names


def run_isoch(cmd_input_file, isoch_file, photsys, isoch_input_kw={},
              cmd_input_kw={}, overwrite=False, dry_run=False):

    if not os.path.isfile(cmd_input_file) or overwrite is True:
        write_cmd_input_for_isoch(cmd_input_file, photsys,
                                  cmd_input_kw=cmd_input_kw)
    else:
        print 'found %s, not overwriting' % cmd_input_file

    cmd = prepare_isoch_cmd(cmd_input_file, isoch_file,
                            isoch_input_kw=isoch_input_kw)

    here = os.getcwd()

    # copy the cmd_input_file to where we run cmd
    cmd_root = os.environ['CMDROOT']
    cmd_input_file_cp = os.path.join(cmd_root, os.path.split(cmd_input_file)[1])

    # if we're using a file in cmd_root, don't want to delete it...
    if cmd_input_file_cp != cmd_input_file:
        os.system('cp %s %s' % (cmd_input_file, cmd_input_file_cp))

    os.chdir(cmd_root)
    if dry_run is True:
        print cmd
    else:
        try:
           retcode = subprocess.call(cmd, shell=True)
           if retcode < 0:
               print >> sys.stderr, 'CMD was terminated by signal', -retcode
           else:
               print >> sys.stderr, 'CMD was terminated successfully'
        except OSError, err:
            print >> sys.stderr, 'CMD failed:', err

    if cmd_input_file_cp != cmd_input_file:
        os.remove(cmd_input_file_cp)

    os.chdir(here)


def prepare_isoch_cmd(cmd_input_file, isoch_file, isoch_input_kw={}):

    iso_inp = fileIO.input_parameters(default_dict=isoch_input_dict())
    iso_dict = dict({'cmd_input_file': cmd_input_file,
                     'isoch_file': isoch_file}.items() + isoch_input_kw.items())
    iso_inp.add_params(iso_dict)
    cmd = isoch_input_fmt() % iso_inp.__dict__
    if iso_inp.kind_iso != 2:
	cmd = cmd.replace('\n%s\n100' % iso_inp.kind_table, '\n100')
    return cmd


def write_cmd_input_for_isoch(cmd_input_file, photsys, cmd_input_kw={}):
    cmd_dict = cmd_input_for_isoch_dict(photsys)
    cmd_inp = fileIO.input_parameters(default_dict=cmd_dict)
    cmd_inp.add_params(cmd_input_kw)
    cmd_inp.write_params(cmd_input_file, cmd_input_for_isoch_fmt())


def cmd_input_for_isoch_fmt():
    fmt = \
        """%(kind_tracks)i %(file_isotrack)s %(file_lowzams)s # kind_tracks, file_isotrack, file_lowzams
%(kind_tpagb)i %(file_tpagb)s # kind_tpagb, file_tpagb
%(kind_postagb)i %(file_postagb)s # kind_postagb, file_postagb DA VERIFICARE file_postagb
%(kind_mag)i %(file_mag)s # kind_mag, file_mag
%(kind_imf)i %(file_imf)s # file_imf
%(kind_rgbmloss)i %(rgbmloss_law)s %(rgbmloss_efficiency).2f # RGB mass loss: kind_rgbmloss, law, and its efficiency
################################explanation######################
kind_tracks: 1= normal file
file_isotrack: tracks for low+int mass
file_lowzams: tracks for low-ZAMS
kind_tpagb: 0= none
	    1= Girardi et al., synthetic on the flight, no dredge up
	    2= Marigo & Girardi 2001, from file, includes mcore and C/O
	    3= Marigo & Girardi 2007, from file, with period, mode and mloss
	    4= Marigo et al. 2012, from file, with period, mode and mloss
file_tpagb: tracks for TP-AGB
kind_pulsecycle: 0= quiescent
		 1= quiescent interpolated in mloss
		 2= quiescent luminosity with mean mloss
		 3= detailed pulse cycle reconstructed

kind_postagb: 0= none
	      1= from file
file_postagb: PN+WD tracks

kind_mag: 1= normal sets of tables
	  2= in addition, uses dust tables
file_mag: list of tables

(note: dust lines present only if kind_mag=2, useful only if kind_tpagb>=3)
kind_dustM: 0= no dust
	    1= Groenewegen table
            2= Bressan table
file_dustM: list of files with tau-deltaBC relations
kind_dustC: 0= no dust
	    1= Groenewegen table
            2= Bressan table
file_dustC: list of files with tau-deltaBC relations

kind_imf:
file_imf:
"""
    return fmt

def cmd_input_for_isoch_dict(photsys):
    file_mag = 'tab_mag_odfnew/tab_mag_%s.dat' % photsys
    return {'kind_tracks': 2,
            'file_isotrack': 'isotrack/parsec/CAF09_S12D_NS_1TP.dat',
            'file_lowzams': 'isotrack/bassazams_fasulla.dat',
            'kind_tpagb': 4,
            'file_tpagb': 'isotrack/isotrack_agb/tracce_CAF09_S_JAN13.dat',
            'kind_postagb': 0,
            'file_postagb': 'isotrack/final/pne_wd_test.dat',
            'kind_mag': 1,
            'file_mag': file_mag,
            'kind_imf': 1,
            'file_imf': 'tab_imf/imf_salpeter.dat',
            'kind_rgbmloss': 1,
            'rgbmloss_law': 'Reimers',
            'rgbmloss_efficiency': 0.2}

def isoch_input_dict():
    '''
    kind_iso:
    0 -> Calcoli preparattori ?
    1 -> Scrive isocrona singola ?
    2 -> Scrive sequenza isocrone in eta' ?
    3 -> Scrive sequenza isocrone in metalicita' ?
    4 -> Calcola colori integrati per isocrona singola ?
    5 -> Calcola colori integrati per sequenza di eta' ?
    6 -> Calcola colori integrati per sequenza di metalicita' ?
    7 -> Scrive funzione di luminosita' ?
    8 -> Scrive funzione di magnitudine ?
    9 -> Scrive funzione di magnitudine per popolazione composta ?
    10 -> Scrive mapa di magnitudine per popolazione composta ?
    11 -> Scrive relazione massa-iniziale X finale ?
    18 -> Altri: calcola delta(V-I) per isocrona singola ?
    19 -> Altri: calcola delta(B-V) per isocrona singola ?
    20 -> Altri: calcola delta(V) per isocrone ?
    21 -> Altri: calcola delta(HB-RGB) per isocrone ?
    22 -> Altri: da i colori per un file di punti logL x logTeff ?
    23 -> Altri: calcola spettro integrato per isocrona ?
    24 -> Altri: calcola indici integrati per isocrona ?
    25 -> Altri: magnitu integrate tra limiti per sequenza di eta ?
    26 -> Altri: magnitu media dell'hb ?
    27 -> Altri: magnitu mediana dell'hb ?
    28 -> Simula ammasso con certa massa o magn limite ?
    29 -> Altri: estima mass partindo da observaveis Mv, B-V, [Fe/H] ?
    30 -> Altri: estima mass partindo da observaveis Mv, Teff, [Fe/H] ?
    31 -> Altri: estima distancia partindo da observaveis mv, Teff, [Fe/H] ?
    34 -> Dumps all tracks to file alltracks.dat ?
    kind_table:
    1 -> Isocrona semplice ?
	2 -> Con i colori ?
	3 -> Con le magnitudine ?
	4 -> Com int_IMF ?
	5 -> Con i punti critici ?
	6 -> Come Bertelli et al. ?
    Ex:
    if kind_iso is 2, other qty will be one value of metallicity
    qty_min, etc will be lage.
    if kind_iso is 3, other qty will be one value of log age
    qty_min, etc will be Z.
    '''
    return {'EOF': '<< $CMDROOT/EOF',
            'cmd_input_file': None,
            'kind_iso': 2,
            'isoch_file': '/dev/tty/',
            'other_qty': 0.001,
            'qty_min': 7.,
            'qty_max': 9.,
            'dq': 0.15,
            'kind_table': 5}

def isoch_input_fmt():

    return """code/main %(cmd_input_file)s %(EOF)s
%(kind_iso)i
%(isoch_file)s
%(other_qty)s
%(qty_min)f %(qty_max)f %(dq)f
%(kind_table)i
100
"""
