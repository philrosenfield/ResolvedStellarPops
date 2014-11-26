from __future__ import print_function
import os
import re
import logging
import numpy as np
import matplotlib.pyplot as plt
logger = logging.getLogger()

from .. import fileio
#from .. import graphics


__all__ = ['calcsfh_dict', 'call_match', 'check_exclude_gates',
           'check_for_bg_file', 'make_calcsfh_param_file', 'strip_header',
           'match_param_default_dict', 'match_param_fmt', 'match_stats',
           'process_match_sfh', 'read_binned_sfh', 'read_match_cmd',
            'write_match_bg', 'cheat_fake']

def float2sci(num):
    return r'$%s}$' % ('%.0E' % num).replace('E', '0').replace('-0', '^{-').replace('+0', '^{').replace('O', '0')


def strip_header(ssp_file, skip_header=10):
    outfile = ssp_file + '.dat'
    with open(ssp_file, 'r') as infile:
        lines = [l.strip() for l in infile.readlines()]
    np.savetxt(outfile, np.array(lines[skip_header:], dtype=str), fmt='%s')


def cheat_fake(infakefile, outfakefile):
    """
    Increase the resolution of a match fake file by repeating the
    fake star entry with slight shifts within the same hess diagram
    cell of mag2.

    Parameters
    ----------
    infakefile, outfakefile : str, str
        input and output fake file names
    """
    # infake format is mag1in, mag2in, mag1idff, mag2diff
    infake = np.loadtxt(infakefile)

    offsets = [0.06, 0.03, -0.03, -0.06]

    outfake = np.copy(infake)
    for offset in offsets:
        tmp = np.copy(infake)
        tmp.T[1] += offset
        outfake = np.concatenate([outfake, tmp])
    np.savetxt(outfakefile, outfake, '%.3f')
    return


def match_stats(sfh_file, match_cmd_file, nfp_nonsfr=5, nmc_runs=10000,
                outfile='cmd_stats.dat'):
    '''
    NFP = # of non-zero time bins
          + dmod + av + 1 for metallicity (zinc) + 2 for background.

    run match/bin/stats on a match_cmd_file. Calculates the non-zero sfr bins
    in sfh_file.
    '''
    stats_exe = '$HOME/research/match2.5/bin/stats'
    sfr_data = read_binned_sfh(sfh_file)
    inds, = np.nonzero(sfr_data.sfr)

    perr_frac = sfr_data.sfr_errp[inds] / sfr_data.sfr[inds]
    merr_frac = sfr_data.sfr_errm[inds] / sfr_data.sfr[inds]

    nonzero_bins = len(inds)
    nfp = nonzero_bins + nfp_nonsfr
    cmd = '%s %s %i %i >> %s \n' % (stats_exe, match_cmd_file, nmc_runs, nfp,
                                    outfile)

    print('writing to', outfile)
    with open(outfile, 'a') as out:
        out.write('min + sfr err: %.3f\n' % np.min(perr_frac))
        out.write('min - sfr err: %.3f\n' % np.min(merr_frac))
        out.write('median + sfr err: %.3f\n' % np.median(perr_frac))
        out.write('median - sfr err: %.3f\n' % np.median(merr_frac))
        out.write('max + sfr err: %.3f\n' % np.max(perr_frac))
        out.write('max - sfr err: %.3f\n' % np.max(merr_frac))
        out.write('# %s' % cmd)
    os.system(cmd)
    return


def read_binned_sfh(filename):
    '''
    reads the file created using zcombine or HybridMC from match
    into a np.recarray.

    NOTE
    calls genfromtext up to 3 times. There may be a better way to figure out
    how many background lines/what if there is a header... (it's a small file)
    '''
    dtype = [('lagei', '<f8'),
             ('lagef', '<f8'),
             ('dmod', '<f8'),
             ('sfr', '<f8'),
             ('sfr_errp', '<f8'),
             ('sfr_errm', '<f8'),
             ('mh', '<f8'),
             ('mh_errp', '<f8'),
             ('mh_errm', '<f8'),
             ('mh_disp', '<f8'),
             ('mh_disp_errp', '<f8'),
             ('mh_disp_errm', '<f8'),
             ('csfr', '<f8'),
             ('csfr_errp', '<f8'),
             ('csfr_errm', '<f8')]
    try:
        data = np.genfromtxt(filename, dtype=dtype)
    except ValueError:
        try:
            data = np.genfromtxt(filename, dtype=dtype, skip_header=6,
                                 skip_footer=1)
        except ValueError:
            data = np.genfromtxt(filename, dtype=dtype, skip_header=6,
                                 skip_footer=2)
    return data.view(np.recarray)


class MatchSFH(object):
    '''
    load the match sfh solution as a class with attributes set by the
    best fits from the sfh file.
    '''
    def __init__(self, filename):
        self.base, self.name = os.path.split(filename)
        self.data = read_binned_sfh(filename)
        self.load_match_header(filename)

    def load_match_header(self, filename):
        '''
        assumes header is from line 0 to 6 and sets footer to be the final
        line of the file

        header formatting is important:
        Line # format requirement
        first  Ends with "= %f (%s)"
        N      is the string "Best fit:\n"
        N+1    has ',' separated strings of "%s=%f+%f-%f"
        last   is formatted "%s %f %f %f"
        '''
        def set_value_err_attr(key, attr, pattr, mattr):
            '''
            set attributes [key], [key]_perr, [key]_merr
            to attr, pattr, mattr (must be floats)
            '''
            self.__setattr__(key, float(attr))
            self.__setattr__(key + '_perr', float(pattr))
            self.__setattr__(key + '_merr', float(mattr))

        with open(filename, 'r') as infile:
            lines = infile.readlines()

        if len(lines) == 0:
            print('empty file: %s' % filename)
            self.header = []
            self.footer = []
            self.bestfit = np.nan
            self.match_out = ''
            self.data = []
            return

        self.header = lines[0:6]
        self.footer = lines[-1]
        bestfit, fout = self.header[0].replace(' ', '').split('=')[1].split('(')
        self.bestfit = float(bestfit)
        self.match_out = fout.split(')')[0]

        try:
            iline = self.header.index('Best fit:\n') + 1
        except ValueError:
            print('Need Best fit line to assign attributes')
            raise ValueError

        line = self.header[iline].strip().replace(' ', '').split(',')
        for l in line:
            key, attrs = l.split('=')
            attr, pmattr = attrs.split('+')
            pattr, mattr = pmattr.split('-')
            set_value_err_attr(key, attr, pattr, mattr)
        # the final line has totalSF
        key, attr, pattr, mattr = self.header[-1].strip().split()
        set_value_err_attr(key, attr, pattr, mattr)

        self.flag = None
        if np.sum(np.diff(self.data.mh)) == 0:
            self.flag = 'setz'
        if len(np.nonzero(np.diff(self.data.mh) >= 0)[0]) == len(self.data.mh):
            self.flag = 'zinc'
        return

    def mh2z(self, num):
        return 0.02 * 10 ** num

    def plot_bins(self, val='sfr', err=False, convertz=False, offset=1.):
        '''make SFH bins for plotting'''
        if type(val) == str:
            if err:
                #import pdb; pdb.set_trace()
                valm = self.data['%s_errm' % val] * offset
                valp = self.data['%s_errp' % val] * offset
            val = self.data[val] * offset
            if convertz:
                val = self.mh2z(val)
                if err:
                    valm = self.mh2z(valm)
                    valp = self.mh2z(valp)
        lagei = self.data.lagei
        lagef = self.data.lagef

        # double up value
        if err:
            lages = (lagei + lagef) / 2
            return lages, val, valm, valp
        else:
            # lagei_i, lagef_i, lagei_i+1, lagef_i+1 ...
            lages = np.ravel([(lagei[i], lagef[i]) for i in range(len(lagei))])
            vals = np.ravel([(val[i], val[i]) for i in range(len(val))])
            return lages, vals

    def age_plot(self, val='sfr', ax=None, plt_kw={}, errors=True,
                 convertz=False, xlabel=None, ylabel=None,
                 sfr_offset=1e3):
        plt_kw = dict({'lw': 3, 'color': 'black'}.items() + plt_kw.items())
        eplt_kw = plt_kw.copy()
        eplt_kw.update({'linestyle': 'None'})

        lages, sfrs = self.plot_bins(offset=sfr_offset)
        rlages, rsfrs, sfr_merrs, sfr_perrs = self.plot_bins(err=True,
                                                             offset=sfr_offset)

        if val != 'sfr':
            lages, vals = self.plot_bins(val=val, convertz=convertz)
            # mask values with no SF
            isfr, = np.nonzero(sfrs==0)
            vals[isfr] = 0.
            if self.flag != 'setz':
                rlages, rvals, val_merrs, val_perrs = self.plot_bins(val=val,
                                                                     err=True)
                # mask values with no SF
                irsfr, = np.nonzero(rsfrs==0)
                val_merrs[irsfr] = 0.
                val_perrs[irsfr] = 0.
            else:
                errors = False
            if 'mh' in val:
                if ylabel is not None:
                    ylabel = r'$\rm{[M/H]}$'
                if convertz:
                    ylabel = r'$Z$'
        else:
            ylabel = r'$SFR\ %s (\rm{M_\odot/yr})$' % \
                     float2sci(1./sfr_offset).replace('$','')
            vals = sfrs
            rvals = rsfrs
            val_merrs = sfr_merrs
            val_perrs = sfr_perrs
        if ax is None:
            fig, ax = plt.subplots()
            xlabel = r'$\log Age\ \rm{(yr)}$'

        ax.plot(lages, vals, **plt_kw)
        if errors:
            ax.errorbar(rlages, rvals, yerr=[val_merrs, val_perrs], **eplt_kw)

        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=20)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=20)
        return ax


def match_param_default_dict():
    ''' default params for match param file'''
    dd = {'ddmod': 0.05, 'dav': 0.05,
          'logzmin': -2.3, 'logzmax': 0.1, 'dlogz': 0.1,
          'logzmin0': -2.3, 'logzmax0': -1.0, 'logzmin1': -1.3,
          'logzmax1': -0.1,
          'BF': 0.35, 'bad0': 1e-6, 'bad1': 1e-6,
          'ncmds': 1,
          'Vstep': 0.1, 'V-Istep': 0.05, 'fake_sm': 5,
          'nexclude_gates': 0, 'excludegates': '',
          'ninclude_gates': 0, 'include_gates': ''}
    return dd


def match_param_fmt(set_z=False, zinc=True):
    '''
    calcsfh parameter format, set up for dan's runs and parsec M<12.
    NOTE exclude and include gates are strings and must have a space at
    their beginning.
    '''

    fmt = '''-1 %(dmod1).3f %(dmod2).3f %(ddmod).3f %(av1).3f %(av2).3f %(dav).3f
%(logzmin).2f %(logzmax).2f %(dlogz).2f %(logzmin0).2f %(logzmax0).2f %(logzmin1).2f %(logzmax1).2f
%(BF).2f %(bad0).6f %(bad1).6f
%(ncmds)i
%(Vstep).2f %(V-Istep).2f %(fake_sm)i %(V-Imin).2f %(V-Imax).2f %(V)s,%(I)s
%(Vmin).2f %(Vmax).2f %(V)s
%(Imin).2f %(Imax).2f %(I)s
%(nexclude_gates)i%(exclude_gates)s %(ninclude_gates)i%(include_gates)s
43
7.30 7.40
7.40 7.50
7.50 7.60
7.60 7.70
7.70 7.80
7.80 7.90
7.90 8.00
8.00 8.10
8.10 8.20
8.20 8.30
8.30 8.40
8.40 8.50
8.50 8.60
8.60 8.70
8.70 8.75
8.75 8.80
8.80 8.85
8.85 8.90
8.90 8.95
8.95 9.00
9.00 9.05
9.05 9.10
9.10 9.15
9.15 9.20
9.20 9.25
9.25 9.30
9.30 9.35
9.35 9.40
9.40 9.45
9.45 9.50
9.50 9.55
9.55 9.60
9.60 9.65
9.65 9.70
9.70 9.75
9.75 9.80
9.80 9.85
9.85 9.90
9.90 9.95
9.95 10.00
10.00 10.05
10.05 10.10
10.10 10.15
-1 5 -1bg.dat
-1  1 -1
'''


def process_match_sfh(sfhfile, outfile='processed_sfh.out', sarah_sim=False):
    '''
    turn a match sfh output file into a sfr-z table for trilegal.

    check: after new isochrones, do we need to go from lage 10.15 to 10.13?
    todo: add possibility for z-dispersion.
    '''

    fmt = '%.6g %.6g %.4g \n'

    data = read_binned_sfh(sfhfile)
    to = data['lagei']
    tf = data['lagef']
    sfr = data['sfr']
    #from ResolvedStellarPops.convertz import convertz
    #dlogz = convertz(mh=data['mh'])[1]  # values of zero could be bad, but sfr == 0 skips below.
    dlogz = data['mh']
    half_bin = np.diff(dlogz[0: 2])[0] / 2.
    # correct age for trilegal isochrones.
    tf[tf == 10.15] = 10.13
    with open(outfile, 'w') as out:
        for i in range(len(to)):
            if sfr[i] == 0.:
                continue
            if sarah_sim is True:
                z1 = dlogz[i] - half_bin
                z2 = dlogz[i] + half_bin
                sfr[i] /= 2.
            else:
                sfr[i] *= 1e3  # sfr is normalized in trilegal
                z1 = 0.02 * 10 ** (dlogz[i] - half_bin)
                z2 = 0.02 * 10 ** (dlogz[i] + half_bin)
            age1a = 1.0 * 10 ** to[i]
            age1p = 1.0 * 10 ** (to[i] + 0.0001)
            age2a = 1.0 * 10 ** tf[i]
            age2p = 1.0 * 10 ** (tf[i] + 0.0001)

            out.write(fmt % (age1a, 0.0, z1))
            out.write(fmt % (age1p, sfr[i], z1))
            out.write(fmt % (age2a, sfr[i], z2))
            out.write(fmt % (age2p, 0.0, z2))
            out.write(fmt % (age1a, 0.0, z2))
            out.write(fmt % (age1p, sfr[i], z2))
            out.write(fmt % (age2a, sfr[i], z1))
            out.write(fmt % (age2p, 0.0, z1))

    print('wrote', outfile)
    return outfile


def check_exclude_gates(matchpars=None, qsub=None, match=None, save=True):
    #mc = 0
    zinc = 0
    if qsub is not None:
        q = open(qsub,'r').readlines()
        for line in q:
            if line.startswith('#'):
                continue
            if line.startswith('cd'):
                os.chdir(line.strip().split()[-1])
                continue
            command = line.split()
            #calcsfh = command[0]
            matchpars = command[1]
            match = command[2]
            #matchfake = command[3]
            #sfh = command[4]
            flags = command[5:command.index('>')]
        for flag in flags:
            #if flag == '-allstars':
            #    mc = 1
            if flag == '-zinc':
                zinc = 1
    else:
        if matchpars is None or match is None:
            print('need either qsub file or matchpars and match file')
            return 0

    mag1, mag2 = np.loadtxt(match, unpack=True)
    with open(matchpars, 'r') as f:
        m = f.readlines()

    if (zinc == 1) & (len(m[1].split()) != 7):
        print('zinc might not be set right in matchpars')

    if (zinc == 0) & (len(m[1].split()) == 7):
        print('zinc flag should be on according to matchpars.')

    for mm in m:
        if re.search('bg.dat', mm):
            bg = 0
            files = os.listdir('.')
            for file in files:
                if file == 'bg.dat':
                    bg = 1
            if bg == 0:
                print('No bg.dat file in this directory, but matchpars calls for one')

    excludes = map(float, m[7].split())
    nregions = excludes[0]
    if nregions > 0:
        col = excludes[1:-1:2]
        mag = excludes[2:-1:2]
        col.append(col[0])
        mag.append(mag[0])
        if len(col) > 5:
            print('not ready to do more than one region')

    colmin = float(m[4].split()[3])
    colmax = float(m[4].split()[4])
    mag1min = float(m[5].split()[1])
    mag1max = float(m[5].split()[0])
    mag2min = float(m[6].split()[1])
    mag2max = float(m[6].split()[0])

    mag2cut = np.nonzero((mag2 > mag2max) & (mag2 < mag2min))

    plt.plot(mag1 - mag2, mag1, '.', color='grey')
    plt.plot(mag1[mag2cut] - mag2[mag2cut], mag1[mag2cut], '.', color='black')
    plt.plot([colmin, colmin], [mag1min, mag1max],'--', lw=3, color='green')
    plt.plot([colmax, colmax], [mag1min, mag1max],'--', lw=3, color='green')
    plt.plot([colmin, colmax], [mag1min, mag1min],'--', lw=3, color='green')
    plt.plot([colmin, colmax], [mag1max, mag1max],'--', lw=3, color='green')
    plt.xlabel(m[4].split(' ')[-1].replace(',','-'))
    plt.ylabel(m[5].split()[-1])
    if qsub is not None:
        plt.title(qsub.split('/')[-1])
    if nregions > 0:
        plt.plot(col, mag, lw=3, color='red')
    off = 0.1
    plt.axis([colmin - off,
              colmax + off,
              mag1min + off,
              mag1max - off])
    if save is True:
        plt.savefig(os.getcwd() + '/' + qsub.split('/')[-1] + '.png')
        print('wrote ' + os.getcwd() + '/' + qsub.split('/')[-1] + '.png')
    plt.show()


def read_match_cmd(filename):
    '''
    reads MATCH .cmd file
    '''
    # mc = open(filename, 'r').readlines()
    # I don't know what the 7th column is, so I call it lixo.
    names = ['mag', 'color', 'Nobs', 'Nsim', 'diff', 'sig', 'lixo']
    cmd = np.genfromtxt(filename, skip_header=4, names=names, invalid_raise=False)
    return cmd


def check_for_bg_file(param):
    '''
    the bg.dat file in the calcsfh parameter file can only be 79 chars long.
    this will check for that. If the length is longer than 78 chars, will
    rewrite the param file with a local copy, and copy the bg.dat file to
    current directory.

    Must delete the local dir's bg.dat in another module (if you want to).
    '''
    import shutil
    # read current param file
    pm = open(param, 'r').readlines()

    # second to last line has bg.dat in it.
    data = pm[-2].strip().split()[-1]
    try:
        float(data)
        bg_file = 0
    except:
        # this needs to be abspath to work.... must get -1 or 1 out of the way
        bg_file = data[data.index('/'):]
        # this is only a problem for large filenames.
        if len(bg_file) <= 78:
            bg_file = 0

    # overwrite param file to have local copy of bg.dat
    if bg_file != 0:
        logger.warning(' % s filename is too long. Copying it locally.' %
                       bg_file)
        pm2 = open(param, 'w')
        for line in pm[:-2]:
            pm2.write(line)
        pm2.write('-1 1 -1%s\n' % os.path.split(bg_file)[1])
        pm2.write(pm[-1])
        pm2.close()
        # copy bg.dat here.
        shutil.copy(bg_file, os.getcwd())
        bg_file = os.path.split(bg_file)[1]
    return bg_file


def write_match_bg(color, mag2, filename):
    mag1 = color + mag2
    assert len(mag1) > 0 and len(mag2) > 0, 'match_bg is empty.'
    np.savetxt(filename, np.column_stack((mag1, mag2)), fmt='%.4f')
    print('wrote match_bg as %s' % filename)
    return


def call_match(param, phot, fake, out, msg, flags=['zinc', 'PADUA_AGB'],
               loud=False):
    '''
    wrapper for calcsfh, takes as many flags as you want.
    '''
    calcsfh = os.path.join(os.environ['MATCH'], 'calcsfh')

    bg_file = check_for_bg_file(param)

    [fileio.ensure_file(f) for f in (param, phot, fake)]

    cmd = ' '.join((calcsfh, param, phot, fake, out))
    cmd += ' -' + ' -'.join(flags)
    cmd += ' > %s' % (msg)
    cmd = cmd.replace(' - ', '')
    logger.debug(cmd)
    if loud is True:
        print(cmd)
    err = os.system(cmd)
    if err != 0:
        print('PROBLEM WITH %s SKIPPING!' % param)
        out = -1
    if bg_file != 0:
        os.remove(bg_file)
    return out


def calcsfh_dict():
    '''
    default dictionary for calcsfh.
    '''
    return {'dmod': 10.,
            'Av': 0.,
            'filter1': None,
            'filter2': None,
            'bright1': None,
            'faint1': None,
            'bright2': None,
            'faint2': None,
            'color': None,
            'mag': None,
            'dmod2': None,
            'colmin': None,
            'colmax': None,
            'Av2': None,
            'imf': 1.30,
            'ddmod': 0.050,
            'dAv': 0.050,
            'logzmin': -2.3,
            'logzmax': 0.1,
            'dlogz': 0.1,
            'zinc': True,
            'bf': 0.35,
            'bad0': 1e-6,
            'bad1': 1e-6,
            'Ncmds': 1,
            'dmag': 0.1,
            'dcol': 0.05,
            'fake_sm': 5,
            'nexclude_gates': 0,
            'exclude_poly': None,
            'ncombine_gates': 0,
            'combine_poly': None,
            'ntbins': 0,
            'dobg': -1,
            'bg_hess': .0,   # neg if it's a .CMD, else it's same fmt as match_phot
            'smooth': 1,
            'ilogzmin': -2.3,
            'ilogzmax': -1.3,
            'flogzmin': -1.9,
            'flogzmax': -1.1,
            'match_bg': ''}


def make_calcsfh_param_file(pmfile, starpop=None, calcsfh_par_dict=None,
                            kwargs={}):
    '''
    to search over range of av: set Av Av2 dAv as kwargs.
    to search over range of dmod: set dmod dmod2 ddmod in kwargs

    NOTE:
    can only handle 1 exclude gate and 1 combine gate.
    bg.dat has limited number of characters!!

    input:
    optional: galaxy object to grab attributes from

    kwargs:
    all parameters can be kwargs, technically, they aren't really kwargs since
    they aren't optional. Sorry python gods.

    '''
    calcsfh_pars = fileio.InputFile(default_dict=calcsfh_par_dict)
    if starpop is not None:
        gal = starpop
        # take attributes from galaxy object
        calcsfh_pars.update_params(gal.__dict__)
        calcsfh_pars.faint1 = gal.comp50mag1
        calcsfh_pars.faint2 = gal.comp50mag2
        calcsfh_pars.colmin = np.min(gal.color)
        calcsfh_pars.colmax = np.max(gal.color)
        calcsfh_pars.bright1 = np.min(gal.mag1)
        calcsfh_pars.bright2 = np.min(gal.mag2)

    # assign kwargs (to overwrite galaxy attributes):
    calcsfh_pars.add_params(kwargs)

    # if dmod and av range is not set, used fixed.
    if calcsfh_pars.dmod2 is None:
        calcsfh_pars.dmod2 = calcsfh_pars.dmod

    if calcsfh_pars.Av2 is None:
        calcsfh_pars.Av2 = calcsfh_pars.Av

    line = match_param_fmt(calcsfh_pars)
    #print calcsfh_pars.__dict__
    pm = open(pmfile, 'w')
    pm.write(line % calcsfh_pars.__dict__)
    pm.close()
    logger.info('%s wrote %s' % (make_calcsfh_param_file.__name__, pmfile))
    return pmfile
