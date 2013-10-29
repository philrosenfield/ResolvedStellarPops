import fileIO
import Galaxies
import graphics
import match_graphics
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import logging
import random
logger = logging.getLogger()


def read_binned_sfh(filename):
    '''
    reads the *.zc.sfh file from match, the one created using HMC from
    Dolphin 2013. Not sure what one of the cols is, so I have it a lixo.
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
    data = np.genfromtxt(filename, dtype=dtype)
    return data.view(np.recarray)


class StarFormationHistories(object):
    '''
    something
    '''
    def __init__(self, match_sfh_file):
        '''
        something
        '''
        self.data = read_binned_sfh(match_sfh_file)

    def random_draw_within_uncertainty(self, attr, npoints=2e5):
        '''
        attr is the string name of the array that also has attr_errm and
        attr_errp (p and m are important due to the sign).

        If errm and errp are equal, just returns a randomly chosen point
        (of npoints) of a gaussian with mean attr and sigma = attr_errm

        If not, will stick to gaussians together at attr using sigma=attr_errm
        and sigm=attr_errp and returning a random value from there.

        If one of the err values is zero, will just use the other half of
        the gaussian.

        If they are both zero, well, just returns attr.
        '''
        assert attr in ['sfr', 'mh'], 'Only set up for sfr and mh'
        val_arr = self.data.__getattribute__(attr)
        errm_arr = self.data.__getattribute__('%s_errm' % attr)
        errp_arr = self.data.__getattribute__('%s_errp' % attr)
        rand_arr = np.array([])

        if attr == 'sfr':
            lowlim = 0
        else:
            lowlim = -999
        for val, errm, errp in zip(val_arr, errm_arr, errp_arr):
            if errp == errm and errp > 0:
                # even uncertainties, easy.
                new_arr = np.random.normal(val, errp,  npoints)
            elif errp != 0 and errm != 0:
                # stitch two gaussians together
                pos_gauss = np.random.normal(val, errp,  npoints)
                neg_gauss = np.random.normal(val, errm, npoints)
                new_arr = np.concatenate([pos_gauss[pos_gauss >= val],
                                          neg_gauss[neg_gauss <= val]])

            elif errp == 0 and errm != 0:
                # no positive uncertainties
                neg_gauss = np.random.normal(val, errm, npoints)
                new_arr = neg_gauss[neg_gauss <= val]
            elif errp !=0 and errm == 0:
                # no negative uncertainties
                pos_gauss = np.random.normal(val, errp,  npoints)
                new_arr = pos_gauss[pos_gauss >= val]
            else:
                # um... no errors, why was this called
                print 'Warning: no uncertainties'
                new_arr = np.ones(4) * val
            new_arr = new_arr[new_arr > lowlim]
            rand_arr = np.append(rand_arr, random.choice(new_arr))
        return rand_arr

    def make_trilegal_sfh(self, match_sfh_file, random_sfr=False, random_z=False,
                          zdisp=True, outfile='default'):
        '''
        turn binned sfh in to trilegal sfh
        '''
        # In MATCH [M/H] = log(Z/Zsun) with Zsun = 0.02 (see MATCH's makemod.cpp)
        # It doesn't matter if this is "correct". Stellar models have absolute Z.
        # Zsun is just a scaling that needs to be undone from MATCH to here.
        zsun = 0.02

        if outfile == 'default':
          outfile = match_sfh_file.replace('.zc.sfh', '.tri.dat')

        msfh = read_binned_sfh(match_sfh_file)

        age1a = 10 ** (msfh.lagei)
        age1p = 1.0 * 10 ** (msfh.lagei + 0.0001)
        age2a = 1.0 * 10 ** msfh.lagef
        age2p = 1.0 * 10 ** (msfh.lagef + 0.0001)

        raw_sfr = msfh.sfr
        if random_sfr is False:
          sfr = raw_sfr

        raw_z = zsun * 10 ** msfh.mh
        if random_z is False:
          metalicity = raw_z

        if zdisp is True:
          zdisp = msfh.mh_disp
        fmt = '%.4e %.3e %.4f %.4f \n'
        with open(outfile, 'w') as out:
          for i in range(len(sfr)):
              if sfr[i] == 0:
                  continue
              out.write(fmt % (age1a[i], 0.0, metalicity[i], zdisp[i]))
              out.write(fmt % (age1p[i], sfr[i], metalicity[i], zdisp[i]))
              out.write(fmt % (age2a[i], sfr[i], metalicity[i], zdisp[i]))
              out.write(fmt % (age2p[i], 0.0, metalicity[i], zdisp[i]))
        return

    def compare_tri_match(self, trilegal_catalog, filter1, filter2,
                          outfig=None):
        '''
        Two plots, one M/H vs Age for match and trilegal, the other
        sfr for match vs age and number of stars of a given age for trilegal.
        '''
        sgal = Galaxies.simgalaxy(trilegal_catalog, filter1=filter1,
                                  filter2=filter2)
        sgal.lage = sgal.data.get_col('logAge')
        sgal.mh = sgal.data.get_col('[M/H]')
        issfr, = np.nonzero(self.sfr > 0)
        age_bins = np.digitize(sgal.lage, self.lagef[issfr])
        mean_mh= [np.mean(sgal.mh[age_bins==i]) for i in range(len(issfr))]

        bins = self.lagei
        sfr = np.array(np.histogram(sgal.lage, bins=bins)[0], dtype=float)

        fig, (ax1, ax2) = plt.subplots(figsize=(8,8), ncols=2, sharex=True)
        # should be density, weighted by number anyway...
        ax1.plot(sgal.lage, sgal.mh, '.', color='grey')
        ax1.plot(self.lagei[issfr], mean_mh, linestyle='steps', color='navy',
                 lw=3, label='TRILEGAL')

        ax1.plot(self.lagei[issfr], self.mh[issfr], linestyle='steps', lw=3,
                color='k', label='MATCH')
        ax1.fill_between(self.lagei[issfr], self.mh[issfr] + self.mh_disp[issfr],
                        self.mh[issfr] - self.mh_disp[issfr],
                        lw=2, color='red', alpha=0.2)
        ax1.set_ylabel('$[M/H]$', fontsize=20)
        ax1.set_xlabel('$\log {\\rm Age (yr)}$', fontsize=20)
        ax1.legend(loc=0, frameon=False)

        ax2.plot(bins[:-1], sfr/(np.sum(sfr)), linestyle='steps', color='navy',
                lw=3, label='TRILEGAL')
        ax2.plot(self.lagei, self.sfr/np.sum(self.sfr),
                 linestyle='steps', lw=2, color='k', label='MATCH')
        ax2.set_ylabel('$ {\propto \\rm SFR}$', fontsize=20)
        ax2.set_xlabel('$\log {\\rm Age (yr)}$', fontsize=20)
        ax2.legend(loc=0, frameon=False)
        ax2.set_xlim(8, 10.5)
        if outfig is not None:
            fig.savefig(outfig, dpi=300)



def make_phot(gal, fname='phot.dat'):
    '''
    makes phot.dat input file for match, a list of V and I mags.
    '''
    np.savetxt(fname, np.column_stack((gal.mag1, gal.mag2)), fmt='%.4f')


def make_match_param(gal, more_gal_kw=None):
    '''
    Make param.sfh input file for match
    see rsp.match_utils.match_param_fmt()

    takes calcsfh search limits to be the photometric limits of the stars in the cmd.
    gal is assumed to be angst galaxy, so make sure attr dmod, Av, comp50mag1,
    comp50mag2 are there.

    only set up for acs and wfpc, if other photsystems need to check syntax with match
    filters.

    All values passed to more_gal_kw overwrite defaults.
    '''

    more_gal_kw = more_gal_kw or {}

    # load parameters
    inp = fileIO.input_parameters(default_dict=match_param_default_dict())

    # add parameteres
    cmin = gal.color.min()
    cmax = gal.color.max()
    vmin = gal.mag1.min()
    imin = gal.mag2.min()

    if 'acs' in gal.photsys:
        V = gal.filter1.replace('F', 'WFC')
        I = gal.filter2.replace('F', 'WFC')
    elif 'wfpc' in gal.photsys:
        V = gal.filter1.lower()
        I = gal.filter2.lower()
    else:
        print gal.photsys, gal.name, gal.filter1, gal.filter2

    # default doesn't move dmod or av.
    gal_kw = {'dmod1': gal.dmod, 'dmod2': gal.dmod, 'av1': gal.Av, 'av2': gal.Av,
              'V': V, 'I': I, 'Vmax': gal.comp50mag1, 'Imax': gal.comp50mag2,
              'V-Imin': cmin, 'V-Imax': cmax, 'Vmin': vmin, 'Imin': imin}

    # combine sources of params
    phot_kw = dict(match_param_default_dict().items() + gal_kw.items() + more_gal_kw.items())

    inp.add_params(phot_kw)

    # write out
    inp.write_params('param.sfh', match_param_fmt())

    phot_kw = {}
    return


def match_param_default_dict():
    dd = {'ddmod': 0.05, 'dav': 0.05,
          'logzmin': -2.3, 'logzmax': 0.1, 'dlogz': 0.1,
          'logzmin0': -2.3, 'logzmax0': -1.0, 'logzmin1': -1.3, 'logzmax1': -0.1,
          'BF': 0.35, 'bad0': 1e-6, 'bad1': 1e-6,
          'ncmds': 1,
          'Vstep': 0.1, 'V-Istep': 0.05, 'fake_sm': 5,
          'nexclude_gates': 0, 'excludegates': '',
          'ninclude_gates': 0, 'include_gates': ''}
    return dd

def match_param_fmt():
    '''
    calcsfh parameter format, set up for dan's runs and parsec M<12.
    NOTE exclude and include gates are strings and must have a space at
    their beginning.
    '''
    return '''-1 %(dmod1).3f %(dmod2).3f %(ddmod).3f %(av1).3f %(av2).3f %(dav).3f
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


def make_exclude_gates(fits_files, trgb=True, make_plot=False):
    '''
    Create the string for the match input file 'exclude gates'
    this only works for the trgb, assumes V vs V-I cmds in match
    takes arbitary values for color max and min (hard coded)

    send a string of fits_files (abspath preferred)

    also has a hardcoded edge for hs117.
    '''
    if make_plot is True:
        fig, (axs) = graphics.GraphicsUtils.setup_multiplot(len(fits_files),
                                                                figsize=(30,30))
        axs = np.squeeze(np.concatenate(axs))
    exclude_gate = ' %(c0).2f %(m0).2f %(c1).2f %(m1).2f %(c2).2f %(m2).2f %(c3).2f %(m3).2f'
    exclude_gates = {}
    gal_kw = {'filetype': 'fitstable', 'hla': True, 'angst': True, 'band': 'opt'}

    for i, fits_file in enumerate(fits_files):
        gal = Galaxies.galaxy(fits_file, **gal_kw)
        if trgb is True:
            cmin = -0.5
            cmax = 3
            Vmax = gal.mag1.min()
            Vmin = gal.trgb
        else:
            print 'need to code in cmin, cmax, vmin, vmax...'
            return exclude_gates
        if 'hs117' in fits_file:
            cmax = 2
            Vmax = 23
        exclude_dict = {'c0': cmin, 'm0': cmin + Vmin,
                        'c1': cmax, 'm1': cmax + Vmin,
                        'c2': cmax, 'm2': Vmax,
                        'c3': cmin, 'm3': Vmax}
        exclude_gates[fits_file] = exclude_gate % exclude_dict
        if make_plot is True:
            test_arr = np.column_stack(([cmin, cmin + Vmin],
                                        [cmax, cmax + Vmin],
                                        [cmax, Vmax],
                                        [cmin, Vmax],
                                        [cmin, cmin + Vmin]))

            gal.plot_cmd(gal.color, gal.mag1, levels=3, threshold=100, ax=axs[i],
                            filter1=gal.filter1)
            gal.photsys = 'wfc3snap'
            gal.decorate_cmd(ax=axs[i], trgb=True, filter1=gal.filter1)
            axs[i].plot(test_arr[0, :], test_arr[1,:], lw=3, ls='--', color='green')
            axs[i].set_ylim(gal.mag1.max()+1, Vmax-1)
    if make_plot is True:
        plt.savefig('exclude_gates_%i.png' % len(fits_files), dpi=300)
        print 'wrote exclude_gates_%i.png' % len(fits_files)
    return exclude_gates



## All the code below is old, and could use rsp.fileIO or soemthing else.

def read_zctmp(filename):
    data = {'To': np.array([]),
            'Tf': np.array([]),
            'sfr': np.array([]),
            'logz': np.array([])}

    with open(filename,'r') as f:
        lines = f.readlines()
    
    nrow, ncol = map(int, lines[0].split())

    data['logz'] = np.array(lines[1].split(), dtype=float)
    for line in lines[2:]:
        to = float(line.split()[0])
        tf = float(line.split()[1])
        sfr = map(float,line.split()[2:])
        data['To'] = np.append(data['To'], to)
        data['Tf'] = np.append(data['Tf'], tf)
        data['sfr'] = np.append(data['sfr'], sfr)

    data['sfr'] = data['sfr'].reshape(nrow, ncol)
    return data


def plot_zctmp(filename):
    data = read_zctmp(filename)
    from ResolvedStellarPops.convertz import convertz
    z = np.round(convertz(feh=data['logz'])[1], 4)
    z_plt_arr = np.array([0.0002, 0.0008, 0.001, 0.004, 0.008, 0.010, 0.015, 0.02, 0.03])

    inds = [np.nonzero(z < zp)[0] for zp in z_plt_arr]
    binned_sfr = np.array([[np.sum(data['sfr'][i][inds[j]]) for i in range(len(data['sfr']))] for j in range(len(inds))])
    fig, ax = plt.subplots()

    [ax.plot(data['To'], binned_sfr[i], ls='steps', lw=2, label='%.4f' % z_plt_arr[i])
        for i in range(len(z_plt_arr))[::-1]]
    plt.legend(loc=0)
    ax.set_xlabel('$Gyr\ ago$', fontsize=20)
    ax.set_ylabel('$SFR\ M_\odot/yr$', fontsize=20)
    plt.savefig(filename + '.png')


def read_match_sfh(filename, bgfile=False):
    footer = 2
    if bgfile is True:
        footer += 2
    col_head = ['to', 'tf', 'sfr', 'nstars', 'dlogz', 'dmod']
    sfh = np.genfromtxt(sfhfile, skip_header=2, skip_footer=footer,
                        names=col_head)
    sfh = sfh.view(np.recarray)
    return sfh


def process_match_sfh(sfhfile, outfile='processed_sfh.out', bgfile=False,
                      sarah_sim=False, footer=2):
    '''
    turn a match sfh output file into a sfr-z table for trilegal.

    check: after new isochrones, do we need to go from lage 10.15 to 10.13?
    todo: add possibility for z-dispersion.
    '''
    if bgfile is True:
        footer += 2

    fmt = '%.6g %.6g %.4g \n'
    to, tf, sfr, nstars, dlogz, dmod = np.genfromtxt(sfhfile, skip_header=2,
                                                     skip_footer=footer,
                                                     unpack=True)

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
                z1 = 0.019 * 10 ** (dlogz[i] - half_bin)
                z2 = 0.019 * 10 ** (dlogz[i] + half_bin)
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

    print 'wrote', outfile
    return outfile


def check_exclude_gates(matchpars=None, qsub=None, match=None, save=True):
    mc = 0
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
            calcsfh = command[0]
            matchpars = command[1]
            match = command[2]
            matchfake = command[3]
            sfh = command[4]
            flags = command[5:command.index('>')]
        for flag in flags:
            if flag == '-allstars':
                mc = 1
            if flag == '-zinc':
                zinc = 1
    else:
        if matchpars is None or match is None:
            print 'need either qsub file or matchpars and match file'
            return 0


    mag1, mag2 = np.loadtxt(match, unpack=True)
    with open(matchpars, 'r') as f:
        m = f.readlines()

    if (zinc == 1) & (len(m[1].split()) != 7): print 'zinc might not be set right in matchpars'
    if (zinc == 0) & (len(m[1].split()) == 7): print 'zinc flag should be on according to matchpars.'

    for mm in m:
        if re.search('bg.dat', mm):
            bg = 0
            files = os.listdir('.')
            for file in files:
                if file == 'bg.dat':
                    bg = 1
            if bg == 0:
                print 'No bg.dat file in this directory, but matchpars calls for one'

    excludes = map(float, m[7].split())
    nregions = excludes[0]
    if nregions > 0:
        col = excludes[1:-1:2]
        mag = excludes[2:-1:2]
        col.append(col[0])
        mag.append(mag[0])
        if len(col) > 5:
            print 'not ready to do more than one region'

    colmin = float(m[4].split()[3])
    colmax = float(m[4].split()[4])
    mag1min = float(m[5].split()[1])
    mag1max = float(m[5].split()[0])
    mag2min = float(m[6].split()[1])
    mag2max = float(m[6].split()[0])

    mag2cut = np.nonzero((mag2 > mag2max) & (mag2 < mag2min))

    plt.plot(mag1-mag2,mag1,'.',color='grey')
    plt.plot(mag1[mag2cut]- mag2[mag2cut], mag1[mag2cut], '.', color='black')
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
        plt.savefig(os.getcwd()+'/'+qsub.split('/')[-1]+'.png')
        print 'wrote '+os.getcwd()+'/'+qsub.split('/')[-1]+'.png'
    plt.show()


def read_match_cmd(filename):
    '''
    reads MATCH .cmd file
    '''
    mc = open(filename, 'r').readlines()
    # I don't know what the 7th column is, so I call it lixo.
    names = ['mag', 'color', 'Nobs', 'Nsim', 'diff', 'sig', 'lixo']
    cmd = np.genfromtxt(filename, skip_header=4, names=names, invalid_raise=False)
    return cmd


def write_qsub(param, phot, fake, qsubfile, zinc=True, mc=False, cwd=None):
    flags = ''
    if zinc is True:
        flags = '-zinc'

    if cwd is None:
        cwd = os.getcwd()
    fits = phot.split('/')[-1].split('.match')[0]
    sfh = fits + '.sfh'
    log = fits + '.log'
    msg = fits + '.msg'
    if mc is True:
        if not re.search('mc', cwd):
            log = 'mc/' + log
            msg = 'mc/' + msg
            sfh = 'mc/' + sfh

    lines = '#PBS -l nodes=1:ppn=1 \n#PBS -j oe \n'
    lines += '#PBS -o %s \n' % os.path.join(cwd, log)
    lines += '#PBS -l walltime=12:00:00 \n'
    if mc is False:
        lines += '#PBS -M philrose@astro.washington.edu \n'

    lines += '#PBS -m abe \n#PBS -V \n'
    lines += 'cd %s \n' % cwd
    if mc is False:
        lines += '%s %s %s %s %s %s > %s \n' % (calcsfh_path, param, phot,
                                                fake, sfh, flags, msg)
    else:
        lines += '%s %s %s %s %s ' % (calcsfh_path, param, phot, fake, sfh)
        lines += '-allstars -logterrsig=0.03 -mbolerrsig=0.41'
        lines += ' %s > %s \n' % (flags, msg)

    with open(qsubfile, 'w') as qsub:
        qsub.write(lines)


def check_for_bg_file(param):
    '''
    the bg.dat file in the calcsfh parameter file can only be 79 chars long.
    this will check for that. If the length is longer than 78 chars, will
    rewrite the param file with a local copy, and copy the bg.dat file to
    current directory.

    Must delete the local dir's bg.dat in another module (if you want to).
    '''
    try:
        shutil
    except:
        import shutil
    # read current param file
    pm = open(param, 'r').readlines()

    # second to last line has bg.dat in it.
    data = pm[-2].strip().split()[-1]
    print data
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
    print 'wrote match_bg as %s' % filename
    return


def call_match(param, phot, fake, out, msg, flags=['zinc', 'PADUA_AGB'],
               loud=False):
    '''
    wrapper for calcsfh, takes as many flags as you want.
    '''
    calcsfh = os.path.join(os.environ['MATCH'], 'calcsfh')

    bg_file = check_for_bg_file(param)

    [fileIO.ensure_file(f) for f in (param, phot, fake)]

    cmd = ' '.join((calcsfh, param, phot, fake, out))
    cmd += ' -' + ' -'.join(flags)
    cmd += ' > %s' % (msg)
    cmd = cmd.replace(' - ', '')
    logger.debug(cmd)
    if loud is True:
        print cmd
    #p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE, close_fds=True)
    #stdout, stderr = (p.stdout, p.stderr)
    #err = p.wait()
    err = os.system(cmd)
    if err != 0:
        print 'PROBLEM WITH %s SKIPPING!' % param
        out = -1
        #logger.error(p.stderr.readlines())
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
            'bg_hess': .0, # neg if it's a .CMD, else it's same fmt as match_phot
            'smooth': 1,
            'ilogzmin': -2.3,
            'ilogzmax': -1.3,
            'flogzmin': -1.9,
            'flogzmax': -1.1,
            'match_bg': ''}


class calcsfh_params(object):
    '''
    someday, this will be a generic parameter house... someday. someday.
    '''
    def __init__(self, default_dict=None):
        if default_dict is None:
            default_dict = calcsfh_dict()
        assert len(default_dict) != 0, 'need values in default dictionary.'

        
        self.calcsfh_possible_params(default_dict)
        
    def calcsfh_possible_params(self, default_dict={}):
        for k, v in default_dict.items():
            self.__setattr__(k, v)
    
    def update_params(self, new_dict):
        '''
        only overwrite attributes that already exist from dictionary
        '''
        for k, v in new_dict.items():
            if hasattr(self, k):
                self.__setattr__(k, v)
    
    def add_params(self, new_dict):
        '''
        add or overwrite attributes from dictionary
        '''
        [self.__setattr__(k, v) for k, v in new_dict.items()]


def make_calcsfh_param_file(pmfile, galaxy=None, calcsfh_par_dict=None,
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
    calcsfh_pars = calcsfh_params(default_dict=calcsfh_par_dict)
    if galaxy is not None:
        gal = galaxy
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

    line = calcsfh_pars_fmt(calcsfh_pars)
    #print calcsfh_pars.__dict__
    pm = open(pmfile, 'w')
    pm.write(line % calcsfh_pars.__dict__)
    pm.close()
    logger.info('%s wrote %s' % (make_calcsfh_param_file.__name__, pmfile))
    return pmfile


def calcsfh_pars_fmt(calcsfh_pars):
    '''
    a pythonic way of writing calcsfh parameter file:
    from match2.4 readme
    IMF m-Mmin m-Mmax d(m-M) Avmin Avmax dAv
    logZmin logZmax dlogZ
    BF Bad0 Bad1
    Ncmds
    Vstep V-Istep fake_sm V-Imin V-Imax V, I  (per CMD)
    Vmin Vmax V                              (per filter)
    Imin Imax I                              (per filter)
    Nexclude_gates exclude_gates Ncombine_gates combine_gates (per CMD)
    Ntbins
      To Tf (for each time bin) <--- NOT IMPLEMENTED!
    optional background bins
    
    exclude/include gates only works for one of each at most.
    '''

    line = '%(imf).2f %(dmod).2f %(dmod2).2f %(ddmod).3f %(Av).2f'
    line += ' %(Av2).2f %(dAv).3f\n'
    if not calcsfh_pars.zinc:
        line += '%(logzmin).1f %(logzmax).1f %(dlogz).1f\n'
    else:
        line += '%(logzmin).1f %(logzmax).1f %(dlogz).1f %(ilogzmin).1f'
        line += ' %(ilogzmax).1f %(flogzmin).1f %(flogzmax).1f\n'

    line += '%(bf).2f %(bad0)f %(bad1)f\n'
    line += '%(Ncmds)i\n'
    line += '%(dmag).1f %(dcol).2f %(fake_sm)i %(colmin).1f %(colmax).1f'
    line += ' %(filter1)s,%(filter2)s\n'
    line += '%(bright1).2f %(faint1).2f %(filter1)s\n'
    line += '%(bright2).2f %(faint2).2f %(filter2)s\n'
    if calcsfh_pars.nexclude_gates != 0:
        line += '%i %s' % (calcsfh_pars.nexclude_gates,
                           poly2str(calcsfh_pars.nexclude_poly))
        if calcsfh_pars.ncombine_gates != 0:
            line += ' %i %s \n' % (calcsfh_pars.ncombine_gates,
                                poly2str(calcsfh_pars.ncombine_poly))
    else:
        line += '%(nexclude_gates)i %(ncombine_gates)i\n'
    line += '%(ntbins)i\n' 
    # if ntbins != 0: something...[to tf\n for to, tf in zip(TO, TF)]
    line += '%(dobg)i %(bg_hess).3f %(smooth)i%(match_bg)s\n'
    line += '-1 1 -1\n'
    return line


def poly2str(arr):
    '''
    print an array as a string without brackets. I coulda done a better
    google search to get this right.
    '''
    poly = ' '.join(str(arr))
    return poly.replace('[','').replace(']','')


def get_fit(filename):
    fh = [f.strip() for f in open(filename, 'r').readlines()
          if len(f.strip()) > 0]
    chi2 = float(fh[-2].split(':')[-1])
    fit = float(fh[-1].split(':')[-1])
    return chi2, fit


def match_light(gal, pm_file, match_phot, match_fake, match_out, msg,
                flags=['zinc', 'PADUA_AGB'], make_plot=False, model_name=None,
                figname=None, match_kwargs={}, loud=False):

    # prepare parameter file
    make_calcsfh_param_file(pm_file, galaxy=gal, kwargs=match_kwargs)

    # run match
    match_out = call_match(pm_file, match_phot, match_fake, match_out, msg,
                           flags=flags, loud=loud)

    if loud is True:
        print [l.strip() for l in open(msg).readlines()]
    
    if match_out == -1:
        return -1., -1
    # read the fit
    chi2, fit = get_fit(match_out)
    logger.info('%s Chi^2: %f Fit: %f' % (match_out, chi2, fit))
    
    if make_plot is True:
        # make plot
        if model_name is None:
            model_name = 'Model'

        alabel = r'$%s$' % model_name.replace('_','\ ')
        cmdgrid = match_out + '.cmd'
        grid = match_graphics.pgcmd(cmdgrid,
                                    labels=[gal.target, alabel, '$Difference$', '$\chi^2=%.3f$' % chi2],
                                    **{'filter1': gal.filter1,
                                       'filter2': gal.filter2})
        # save plot
        if figname is None:
            figname = fileIO.replace_ext(cmdgrid, '.png')
        plt.savefig(figname, dpi=300)
        plt.close()
        logger.info('%s wrote %s' % (match_light.__name__, figname))

    return chi2, fit

