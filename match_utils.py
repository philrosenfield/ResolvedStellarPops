import ResolvedStellarPops as rsp
import numpy as np
from subprocess import PIPE, Popen
import matplotlib.pyplot as plt
import re
import os
import logging
logger = logging.getLogger()
if logger.name == 'root':
    rsp.fileIO.setup_logging()

def make_exclude_gates(gal, outfile=None):
    if outfile is None:
        outfile = gal.name + 'exclude_gate'
    pass
    # i just did this by hand looking at a cmd...
    
def read_match_sfh(filename, bgfile=False):
    footer = 2
    if bgfile is True:
        footer += 2
    col_head = ['to', 'tf', 'sfr', 'nstars', 'dlogz', 'dmod']
    sfh = np.genfromtxt(sfhfile, skip_header=2, skip_footer=footer,
                        names=col_head)
    sfh = sfh.view(np.recarray)
    return sfh
    
def process_match_sfh(sfhfile, outfile='processed_sfh.out', bgfile=False):
    '''
    turn a match sfh output file into a sfr-z table for trilegal.

    check: after new isochrones, do we need to go from lage 10.15 to 10.13?
    todo: add possibility for z-dispersion.
    '''
    footer = 2
    if bgfile is True:
        footer += 2

    out = open(outfile, 'w')
    fmt = ' % g %g %g\n'
    to, tf, sfr, nstars, dlogz, dmod = np.genfromtxt(sfhfile, skip_header=2,
                                                     skip_footer=footer,
                                                     unpack=True)

    half_bin = np.diff(dlogz[0: 2])[0] / 2.
    # correct age for trilegal isochrones.
    tf[tf == 10.15] = 10.13
    for i in range(len(to)):
        if sfr[i] == 0.:
            continue
        sfr[i] = sfr[i] * 1e3  # sfr is normalized in trilegal
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

    out.close()
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
    cmd = np.genfromtxt(filename, skip_header=4, names=names)
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

def call_match(param, phot, fake, out, msg, flags=['zinc', 'PADUA_AGB']):
    '''
    wrapper for calcsfh, takes as many flags as you want.
    '''
    flags.insert(0, '')
    calcsfh = os.path.join(os.environ['MATCH'], 'calcsfh')

    bg_file = check_for_bg_file(param)

    [rsp.fileIO.ensure_file(f) for f in (param, phot, fake)]

    cmd = ' '.join((calcsfh, param, phot, fake, out))
    cmd += ' -'.join(flags)
    cmd += ' > %s' % (msg)
    logger.debug(cmd)
    #print cmd
    p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE, close_fds=True)
    stdout, stderr = (p.stdout, p.stderr)
    err = p.wait()

    if err != 0:
        logger.error(p.stderr.readlines())
    if bg_file != 0:
        os.remove(bg_file)
    return out

def match_param_kwargs(filename, track_time=False):
    '''
    a horribly ugly way to read a file into kwargs. not caring....
    '''
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    (imf, dmod, dmod2, ddmod, Av, Av2, dAv) = map(float, lines[0].strip().split())
    zline = map(float, lines[1].strip().split())
    if len(zline) <= 3:
        (logzmin, logzmax, dlogz) = zline[:3]
        (ilogzmin, ilogzmax, flogzmin, flogzmax) = (None, None, None, None)
    else:
        (logzmin, logzmax, dlogz, ilogzmin, ilogzmax, flogzmin, flogzmax) = zline
    
    (bf, bad0, bad1) = map(float, lines[2].strip().split())
    Ncmds = int(lines[3].strip())
    (dmag, dcol, fake_sm, colmin, colmax) = map(float, 
                                                lines[4].strip().split()[:5])
    (filter1, filter2) = lines[4].strip().split()[-1].split(',')
    (bright1, faint1) = map(float, lines[5].strip().split()[:2])
    (bright2, faint2) = map(float, lines[6].strip().split()[:2])
    (nexclude_gates, ncombine_gates) = map(int, lines[7].strip().split())
    if track_time is True:
        print 'write some code if you want to care about time bins.'
    
    kwargs = {'imf': imf, 'dmod': dmod, 'dmod2': dmod2, 'ddmod': ddmod,
              'Av': Av, 'Av2': Av2, 'dAv': dAv, 'logzmin': logzmin,
              'logzmax': logzmax, 'dlogz': dlogz, 'ilogzmin': ilogzmin,
              'ilogzmax': ilogzmax, 'flogzmin': flogzmin, 'flogzmax': flogzmax,
              'logzmin': logzmin, 'logzmax': logzmax, 'dlogz': dlogz, 
              'ilogzmin': ilogzmin, 'ilogzmax': ilogzmax, 'flogzmin': flogzmin,
              'flogzmax': flogzmax, 'bf': bf, 'bad0': bad0, 'bad1': bad1,
              'Ncmds': Ncmds, 'dmag': dmag, 'dcol': dcol, 'fake_sm': fake_sm, 
              'colmin': colmin, 'colmax': colmax, 'filter1': filter1,
              'filter2': filter2, 'bright1': bright1, 'faint1': faint1,
              'bright2': bright2, 'faint2': faint2,
              'nexclude_gates': nexclude_gates,
              'ncombine_gates': ncombine_gates}
    return kwargs

def calcsfh_dict():
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
            'bg_hess': -1, # if it's not a hess, set to +1 see match readme
            'smooth': 1,
            'ilogzmin': -2.3,
            'ilogzmax': -1.3,
            'flogzmin': -1.9,
            'flogzmax': -1.1,
            'match_bg': None}

class calcsfh_params(object):
    '''
    someday, this will be a generic parameter house... someday. someday.
    '''
    def __init__(self, default_dict=None):
        if default_dict is None:
            default_dict = calcsfh_dict()
        if len(default_dict) == 0: 
            print 'need values in default dictionary.'
            return -1
        
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
                            inputs=None, **kwargs):
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
    line += '%(dobg)i %(bg_hess)i %(smooth)i%(match_bg)s\n'
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


