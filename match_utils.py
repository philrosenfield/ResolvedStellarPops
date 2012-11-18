import ResolvedStellarPops as rsp
import numpy as np
from BRparams import *
import logging
from subprocess import PIPE, Popen
import matplotlib.pyplot as plt
logger = logging.getLogger()

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


def read_match_cmd(filename):
    '''
    reads MATCH .cmd file
    '''
    mc = open(filename, 'r').readlines()
    # I don't know what the 7th column is, so I call it lixo.
    names = ['mag', 'color', 'Nobs', 'Nsim', 'diff', 'sig', 'lixo']
    cmd = np.genfromtxt(filename, skip_header=4, names=names)
    return cmd


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

def write_match_bg(color, mag, filename):
    np.savetxt(filename, np.column_stack((color, mag)), fmt='%.4f')
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

    

def make_calcsfh_param_file(match_bg, **kwargs):
    '''
    bg.dat has limited number of characters!!

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
      To Tf (for each time bin)
    optional background bins

    also need match_bg for the background file and the mag, col limits.
    not implemented: search over range of dmod, av and ntbins>0.

    input:
    match_bg

    kwargs:
    photometry:
    dmod
    Av
    filter1
    filter2

    will build these three from match_bg name and MODELS_LOC
    param_dir place to put the param file
    pmfilename name of the param file
    pmfile full path to param file
    color mag arrays:
    color
    mag
    (if not color, mag, must give bright1, faint1, bright2, faint2)
    '''
    param_dir = kwargs.get('param_dir',
                           os.path.join(MODELS_LOC, 'MATCH', 'params'))
    pmfilename = kwargs.get('pmfilename',
                            rsp.fileIO.replace_ext(os.path.split(match_bg)[1],
                                                   '.par'))
    pmfile = kwargs.get('pmfile',
                        os.path.join(param_dir, pmfilename))
    dmod = kwargs.get('dmod')
    Av = kwargs.get('Av')
    filter1 = kwargs.get('filter1')
    filter2 = kwargs.get('filter2')

    bright1 = kwargs.get('bright1')
    faint1 = kwargs.get('faint1')
    bright2 = kwargs.get('bright2')
    faint2 = kwargs.get('faint2')
    color = kwargs.get('color')
    mag = kwargs.get('mag')

    # Defaults:
    imf = kwargs.get('imf', 1.30)
    dmod2 = kwargs.get('dmod2', dmod)
    Av2 = kwargs.get('Av2', Av)
    ddmod = kwargs.get('ddmod', 0.050)
    dAv = kwargs.get('dAv', 0.050)
    logzmin = kwargs.get('logzmin', -2.3)
    logzmax = kwargs.get('logzmax', 0.1)
    dlogz = kwargs.get('dlogz', 0.1)
    zinc = kwargs.get('zinc', True)
    bf = kwargs.get('bf', 0.35)
    bad0 = kwargs.get('bad0', 1e-6)
    bad1 = kwargs.get('bad1', 1e-6)
    Ncmds = kwargs.get('Ncmds', 1)
    dmag = kwargs.get('dmag', 0.1)
    dcol = kwargs.get('dcol', 0.05)
    fake_sm = kwargs.get('fake_sm', 5)
    colmin = kwargs.get('colmin', -1)
    colmax = kwargs.get('colmax', 3)
    nexclude_gates = kwargs.get('nexclude_gates', 0)
    ncombine_gates = kwargs.get('ncomine_gates', 0)
    ntbins = kwargs.get('ntbins', 0)
    dobg = kwargs.get('dobg', -1)
    bg_hess = kwargs.get('bg_hess', -1)  # if not, make +
    smooth = kwargs.get('smooth', 1)

    if color is not None:
        colmin = np.min(color)
        colmax = np.max(color)
        bright2 = np.min(mag)
        faint2 = np.max(mag)
        bright1 = bright2 + color[np.argmin(mag)]
        faint1 = faint2 + color[np.argmax(mag)]
        # Color = V-I so V = Color+I

    if zinc:
        ilogzmin = kwargs.get('ilogzmin', -2.3)
        ilogzmax = kwargs.get('ilogzmax', -1.3)
        flogzmin = kwargs.get('flogzmin', -1.9)
        flogzmax = kwargs.get('flogzmax', -1.1)

    line = ' % .2f %.2f %.2f %.3f %.2f %.2f %.3f\n' % \
           (imf, dmod, dmod2, ddmod, Av, Av2, dAv)
    if not zinc:
        line += ' % .1f %.1f %.1f\n' % \
                (logzmin, logzmax, dlogz)
    else:
        line += ' % .1f %.1f %.1f %.1f %.1f %.1f %.1f\n' % \
                (logzmin, logzmax, dlogz,
                 ilogzmin, ilogzmax, flogzmin, flogzmax)
    line += ' % .2f %f %f\n' % (bf, bad0, bad1)
    line += ' % i\n' % Ncmds
    line += ' % .1f %.2f %i %.1f %.1f %s, %s\n' % (dmag, dcol, fake_sm, colmin,
                                                   colmax, filter1, filter2)
    line += ' % .2f %.2f %s\n' % (bright1, faint1, filter1)
    line += ' % .2f %.2f %s\n' % (bright2, faint2, filter2)
    line += ' % i %i\n' % (nexclude_gates, ncombine_gates)
    line += ' % i\n' % (ntbins)
    # if ntbins != 0: something...[to tf\n for to, tf in zip(TO, TF)]
    line += ' % i %i %i%s\n' % (dobg, bg_hess, smooth, match_bg)
    line += '-1 1 -1\n'

    pm = open(pmfile, 'w')
    pm.write(line)
    pm.close()
    logger.info(' % s wrote %s' % (make_calcsfh_param_file.__name__, pmfile))
    return pmfile


def get_fit(filename):
    fh = [f.strip() for f in open(filename, 'r').readlines()
          if len(f.strip()) > 0]
    chi2 = float(fh[-2].split(':')[-1])
    fit = float(fh[-1].split(':')[-1])
    return chi2, fit


