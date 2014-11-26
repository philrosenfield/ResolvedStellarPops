"""
A hacky way to run match in parallel. Search for the best Z, IMF, COV etc for
a population or ssp.

The first reason to do this was to get SFH and HMC uncertainties for the best
value of one fixed Z. Then it was expanded to do the same but for each COV.
Then, IMF...

These functions don't run anything. They make bash script files to be run on
their own. I think it's better to separate creating files to do the runs and
doing the runs themselves.
"""
import numpy as np
import sys
from ResolvedStellarPops.fileio.fileIO import readfile, savetxt
import os
import glob

# max processors for taskset will order 0 to max_proc-1
max_proc = 11

def check_proc(nproc, cmd):
    """if nproc >= max_proc will call reset_proc"""
    if nproc >= max_proc:
        nproc, cmd = reset_proc(nproc, cmd)
    return nproc, cmd


def reset_proc(nproc, cmd):
    """add a wait signal to cmd and set nproc to 0"""
    cmd += 'wait\n'
    nproc = 0
    return nproc, cmd


def diag_calls(cmd):
    """call match plotting routine and grep the best fits to file"""
    cmd += 'python ~/research/python/ResolvedStellarPops/match/graphics.py F555W F814W\n'
    cmd += 'grep Best *.scrn | sed \'s/=/ /g\' | sort -g -r -k 8 > sorted_best_fits.dat\n'
    return cmd


def write_script(filename, cmd):
    """add diag_calls to cmd and write to filename"""
    cmd = diag_calls(cmd)
    with open(filename, 'w') as out:
        out.write(cmd)


def insert_ext(ext, string):
    """instert text before the last '.xxx' """
    strings = string.split('.')
    return '.'.join(np.insert(strings, -1, ext))


def vary_mpars(template_match_param, gmin, gmax, dg, zdisp=0.0005, flag='',
               vary='setz', zs=None):
    """
    take a match parameter file set up with the -setz option and make many
    at a given z and zdisp.
    """
    with open(template_match_param, 'r') as infile:
        lines = [l.strip() for l in infile.readlines()]

    if vary == 'setz':
        if zs is None:
            zs = np.arange(gmin, gmax + dg / 2, dg)
        gs = np.log10(zs / 0.02)
        mdisp = np.log10(zdisp / 0.02)

    if vary == 'imf':
        gs = np.arange(gmin, gmax, dg)

    fnames = []
    for i in range(len(gs)):
        if vary == 'setz':
            lines[1] = '%.3f' % mdisp
            for j in np.arange(9, len(lines)-1):
                to, tf, mh = lines[j].strip().split()
                lines[j] = '     %s %s %.3f' % (to, tf, gs[i])
                sext = 'z%g' % zs[i]

        if vary == 'imf':
            data = lines[0].split()
            data[0] = str(gs[i])
            lines[0] = ' '.join(data)
            sext = 'imf%g' % np.abs(gs[i])

        outfile = insert_ext(sext, template_match_param)
        if flag != '':
            sflag = flag.split('=')[1]
            if not sflag in outfile:
                outfile = insert_ext(sflag, outfile)
        with open(outfile, 'w') as out:
            out.write('\n'.join(lines))

        fnames.append(outfile)
    return fnames


def calcsfh_combine(phot, fake, fnames, flag='', flag0='-setz', cmd='',
                     nproc=0):
    outfiles = []
    for i, mparam in enumerate(fnames):
        cmd, nproc, outfile = call_calcsfh(phot, fake, mparam, flag0=flag0,
                                           flag=flag, nproc=nproc, cmd=cmd)
        outfiles.append(outfile)
        nproc += 1
    nproc, cmd = reset_proc(nproc, cmd)

    if flag0 == '-ssp':
        scrns = []
        # creates a outfile.dat with no header
        for mparam in fnames:
            scrn = mparam.replace('matchparam', 'scrn')
            nproc, cmd = check_proc(nproc, cmd)
            cmd += 'taskset -c %i python -c \"from ResolvedStellarPops.match.utils import strip_header; strip_header(\'%s\')\" & \n' % (nproc, scrn)
            nproc += 1
            scrns.append(scrn)
    nproc, cmd = reset_proc(nproc, cmd)

    for i, outfile in enumerate(outfiles):
        if flag0 == '-ssp':
            cmd, nproc, stats_file = call_sspcombine(scrns[i], nproc=nproc,
                                                     cmd=cmd)
        else:
            cmd, nproc, sfh_file = call_zcombine(outfile, nproc=nproc, cmd=cmd)
        nproc += 1
    nproc, cmd = reset_proc(nproc, cmd)

    return nproc, cmd


def run_grid(phot, fake, template_match_param, gmin, gmax, dg, zdisp=0.0005,
             flag='', flag0='-setz', cmd='', nproc=0, vary='setz', zs=None):
    """
    create a bash script to run calsfh, zcombine, and make plots in parallel
    """
    fnames = vary_mpars(template_match_param, gmin, gmax, dg, zdisp=zdisp,
                        flag=flag, vary=vary, zs=zs)
    nproc, cmd = calcsfh_combine(phot, fake, fnames, flag=flag, flag0=flag0,
                                 cmd=cmd, nproc=nproc)
    return nproc, cmd


def call_sspcombine(outfile, nproc=0, cmd=''):
    """add a line calling sspcombine"""
    nproc, cmd = check_proc(nproc, cmd)
    sspcombine = 'taskset -c %i $HOME/research/match2.5/bin/sspcombine' % nproc
    stats_file = outfile.replace('scrn', 'stats')

    cmd += '%s %s.dat > %s\n' % (sspcombine, outfile, stats_file)
    return cmd, nproc, stats_file


def call_calcsfh(phot, fake, mparam, flag0='-setz', flag='', nproc=0, cmd=''):
    """add a line to a script to run calcsfh"""
    nproc, cmd = check_proc(nproc, cmd)

    if flag == '':
        flag = [o for o in mparam.split('.') if 'ov' in o or 'v1.1' in o or 'hb2' in o]
        if len(flag) > 0:
            flag = '-sub=%s' % flag[0]
        else:
            flag = ''
    if 's12' in flag and not '_' in flag:
        flag = ''
    calcsfh = 'taskset -c %i $HOME/research/match2.5/bin/calcsfh' % nproc
    scrn = mparam.replace('matchparam', 'scrn')
    outfile = mparam.replace('matchparam', 'out')
    cmd += '%s %s %s %s %s -PARSEC %s %s > %s &\n' % (calcsfh, mparam, phot,
                                                      fake, outfile, flag0,
                                                      flag, scrn)
    return cmd, nproc, outfile


def call_zcombine(outfile, nproc=0, cmd='', flag='-bestonly'):
    """add a line to a script to run zcombine"""
    nproc, cmd = check_proc(nproc, cmd)

    zcombine = 'taskset -c %i $HOME/research/match2.5/bin/zcombine' % nproc
    if 'bestonly' in flag:
        sfh_file =  outfile.replace('out', 'sfh')
    else:
        sfh_file = outfile + '.zc'
    cmd += '%s %s %s > %s & \n' % (zcombine, outfile, flag, sfh_file)
    return cmd, nproc, sfh_file


def call_hmc(hmcinp, nproc=0, cmd='', flag='-tint=2.0 -nmc=10000 -dt=0.015'):
    """add a line to a script to run hybricMC"""
    nproc, cmd = check_proc(nproc, cmd)
    hmcout = hmcinp + '.mcmc'
    hmcscrn = hmcinp + '.scrn'
    hmc = 'taskset -c %i $HOME/research/match2.5/bin/hybridMC' % nproc
    cmd += '%s %s.dat %s %s > %s & \n' % (hmc, hmcinp, hmcout, flag, hmcscrn)
    return cmd, nproc, hmcout


def mcmc_run(phot, fake, match_params, cmd='', nproc=0, flags=None,
             flag0='-setz -mcdata'):
    outfiles = []
    hmc_files = []
    sfh_files = []

    # calcsfh
    for mparam in match_params:
        cmd, nproc, outfile = call_calcsfh(phot, fake, mparam, flag0=flag0,
                                           nproc=nproc, cmd=cmd)
        outfiles.append(outfile)
        nproc += 1
    nproc, cmd = reset_proc(nproc, cmd)

    # zcombine from calcsfh
    for outfile in outfiles:
        cmd, nproc, sfh_file = call_zcombine(outfile, nproc=nproc, cmd=cmd)
        sfh_files.append(sfh_file)
        nproc += 1
    nproc, cmd = reset_proc(nproc, cmd)

    # Hybrid MonteCarlo
    for outfile in outfiles:
        cmd, nproc, hmc_file = call_hmc(outfile, nproc=nproc, cmd=cmd)
        hmc_files.append(hmc_file)
        nproc += 1
    nproc, cmd = reset_proc(nproc, cmd)

    # zcombine on HMC
    flag = ' -unweighted -medbest -jeffreys -best=%s '
    for i in range(len(hmc_files)):
        cmd, nproc, zc_file = call_zcombine(hmc_files[i], flag=flag % sfh_files[i],
                                           cmd=cmd, nproc=nproc)
        nproc += 1
    nproc, cmd = reset_proc(nproc, cmd)

    write_script('mcdata_script.sh', cmd)
    return


def setz_results(fname='setz_results.dat'):
    import matplotlib.pylab as plt
    from brewer2mpl import sequential
    data = readfile(fname)
    unc = np.unique(data['COV'])
    colors = sequential.Blues[len(unc)].mpl_colors
    #colors = ['black', 'orange', 'green', 'purple', 'darkred']
    fig, ax = plt.subplots()
    for i, c in enumerate(unc):
        inds, = np.nonzero(data['COV'] == c)
        isort = np.argsort(data['Z'][inds])
        xarr = data['Z'][inds][isort]
        yarr = data['fit'][inds][isort]
        ax.plot(xarr, yarr, color='k', lw=3)
        ax.plot(xarr, yarr, color=colors[i], lw=2,
                label='$\Lambda_c=%.2f$' % c)
        ax.plot(xarr, yarr, 'o', color=colors[i], mec='white')

    for i in range(len(data)):
        ax.annotate('$%.2f, %.1f$' % (data['dmod'][i], data['Av'][i]),
                    (data['Z'][i], data['fit'][i]), fontsize=7)
    ax.legend(loc=0, frameon=False)
    ax.set_ylabel('Fit Parameter', fontsize=20)
    ax.set_xlabel('$Z$', fontsize=20)
    # not so interesting if varying IMF....
    #plt.savefig(fname.replace('.dat', '.png'))
    #print 'wrote %s' % fname.replace('.dat', '.png')
    plt.close()

    from ResolvedStellarPops.match.match_grid import MatchGrid

    mg = MatchGrid(fname, ssp=False)
    ycols = ['COV', 'COV', 'dmod']
    xcols = ['Z', 'IMF', 'Av']
    zcols = ['fit', 'fit', 'fit']
    stat = np.min
    for i in range(len(xcols)):
        ax, cb = mg.pdf_plot(xcols[i], ycols[i], zcols[i], stat=stat)
        ffmt = '_'.join([xcols[i], ycols[i], zcols[i]])
        figname = fname.replace('.dat', '_%s.png' % ffmt)
        plt.savefig(figname)
        print 'wrote %s' % figname
        plt.close()
    return


def find_best():
    """

    """
    def grab_val(s, val, v2=None, v3=None):
        try:
            s = float('.'.join(s.split('.%s' % val)[1].split('.')[:2]))
        except:
            if v2 is not None:
                try:
                    s = float('.'.join(s.split('.%s' % v2)[1].split('.')[:2]))
                except:
                    if v3 is not None:
                        s = float('.'.join(s.split('.%s' % v3)[1].split('.')[:2]))
        return s

    def get_imf(fname):
        """
        imf is the second line, second value of the screen output from match
        """
        with open(fname, 'r') as f:
            f.readline()
            line = f.readline().replace(',', '')
        return float(line.split()[1])

    with open('sorted_best_fits.dat', 'r') as inp:
        lines = inp.readlines()
    fnames, data = zip(*[l.strip().split(':Best fit:') for l in lines])
    # values from filename
    z = np.array([grab_val(f, 'z') for f in fnames])

    try:
        ov = np.array([grab_val(f, 'ov', v2='s', v3='v') for f in fnames])
    except:
        ov = np.zeros(len(z)) + 5
    cov = ov * 0.1

    # values after the filename
    _, av, _, dmod, _, fit = zip(*[d.replace(',', '').split() for d in data])
    fit = np.array(fit, dtype=float)
    av = np.array(av, dtype=float)
    dmod = np.array(dmod, dtype=float)

    # values from within the file
    imf = np.array([get_imf(f) for f in fnames])

    savetxt('setz_results.dat', np.column_stack([z, cov, av, dmod, imf, fit]),
            fmt='%.4f %.2f %.2f %.2f %.2f %6f', header='# Z COV Av dmod IMF fit\n',
            overwrite=True)

    inds = np.digitize(ov, bins=np.unique(ov))
    mparams = []
    for iov in range(len(np.unique(ov))):
        best = np.min(fit[np.nonzero(inds==iov+1)])
        ind, = np.nonzero(fit == best)
        #print lines[ind]
        mparam = fnames[ind].replace('scrn', 'matchparam')
        mparams.append(mparam)
    return mparams


def check_boundaries():
    mparam = find_best()[0]
    data = readfile('setz_results.dat')
    _, dmod0, dmod1, _, av0, av1, _ = map(float,
                                          open(mparam).readline().strip().split())
    if len(np.nonzero(data['dmod'] == dmod1)[0]) > 0:
        print 'error need to increase dmod1 past %.2f' % dmod1
    if len(np.nonzero(data['dmod'] == dmod0)[0]) > 0:
        print 'error need to decrease dmod0 past %.2f' % dmod0
    if len(np.nonzero(data['Av'] == av1)[0]) > 0:
        print 'error need to increase av1 past %.2f' % av1
    if av0 > 0:
        if len(np.nonzero(data['Av'] == av0)[0]) > 0:
            print 'error need to decrease av0 past %.2f' % av0


def call_grid(template_match_param, gmin=0.002, gmax=0.008, dg=0.0005,
              zdisp=0.0005, func='setz', dirs=None, flags=[''], cmd='',
              flag0='-setz', nproc=0, zs=None):
    phot, = glob.glob1('.', '*match')
    fake, = glob.glob1('.', '*fake')
    for f in flags:
        nproc, cmd = run_grid(phot, fake, template_match_param, gmin, gmax, dg,
                              zdisp=zdisp, vary=func, flag=f, cmd=cmd, zs=zs,
                              nproc=nproc, flag0=flag0)
    write_script('%s_script.sh' % func, cmd)
    return nproc, cmd

def main(func):
    #flags = ['-sub=s12_hb2']
    flags = ['-sub=ov%.1f' % o for o in np.arange(3.0, 6.5, 0.5)]
    zs = np.array([0.002, 0.003, 0.004, 0.006])

    if 'setz' in func:
        template_match_param, = glob.glob1('.', '*matchparam')
        call_grid(template_match_param, zs=zs, func=func, flags=flags)
    elif 'res' in func:
        # check the solution for boundary issues
        check_boundaries()
        # write the best values
        find_best()
        if 'z' in func:
            # make a plot
            setz_results()
    elif 'mcmc' in func:
        # run mcmc on the best mparams
        phot, = glob.glob1('.', '*match')
        fake, = glob.glob1('.', '*fake')
        match_params = find_best()
        mcmc_run(phot, fake, match_params)

    if 'imf' in func:
        # make a grid of varying IMF values based on either the best
        # mparams or a given mparam
        cmd = ''
        nproc = 0
        template_match_param, = glob.glob1('.', '*matchparam')
        flag0 = '-setz'
        if 'setz' in func:
            # make a grid of setz values that all vary in IMF
            mparams = glob.glob1('.', '*z0*matchparam')
        elif 'ssp' in func:
            flag0 = '-ssp'
            mparams = [template_match_param]
        else:
            mparams = find_best()
        for i in range(len(mparams)):
            nproc, cmd = call_grid(mparams[i], gmin=0.5, gmax=1.9, dg=0.1,
                                   func='imf', flags=flags, nproc=nproc,
                                   cmd=cmd, flag0=flag0)
    else:
        print('choose from setz, res, zres, mcmc, imf, imfsetz, imfssp')


if __name__ == '__main__':
    main(sys.argv[1])