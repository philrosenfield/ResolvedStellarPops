"""
A hacky way to run match in a loop... find the best fixed Z, COV value for a
population and run those with mcdata. The first reason to do this was to
get SFH and HMC uncertainties for the best value of one fixed Z. Then it was
expanded to do the same but for each COV.

These functions don't run anything. They make bash script files to be run on
their own. I think it's better to separate creating files to do the runs and
doing the runs themselves.
"""
import numpy as np
import sys
from ResolvedStellarPops.fileio.fileIO import readfile, savetxt
import glob

# max processors for taskset will order 0 to max_proc-1
max_proc = 8

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
               vary='setz'):
    """
    take a match parameter file set up with the -setz option and make many
    at a given z and zdisp.
    """
    with open(template_match_param, 'r') as infile:
        lines = [l.strip() for l in infile.readlines()]

    if vary == 'setz':
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


def calcsfh_zcombine(phot, fake, fnames, flag='', flag0='-setz', cmd='',
                     nproc=0):
    outfiles = []
    for i, mparam in enumerate(fnames):
        cmd, nproc, outfile = call_calcsfh(phot, fake, mparam, flag0=flag0,
                                           flag=flag, nproc=nproc, cmd=cmd)
        outfiles.append(outfile)
        nproc += 1
    nproc, cmd = reset_proc(nproc, cmd)

    for i, outfile in enumerate(outfiles):
        cmd, nproc, sfh_file = call_zcombine(outfile, nproc=nproc, cmd=cmd)
        nproc += 1
    nproc, cmd = reset_proc(nproc, cmd)

    return nproc, cmd


def run_grid(phot, fake, template_match_param, gmin, gmax, dg, zdisp=0.0005,
             flag='', flag0='-setz', cmd='', nproc=0, vary='setz'):
    """
    create a bash script to run calsfh, zcombine, and make plots in parallel
    """
    fnames = vary_mpars(template_match_param, gmin, gmax, dg, zdisp=zdisp,
                        flag=flag, vary=vary)

    nproc, cmd = calcsfh_zcombine(phot, fake, fnames, flag=flag, flag0=flag0,
                                  cmd=cmd, nproc=nproc)
    return nproc, cmd

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
    data = readfile(fname)
    unc = np.unique(data['COV'])
    colors = ['black', 'orange', 'green', 'purple', 'darkred']
    fig, ax = plt.subplots()
    for i, c in enumerate(unc):
        inds, = np.nonzero(data['COV'] == c)
        isort = np.argsort(data['Z'][inds])
        xarr = data['Z'][inds][isort]
        yarr = data['fit'][inds][isort]
        ax.plot(xarr, yarr, color=colors[i], alpha=0.5, label='$\Lambda_c=%.1f$' % c)
        ax.plot(xarr, yarr, 'o', color=colors[i], alpha=0.5)

    for i in range(len(data)):
        ax.annotate('$%.2f, %.1f$' % (data['dmod'][i], data['Av'][i]),
                    (data['Z'][i], data['fit'][i]), fontsize=7)
    ax.legend(loc=0, frameon=False)
    ax.set_ylabel('Fit Parameter', fontsize=20)
    ax.set_xlabel('$Z$', fontsize=20)
    plt.savefig(fname.replace('.dat', '.png'))
    plt.close()
    return


def find_best():
    """

    """
    with open('sorted_best_fits.dat', 'r') as inp:
        lines = inp.readlines()
    fnames, data = zip(*[l.strip().split(':Best fit:') for l in lines])
    _, av, _, dmod, _, fit = zip(*[d.replace(',', '').split() for d in data])
    fit = np.array(fit, dtype=float)
    av = np.array(av, dtype=float)
    dmod = np.array(dmod, dtype=float)
    try:
        ov = np.array([f.split('.')[-2].replace('ov', '').replace('s', '').replace('v', '')
                       for f in fnames], dtype=int)
    except:
        ov = np.zeros(len(dmod)) + 5
    z = np.array(['0.' + f.split('.')[-3] for f in fnames], dtype=float)

    cov = ov * 0.1
    savetxt('setz_results.dat', np.column_stack([z, cov, av, dmod, fit]),
            fmt='%.4f %.1f %.2f %.2f %6f', header='# Z COV Av dmod fit\n',
            overwrite=True)

    inds = np.digitize(ov, bins=np.unique(ov))
    mparams = []
    for iov in range(len(np.unique(ov))):
        best = np.min(fit[np.nonzero(inds==iov+1)])
        ind, = np.nonzero(fit == best)
        print lines[ind]
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
              zdisp=0.0005, func='setz', dirs=None, flags=['']):
    cmd = ''
    nproc = 0
    phot, = glob.glob1('.', '*match')
    fake, = glob.glob1('.', '*fake')
    for f in flags:
        nproc, cmd = run_grid(phot, fake, template_match_param, gmin, gmax, dg,
                              zdisp=zdisp, vary=func, flag=f, cmd=cmd,
                              nproc=nproc)
    write_script('%s_script.sh' % func, cmd)

def call_mcmc_run():
    phot, = glob.glob1('.', '*match')
    fake, = glob.glob1('.', '*fake')
    match_params = find_best()
    mcmc_run(phot, fake, match_params)

if __name__ == '__main__':
    func = sys.argv[1]
    flags = ['-sub=s12_hb2']
    if 'setz' in func:
        template_match_param, = glob.glob1('.', '*matchparam')
        #flags = ['-sub=%s' % s for s in ['s12', 'ov3', 'ov4', 'ov5', 'ov6']]
        call_grid(template_match_param, gmax=0.0079, func=func, flags=flags)
    elif 'zres' in func:
        check_boundaries()
        find_best()
        setz_results()
    elif 'mcmc' in func:
        call_mcmc_run()
    elif 'imf' in func:
        call_grid(sys.argv[2], gmin=-0.5, gmax=-0.1, dg=0.1, func=func, flags=flags)
    else:
        print('choose from setz, zres, mcmc, imf mparam')
