import matplotlib.pyplot as plt
import numpy as np
import fileIO
import math_utils
import time
import os
import logging
from subprocess import Popen, PIPE
logger = logging.getLogger()


def find_photsys_number(photsys, filter):
    mag_file = os.path.join(os.environ['BCDIR'],
                            'tab_mag_odfnew/tab_mag_%s.dat' % photsys)
    mf = open(mag_file, 'r')
    nmags = mf.readline()
    magline = mf.readline().strip().split()
    return magline.index(filter)


def write_pytrilegal_params(sfh, parfile, photsys, filter, object_mass=1e7):
    mag_num = find_photsys_number(photsys, filter)
    lines = (
            "photosys           %s" % photsys,
            "mag_num            %s" % mag_num,
            "mag_lim            2               ",
            "mag_res            0.1             ",
            "dust               0               ",
            "dustM              dpmod60alox40   ",
            "dustC              AMCSIC15        ",
            "binary_kind        0               ",
            "binary_frac        0.              ",
            "extinction_kind    0               ",
            "thindisk_kind      0               ",
            "thickdisk_kind     0               ",
            "halo_kind          0               ",
            "bulge_kind         0               ",
            "object_kind        1               ",
            "object_mass        %g" % object_mass,
            "object_dist        10.0            ",
            "object_avkind      1               ",
            "object_av          0.000           ",
            "object_cutoffmass  0.8             ",
            "object_sfr         %s" % os.path.abspath(sfh),
            "object_sfr_A       1.              ",
            "object_sfr_B       0.0             ")

    with open(parfile, 'w') as oo:
        [oo.write(line + '\n') for line in lines]

    logger.info('wrote  %s' % parfile)
    return


def run_trilegal(cmd_input, parfile, inp, out, agb=True, tagstages=True):
    cmd = "/Users/phil/research/PyTRILEGAL/run_trilegal.py  -e code/main"
    cmd += "%s" % parfile
    if agb is True:
        cmd += " -a"
    if tagstages is True:
        cmd += " -l"
    cmd += " -i  %s" % inp
    cmd += " -o  %s" % out
    cmd += " -f  %s" % cmd_input

    logger.debug(cmd)
    p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE, close_fds=True)
    stdout, stderr = (p.stdout, p.stderr)
    lines = stdout.readlines()
    p.wait()
    # append the standard output to the end of the output file
    with open(out, 'a') as ff:
        for l in lines:
            ff.write("# %s" % l)
    
    logger.debug([l.strip() for l in lines])
    return


def get_args(filename, ext='.dat'):
    filename = os.path.split(filename)[1]
    a = filename.split(ext)[0]
    a = a.replace('_', ' ').replace('R1', '').replace('isoch', '')
    s = ''.join(c for c in a if not c.isdigit())
    s = s.replace('.', ' ').split()
    d = {}
    x = a[:]
    s.append(' ')
    for i in range(len(s) - 1):
        if s[i] in a:
            x = x.replace(s[i], '')
            y = x.split(s[i + 1])[0]
            x = x.split(s[i + 1])[-1]
            d[s[i]] = float(y)
    try:
        del d['F']
    except KeyError:
        pass
    return d


def get_stage_label(region):
    # see parametri.h
    regions = ['PMS', 'MS', 'SUBGIANT', 'RGB', 'HEB', 'RHEB', 'BHEB', 'EAGB',
               'TPAGB', 'POSTAGB', 'WD']
    stage_lab = regions.index(region.upper())
    return stage_lab


def get_label_stage(stage_lab):
    # see parametri.h
    regions = ['PMS', 'MS', 'SUBGIANT', 'RGB', 'HEB', 'RHEB', 'BHEB', 'EAGB',
               'TPAGB', 'POSTAGB', 'WD']
    return regions[stage_lab]


def get_loop_data(cmd_input_file, metallicity):
    filename = read_cmd_input_file(cmd_input_file)['file_isotrack']
    file_isotrack = os.path.join(os.environ['ISOTRACK'],
                                 filename.replace('isotrack/', ''))
    z, y, mh, files = read_isotrack_file(file_isotrack)
    if not metallicity in z:
        logger.warning('%s: metallicity not found.' % get_loop_data.__name__)
        data = 0
    ptcri = [f[0] for f in files if str(metallicity) in f[0] and
             f[0].endswith('INT2')][0].replace('isotrack/', '')
    ptcri_file = os.path.join(os.environ['ISOTRACK'], ptcri)
    d = read_loop_from_ptrci(ptcri_file)
    return d


def string_in_lines(lines):
    for line in lines:
        strs = [math_utils.is_numeric(l) for l in line.strip().split()
                if type(math_utils.is_numeric(l)) == str]
        if len(strs) != 0:
            if ' M=' in line:
                print line, strs


def read_ptcri(ptcri_file):
    d = {}
    lines = open(ptcri_file, 'r').readlines()
    for line in lines:
        if not 'kind' in line:
            continue
        try:
            age, logL, logTe, mod, model, cript, M, mass, npt, kind = \
                line.strip().split()
            try:
                float(cript)
            except ValueError:
                mass = float(mass)
                if not mass in d.keys():
                    d[mass] = {}
                if not cript in d[mass].keys():
                    d[mass][cript] = {}
                d[mass][cript] = {'age': float(age),
                                  'logTe': float(logTe),
                                  'logL': float(logL),
                                  'model': np.int(model),
                                  'mod': np.int(mod)}
        except ValueError:
            pass
    return d


def read_ptcri2(ptcri_file):
    '''
    after .revisegrid
    '''
    lines = open(ptcri_file, 'r').readlines()
    mline = []
    for i, line in enumerate(lines):
        if 'M=' in line:
            mline.append(i)
            continue
    d = {}
    for j in range(len(mline) - 1):
        mass = float(lines[mline[j]].split('M=')[-1].split('PMS')[0])
        for k in range(mline[j] + 1, mline[j + 1] - 1):
            try:
                a, l, t, m = map(float, lines[k].strip().split())
            except:
                data = lines[k].strip().split()
                age, logL, logTe = map(float, data[0:3])
                cript = data[-2]
                model = int(data[-1])
                if not mass in d.keys():
                    d[mass] = {}
                if not cript in d[mass].keys():
                    d[mass][cript] = {}
                d[mass][cript] = {'age': age, 'logTe': logTe, 'logL': logL,
                                  'model': model}
        if mass == 12:
            break
    return d


def read_loop_from_ptrci(ptcri_file):
    lines = open(ptcri_file, 'r').readlines()
    # Get starting points for each track
    line_nums = map(int, lines[4].strip().split())
    # don't forget the first
    line_nums.insert(0, 0)
    # they actually start at line 7, that is 6 if you start counting with 0.
    line_nums = np.array(line_nums) + 6
    mass_lines = np.cumsum(line_nums)
    #ax = plt.axes()
    #col = {'Loop_A': 'red', 'Loop_B': 'blue', 'Loop_C': 'green'}
    d = {}
    for i, line in enumerate(lines):
        if not 'Loop' in line and not 'M=' in line:
            continue
        if 'M=' in line:
            mass = float(line.split('M=')[-1].split('P')[0])
            mline = i
        if 'Loop' in line:
            data = line.strip().split()
            age, logL, logTe = map(float, data[0:3])
            loop = data[3]
            model = int(data[4])
            #ax.plot(logTe, logL, 'o', color = col[loop])
            if not mass in d.keys():
                d[mass] = {}
            if not loop in d[mass].keys():
                d[mass][loop] = {}
            d[mass][loop] = {'age': age, 'logTe': logTe, 'logL': logL,
                             'model': model}
    return d


def read_isotrack_file(filename):
    if os.path.split(filename)[1].startswith('cmd_input'):
        file_isotrack = read_cmd_input_file(filename)['file_isotrack']
        filename = os.path.join(os.environ['ISOTRACK'],
                                file_isotrack.replace('isotrack / ', ''))
    try:
        isotf = open(filename, 'r').readlines()
    except:
        fname = os.path.join(os.environ['ISOTRACK'], 'parsec', filename)
        isotf = open(fname, 'r').readlines()
    nmets = int(isotf[0])
    z = np.zeros(nmets)
    y = np.zeros(nmets)
    mh = np.zeros(nmets)
    files = []
    met_count = 0
    for j in range(len(isotf)):
        if  j == 0:
            continue
        line = isotf[j].strip().split(' ')
        if len(line) == 3:
            z[met_count], y[met_count], mh[met_count] = map(float, line)
            met_count += 1
        else:
            files.append(line)
    return z, y, mh, files


def read_cmd_input_file(filename):
    lines = open(filename, 'r').readlines()
    d = {}
    for line in lines:
        if line.startswith('##'):
            break
        data_line, comment_line = line.strip().split('#')
        data = data_line.split()
        comment = comment_line.strip().split('DA ')[0].split(', ')
        for c, dat in zip(comment, data):
            d[c.strip().replace(' ', '_')] = math_utils.is_numeric(dat)
    return d


def change_trilegal_input_file(input_file, over_write=True, **kwargs):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    for k, v in kwargs.items():
        try:
            (i, line), = [(i, l) for i, l in enumerate(lines) if k in l]
        except ValueError:
            print '%s not found' % k
            continue
        vals, info = line.strip().split('#')
        val_ind = info.strip().index(k)
        old_vals = map(float, vals.split())
        if over_write is False:
            print 'current: %s=%g:' % (k, old_vals[val_ind])
            return old_vals[val_ind]
        
        new_vals = old_vals.copy()
        new_vals[val_ind] = v
        new_line = '%s # %s\n' % (' '.join(['%g' % x for x in new_vals]), info)
        print 'new line: %s' % new_line.strip()
        lines[i] = new_line

    with open(input_file, 'w') as o:
        [o.write(l) for l in lines]

    return

def write_cmd_input_file(**kwargs):
    '''
    make a TRILEGAL cmd_input file based on default.

    Send each parameter that is different than default by:
    kwargs = { 'kind_tpagb': 4, 'file_tpagb': 'isotrack/tracce_CAF09_S0.dat'}
    cmd_input_file = write_cmd_input_file(**kwargs)

    To make the default file:
    cmd_input_file = write_cmd_input_file()

    if you don't specify cmd_input_file, output goes to cmd_input_TEMP.dat
    '''
    kind_tracks = kwargs.get('kind_tracks', 2)
    file_isotrack = kwargs.get('file_isotrack', 'isotrack_parsec/CAF09.dat')
    file_lowzams = kwargs.get('file_lowzams', 'isotrack/bassazams_fasulla.dat')
    kind_tpagb = kwargs.get('kind_tpagb', 4)
    # should be isotrack_agb...
    file_tpagb = kwargs.get('file_tpagb',
                            'isotrack/tracce_CAF09_AFEP02_I1_S1.dat')
    kind_postagb = kwargs.get('kind_postagb', 0)
    file_postagb = kwargs.get('file_postagb', 'isotrack/final/pne_wd_test.dat')

    # these are for using cmd2.2:
    kind_mag = kwargs.get('kind_mag', None)
    photsys = kwargs.get('photsys', 'wfpc2')
    file_mag = 'tab_mag_odfnew/tab_mag_' + photsys + '.dat'
    kind_imf = kwargs.get('kind_imf', None)
    file_imf = kwargs.get('file_imf', 'tab_imf/imf_chabrier_lognormal.dat')

    # if not using cmd2.2:
    if kind_imf is None:
        kind_imfr = kwargs.get('kind_imfr', 0)
        file_imfr = kwargs.get('file_imfr', 'tab_ifmr/weidemann.dat')

    track_comments = '# kind_tracks, file_isotrack, file_lowzams'
    tpagb_comments = '# kind_tpagb, file_tpagb'
    pagb_comments = '# kind_postagb, file_postagb DA VERIFICARE file_postagb'
    mag_comments = '# kind_mag, file_mag'
    imf_comments = '# kind_imf, file_imf'
    imfr_comments = '# ifmr_kind, file with ifmr'
    footer = (
        "################################explanation######################\n"
        "kind_tracks: 1 = normal file\n"
        "file_isotrack: tracks for low + int mass\n"
        "file_lowzams: tracks for low - ZAMS\n"
        "kind_tpagb:\n"
        "    0 = none\n"
        "    1 = Girardi et al., synthetic on the flight, no dredge up\n"
        "    2 = Marigo & Girardi 2001, from file, includes mcore and C/O\n"
        "    3 = Marigo & Girardi 2007, from file, includes period,\n"
        "        mode and mloss\n"
        "    4 = Marigo et al. 2011, from file, includes slope\n"
        "file_tpagb: tracks for TP - AGB\n"
        "kind_postagb:\n"
        "    0 = none\n"
        "    1 = from file\n"
        "file_postagb: PN + WD tracks\n"
        "kind_ifmr: 0 = default\n"
        "           1 = from file\n")
    cmd_input_file = kwargs.get('cmd_input_file', 'cmd_input_TEMP.dat')
    fh = open(cmd_input_file, 'w')
    formatter = '%i %s %s \n'
    fh.write('%i %s %s %s \n' % (kind_tracks, file_isotrack, file_lowzams,
                                 track_comments))
    fh.write(formatter % (kind_tpagb, file_tpagb, tpagb_comments))
    fh.write(formatter % (kind_postagb, file_postagb, pagb_comments))
    if kind_mag is not None:
        fh.write(formatter % (kind_mag, file_mag, mag_comments))
    if kind_imf is None:
        fh.write(formatter % (kind_imfr, file_imfr, imfr_comments))
    else:
        fh.write(formatter % (kind_imf, file_imf, imf_comments))
    fh.write(footer)
    fh.close()
    logger.info('%s wrote %s' % (write_cmd_input_file.__name__,
                                 cmd_input_file))
    return cmd_input_file


def write_trilegal_input_file(**kwargs):
    default_file = os.path.join(os.environ['TRILEGAL_ROOT'],
                                'input_default.dat')


def colorplot_by_bin(x, y, marker, z, bins=None, labels=[], **kwargs):
    if bins is None:
        bins = np.linspace(min(z), max(z), 8)
    for bin in bins:
        labels.append('%.3f' % bin)
    colors = kwargs.get('colors', GraphicsUtils.discrete_colors(len(bins)))
    ax = kwargs.get('ax', plt.axes())
    for i in range(len(bins) - 1):
        # bins go 0 to bin[0], bin[i-1] to bin[i], bin[-1] and up
        if i == 0:
            ind = np.nonzero(z < bins[i])[0]
        elif i == len(bins) - 1:
            ind = np.nonzero(z > bins[i])[0]
        else:
            ind = np.nonzero((z < bins[i]) & (z > bins[i - 1]))[0]
        if len(ind) == 0:
            continue
        if labels is None:
            ax.plot(x[ind], y[ind], marker, color=colors[i])
        else:
            ax.plot(x[ind], y[ind], marker, label=labels[i], color=colors[i])
    return ax


def color_color_diagnostic(trilegal_output_file, filter1, filter2, filter3,
                           filter4, ax=None, **pltkwargs):
    logger.info('reading %s' % trilegal_output_file)
    tstart = time.time()
    t = fileIO.read_table(trilegal_output_file)
    tend = time.time()
    logger.info('reading took %.2f seconds' % (tend - tstart))
    #mag1 = np.array(t[filter1])
    #mag2 = np.array(t[filter2])
    #mag3 = np.array(t[filter3])
    #mag4 = np.array(t[filter4])

    mag1 = t.get_col(filter1)
    mag2 = t.get_col(filter2)
    mag3 = t.get_col(filter3)
    mag4 = t.get_col(filter4)

    if ax is None:
        ax = plt.axes()

    ax.plot(np.array(mag1) - np.array(mag2), np.array(mag3) - np.array(mag4),
            ',', label=trilegal_output_file, **pltkwargs)
    ax.set_ylabel('$%s - %s$' % (filter1, filter2))
    ax.set_xlabel('$%s - %s$' % (filter3, filter4))
    return ax

def mc_tests(ID, dir_name, outdir="MC_TESTS", bestfit_loc="MODELS/"):
    '''
    ID is the Galaxy name
    dir_name is PropID_GalaxyName
    Will put mc tests in MC_TESTS/*/dirname
    '''

    # Make new folders
    folders = ['PARS/', 'INPUT/', 'OUTPUT/']
    outdir = fileIO.ensure_dir(outdir)
    dummy = [fileIO.ensure_dir(os.path.join(outdir, folder))
             for folder in folders]
    this_outdir = fileIO.ensure_dir(os.path.join(outdir, folder, dir_name))

    # A place to write the commands if this needs to be run again
    cmds_out = fileIO.ensure_dir(os.path.join(outdir, 'MC_COMMANDS/'))
    cmds_file = open('%s/%s_MC_commands.dat' % (cmds_out, ID), 'r')

    # Place for the output table
    table_out = fileIO.ensure_dir(os.path.join(outdir, 'TABLES/'))
    out = open('%s/%s_MC.dat' % (table_out, ID), 'w')
    out.write("# mc_ID p_value NRGB_data NAGB_data NRGB_model "
              "NAGB_model mass_model N_wind Flux1_wind Flux2_wind\n")

    # best fits:
    # bfs[0] = pars for run_trilegal,
    # bfs[1] = input for trilegal
    # bfs[2] = output for trilegal
    bfs = [get_afile(bestfit_loc + folder, '*' + ID + '*')[0]
           for folder in folders]

    # find model from file!!! This is not general!!
    model = bfs[-1].split('model_')[-1].replace('.dat.dat', '.dat')

    # Switch to the MC test directories
    new_place = [bf.replace(bestfit_loc, 'MC_TESTS/').replace(folder, folder +
                 dir_name + '/') for bf in bfs]

    # Best fit pars is the template, we'll swap out sfh file
    in_pars = open(bfs[0]).readlines()

    # Load SFHs
    sfhs = get_afile(mc_dir + '/', 'SFR*mc*dat')

    for i, sfh in enumerate(sfhs):
        mcid = sfh.split('/')[-1].split('.')[1]  # need to change jason / me
        new_names = [np.replace(ext, '.' + mcid + ext)
                     for np, ext in zip(new_place,
                                        ['.pars', '.dat', '.dat.dat'])]

        # New pars file for run_trilgal.py
        pfile = open(new_names[0], 'w')
        # sfh line is at the bottom.
        [pfile.write(inpar) for inpar in in_pars[:-1]]
        pfile.write("%-18s %s\n" % ('object_sfr  ', sfh))
        pfile.close()

        cmd = "/Users/Phil/research/Italy/WXTRILEGAL/run_trilegal.py "
        # cmd = "/home/philrose/WXTRILEGAL/run_trilegal.py "
        cmd += "-e code_2.0/main "
        cmd += "-a "
        cmd += "-i %s " % new_names[1]
        cmd += "-o %s " % new_names[2]
        cmd += "-f %s " % model
        cmd += new_names[0]

        cmds_file.write('%s \n' % cmd)
        print 'running TRILEGAL: ', model, ID
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        p = subprocess.Popen(cmd, shell=True)
        sts = os.waitpid(p.pid, 0)[1]

        fak_file = get_fakFILE(ID, jason=jason)
        tmp = new_names[2].split('/')[-1]
        ast_file = new_names[2].split('/')[0] + '/ast_' + tmp
        cmd = "AST/code/spread_angst <<EOF \n"
        cmd += fak_file + "\n"
        cmd += new_names[2] + "\n"
        cmd += ast_file + "\n"
        cmd += "EOF \n"
        print "  ... completeness using %s" % fak_file
        print "  %s -> %s" % (new_names[2], ast_file)
        print 'Running spread_angst...'
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        sts = os.waitpid(p.pid, 0)[1]
        os.system("wc  -l %s %s|head -2" % (new_names[2], ast_file))

        cmds_file.write('%s \n' % cmd)

        synthcmd = fileIO.read_table(ast_file)
        tmp2 = synthcmd.get_col('diff_mag2'.strip())
        tmp1 = synthcmd.get_col('diff_mag1'.strip())
        s_mag2 = synthcmd.get_col('mag2') + tmp2
        s_mag2 = s_mag2[np.nonzero(abs(tmp2) < 90.)[0]]
        s_mag1 = synthcmd.get_col('mag1') + tmp1
        s_mag1 = s_mag1[np.nonzero(abs(tmp1) < 90.)[0]]
        s_color = s_mag1 - s_mag2
        Norm = trgb + 1.5
        # this no longer works...
        #ind, nB_AGB, nNorm, ps_nNorm, ps_nB_AGBm, hist, bins, s_hist_normed,
        # p_value, normalization = calc_LF(mag2, s_mag2, Norm, trgb)
        #Nstars, flux_rates = flux_from_mass_loss(synthcmd, rates,
        #[filt1, filt2], ast_inds=ind, rel_flux=True)
        #out.write('# mc_ID p_value NRGB_data NAGB_data NRGB_model NAGB_model
        # mass_model N_wind Flux1_wind Flux2_wind\n')
        #out.write('%s  %.3f  %i  %i  %i  %i  %e  %i  %e  %e \n' %
        #           (mcid, p_value, nNorm, nB_AGB, ps_nNorm, ps_nB_AGBm,
        #            object_mass, Nstars[0], flux_rates[0][1],
        #            flux_rates[1][0]))

        os.remove(ast_file)
        print 'deleted', ast_file
        os.remove(new_names[2])
        print 'deleted', new_names[2]

    out.close()
    cmds_file.close()
