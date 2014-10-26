'''
Interpolate PARSEC tracks for use in MATCH.
This code calls padova_tracks which does three main things:
1) Redefines some equivalent evolutionary points (EEPs) from PARSEC
2) Interpolates the tracks so they all have the same number of points between
   defined EEPs.
3) 
'''
from __future__ import print_function
from copy import deepcopy
import numpy as np
import os
import sys

from ResolvedStellarPops import fileio
from eep.define_eep import DefineEeps
from eep.critical_point import critical_point, Eep
from match import TracksForMatch
from match import CheckMatchTracks

def parsec2match(input_obj, loud=False):
    '''do an entire set and make the plots'''
    if loud:
        print('setting prefixs')
    prefixs = set_prefixs(input_obj)

    for prefix in prefixs:
        print('Current mix: %s' % prefix)
        inps = set_outdirs(input_obj, prefix)

        if loud:
            print('loading ptcri')
        inps = load_ptcri(inps)

        if loud:
            print('loading Tracks')
        tfm = TracksForMatch(inps)

        if not inps.from_p2m:
            # find the parsec2match eeps for these tracks.
            ptcri_file = load_ptcri(inps, find=True, from_p2m=True)
            if not inps.overwrite_ptcri and os.path.isfile(ptcri_file):
                print('not overwriting %s' % ptcri_file)
            else:
                if loud:
                    print('defining eeps')
                tfm = define_eeps(tfm, inps)

            if inps.diag_plot and not inps.do_interpolation:
                # make diagnostic plots using new ptcri file
                inps.ptcri_file = None
                inps.from_p2m = True
                inps = load_ptcri(inps)
                pat_kw = {'ptcri': inps.ptcri}
                if loud:
                    print('making parsec diag plots')
                if inps.hb:
                    tfm.diag_plots(tfm.hbtracks, hb=inps.hb, pat_kw=pat_kw,
                                   extra='parsec', plot_dir=inps.plot_dir)
                else:
                    #tfm.diag_plots(xcols=xcols, pat_kw=pat_kw)
                    tfm.diag_plots(tfm.tracks, pat_kw=pat_kw, extra='parsec',
                                   plot_dir=inps.plot_dir)

        # do the match interpolation (produce match output files)
        if inps.do_interpolation:
            # force reading of my eeps
            inps.ptcri_file = None
            inps.from_p2m = True
            inps = load_ptcri(inps)

            if loud:
                print('doing match interpolation')
            inps.flag_dict = tfm.match_interpolation(inps)

            # check the match interpolation
            if loud:
                print('checking interpolation')
            CheckMatchTracks(inps)

    print('DONE')
    return prefixs


def set_prefixs(inputs, harsh=True):
    '''
    find which prefixes (Z, Y mixes) to run based on inputs.prefix or
    inputs.prefixs.
    '''
    # tracks location
    tracks_dir = inputs.tracks_dir
    if inputs.prefixs == 'all':
        # find all dirs in tracks dir skip .DS_Store and crap
        prefixs = [d for d in os.listdir(tracks_dir)
                   if os.path.isdir(os.path.join(tracks_dir, d))
                   and not d.startswith('.')]
    elif inputs.prefixs is not None:
        # some subset listed in the input file (seperated by comma)
        prefixs = inputs.prefixs
    else:
        if inputs.prefix is None:
            print('prefix or prefixs not set')
            sys.exit(2)
        # just do one
        prefixs = [inputs.prefix]

    if harsh:
        del inputs.prefixs
    assert type(prefixs) == list, 'prefixs must be a list'
    return prefixs


def load_ptcri(inputs, find=False, from_p2m=False):
    '''
    load the ptcri file, either sandro's or mine
    if find is True, just return the file name, otherwise, return inputs with
    ptcri and ptcri_file attributes set.

    if from_p2m is True, force find/load my ptcri file regardless
    of inputs.from_p2m value.
    '''
    # find the ptcri file
    if inputs.from_p2m or from_p2m:
        sandro = False
        search_term = 'p2m_p*'
        if inputs.hb:
            search_term = 'p2m_hb*'
        #print('reading ptcri from saved p2m file.')
    else:
        sandro = True
        search_term = 'pt*'

    search_term += '%s*dat' % inputs.prefix
    if inputs.ptcri_file is not None:
        ptcri_file = inputs.ptcri_file
    else:
        ptcri_file, = fileio.get_files(inputs.ptcrifile_loc, search_term)

    if find:
        return ptcri_file
    else:
        inputs.ptcri_file = ptcri_file
        inputs.ptcri = critical_point(inputs.ptcri_file, sandro=sandro)
        return inputs


def define_eeps(tfm, inputs):
    '''add the ptcris to the tracks'''
    # assign eeps track.iptcri and track.sptcri
    de = DefineEeps()
    crit_kw = {'plot_dir': inputs.plot_dir,
               'diag_plot': inputs.track_diag_plot,
               'debug': inputs.debug}

    crit_kw['hb'] = inputs.hb
    track_str = 'tracks'
    defined = inputs.ptcri.please_define
    filename = 'define_eeps_%s.log'
    if inputs.hb:
        track_str = 'hbtracks'
        defined = inputs.ptcri.please_define_hb
        filename = 'define_eeps_hb_%s.log'
    # load critical points calles de.define_eep
    tracks = [de.load_critical_points(track, ptcri=inputs.ptcri, **crit_kw)
              for track in tfm.__getattribute__(track_str)]

    # write log file
    info_file = os.path.join(inputs.outfile_dir,
                             filename % inputs.prefix.lower())
    with open(info_file, 'w') as out:
        for t in tracks:
            out.write('# %.3f\n' % t.mass)
            if t.flag is not None:
                out.write(t.flag)
            else:
                [out.write('%s: %s\n' % (ptc, t.info[ptc])) for ptc in defined]
    if not inputs.from_p2m:
        inputs.ptcri.save_ptcri(tracks, hb=inputs.hb)

    tfm.__setattr__(track_str, tracks)
    return tfm


def set_outdirs(input_obj, prefix):
    '''
    set up the directories for output and plotting
    returns a copy of input_obj
    '''
    new_inputs = deepcopy(input_obj)
    new_inputs.prefix = prefix

    if input_obj.plot_dir == 'default':
        new_inputs.plot_dir = os.path.join(input_obj.tracks_dir,
                                           'diag_plots',
                                           new_inputs.prefix)
        fileio.ensure_dir(new_inputs.plot_dir)

    if input_obj.outfile_dir == 'default':
        new_inputs.outfile_dir = os.path.join(input_obj.tracks_dir,
                                              'match',
                                              new_inputs.prefix)
        fileio.ensure_dir(new_inputs.outfile_dir)

    return new_inputs


def initialize_inputs():
    '''
    Load default inputs, the eep lists, and number of equally spaced points
    between eeps. Input file will overwrite these. These should be all possible
    input options.
    '''
    input_dict = {'track_search_term': '*F7_*PMS',
                  'hbtrack_search_term':'*F7_*HB',
                  'from_p2m': False,
                  'masses': None,
                  'do_interpolation': True,
                  'debug': False,
                  'hb': True,
                  'prefix': None,
                  'prefixs': None,
                  'tracks_dir': os.getcwd(),
                  'ptcrifile_loc': None,
                  'ptcri_file': None,
                  'plot_dir': None,
                  'outfile_dir': None,
                  'diag_plot': False,
                  'match': False,
                  'agb': False,
                  'overwrite_ptcri': True,
                  'overwrite_match': True,
                  'prepare_makemod': False,
                  'track_diag_plot': False}
    return input_dict


def prepare_makemod(inputs):
    zsun = 0.02
    prefixs = set_prefixs(inputs, harsh=False)
    zs = np.unique(np.array([p.split('Z')[1].split('_')[0] for p in prefixs],
                            dtype=float))

    # limits of metallicity grid
    modelIZmin = np.int(np.floor(np.log10(np.min(zs) / zsun) * 10))
    modelIZmax = np.int(np.ceil(np.log10(np.max(zs) / zsun) * 10))

    # metallicities
    zs_str = ','.join(np.array(zs, dtype=str))
    # max number of characters needed for directory names
    FNZ = np.max([len(p) for p in prefixs]) + 1.

    # directories
    prefix_str = '\"%s\"' % '\",\"'.join(prefixs)

    all_masses = np.array([])
    # masses N x len(zs) array
    for p in prefixs:
        this_dir = os.path.join(inputs.tracks_dir, p)
        track_names = fileio.get_files(this_dir, '*.PMS')
        masses = np.array([os.path.split(t)[1].split('M')[1].replace('.P', '')
                           for t in track_names], dtype=float)
        all_masses = np.append(all_masses, masses)

    # find a common low and high mass at all Z.
    umasses = np.sort(np.unique(all_masses))
    min_mass = umasses[0]
    max_mass = umasses[-1]
    imin = 0
    imax = -1
    while len(np.nonzero(all_masses == min_mass)[0]) != len(zs):
        imin += 1
        min_mass = umasses[imin]

    while len(np.nonzero(all_masses == max_mass)[0]) != len(zs):
        imax -= 1
        max_mass = umasses[imax]

    if imax == -1 and imin == 0:
        masses = umasses
    elif imax == -1:
        masses = umasses[imin - 1::]    
    else:
        masses = umasses[imin - 1: imax + 1]

    masses_str = ','.join(map(str, masses))

    eep = Eep()
    mdict = {'npt_low': eep.nlow,
             'npt_hb': eep.nhb,
             'npt_tr': eep.ntot - eep.nms - eep.nhb,
             'npt_ms': eep.nms,
             'masses_str': masses_str,
             'prefix_str': prefix_str,
             'FNZ': FNZ,
             'zs_str': zs_str,
             'modelIZmax': modelIZmax,
             'modelIZmin': modelIZmin,
             'zsun': zsun}
    fname = 'makemod_%s_%s.txt' % (inputs.tracks_dir.split('/')[-2], p.split('_Z')[0])
    with open(fname, 'w') as out:
        out.write(makemod_fmt() % mdict)

def makemod_fmt():
    return """
const double Zsun = %(zsun).2f;
const double Z[] = {%(zs_str)s};
static const int NZ = sizeof(Z)/sizeof(double);
const char FNZ[NZ][%(FNZ)i] = {%(prefix_str)s};

const double M[] = {%(masses_str)s};
static const int NM = sizeof(M)/sizeof(double);

// limits of age and metallicity coverage
static const int modelIZmin = %(modelIZmin)i;
static const int modelIZmax = %(modelIZmax)i;

// number of values along isochrone (in addition to logTe and Mbol)
static const int NHRD=3;

// range of Mbol and logTeff
static const double MOD_L0 = -8.0;
static const double MOD_LF = 13.0;
static const double MOD_T0 = 3.30;
static const double MOD_TF = 5.00;

static const int ML0 = 9; // number of mass loss steps
//static const int ML0 = 0; // number of mass loss steps
static const double ACC = 3.0; // CMD subsampling

// Number of points along tracks
static const int NPT_LOW = %(npt_low)i; // low-mass tracks points
static const int NPT_MS = %(npt_ms)i; // MS tracks points
static const int NPT_TR = %(npt_tr)i; // transition MS->HB points
static const int NPT_HB = %(npt_hb)i; // HB points

cd ..; make PARSEC; cd PARSEC; ./makemod
"""


if __name__ == '__main__':
    inp_obj = fileio.InputFile(sys.argv[1], default_dict=initialize_inputs())
    loud = False
    if len(sys.argv) > 2:
        loud = True

    if inp_obj.debug:
        import pdb
        pdb.set_trace()

    if inp_obj.prepare_makemod:
        prepare_makemod(inp_obj)

    parsec2match(inp_obj, loud=loud)

