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
import pprint
import sys

from ResolvedStellarPops.fileio import fileIO
from ResolvedStellarPops.padova_tracks.eep.define_eep import DefineEeps
from ResolvedStellarPops.padova_tracks.eep.critical_point import critical_point
from ResolvedStellarPops.padova_tracks.match import TracksForMatch
from ResolvedStellarPops.padova_tracks.match import CheckMatchTracks

def parsec2match(input_obj):
    '''do an entire set and make the plots'''
    prefixs = set_prefixs(input_obj)

    for prefix in prefixs:
        print('Current mix: %s' % prefix)
        inps = set_outdirs(input_obj, prefix)
    
        inps = load_ptcri(inps)
        
        tfm = TracksForMatch(inps)

        if not inps.from_p2m:
            # find the parsec2match eeps for these tracks.
            tfm = define_eeps(tfm, inps)
        
            if inps.diag_plot:
                # make diagnostic plots using new ptcri file
                inps.ptcri_file = None
                inps.from_p2m = True
                inps = load_ptcri(inps)
                pat_kw = {'ptcri': inps.ptcri, 'extra': 'parsec'}
                xcols = ['LOG_TE', 'AGE']
                if inps.hb:
                    tfm.diag_plots(xcols=xcols, hb=inps.hb, pat_kw=pat_kw)
                else:
                    #tfm.diag_plots(xcols=xcols, pat_kw=pat_kw)
                    tfm.diag_plots(xcols=xcols, pat_kw=pat_kw,
                                   mass_split=None)

        # do the match interpolation (produce match output files)
        if inps.do_interpolation:
            if not hasattr(inps, 'ptcri'):
                inps.ptcri_file = None
                inps.from_p2m = True
                inps = load_ptcri(inps)
            inps.flag_dict = tfm.match_interpolation(inps)

            # check the match interpolation
            cmt = CheckMatchTracks(inps)
            if inps.diag_plot:
                pat_kw = {'extra': 'match', 'plot_dir': inps.outfile_dir}
                #cmt.diag_plots(pat_kw=pat_kw, hb=inps.hb)
                cmt.diag_plots(pat_kw=pat_kw, hb=inps.hb, mass_split=None)

    print('DONE')
    return


def set_prefixs(inputs):
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
            print('nothing to do')
            sys.exit(2)
        # just do one
        prefixs = [inputs.prefix]

    del inputs.prefixs
    assert type(prefixs) == list, 'prefixs must be a list'
    return prefixs


def load_ptcri(inputs):
    '''load the ptcri file, either sandro's or mine'''
    # find the ptcri file
    if inputs.from_p2m:
        sandro = False
        search_term = 'p2m_p*'
        if inputs.hb:
            search_term = 'p2m_hb*'         
        print('reading ptcri from saved p2m file.')
    else:
        sandro = True
        search_term = 'pt*'

    search_term += '%s*dat' % inputs.prefix
    if inputs.ptcri_file is not None:
        inputs.ptcri_file = inputs.ptcri_file
    else:
        inputs.ptcri_file, = fileIO.get_files(inputs.ptcrifile_loc, search_term)
        
    inputs.ptcri = critical_point(inputs.ptcri_file, sandro=sandro)
    return inputs


def define_eeps(tfm, inputs):
    '''add the ptcris to the tracks'''
    # assign eeps track.iptcri and track.sptcri
    de = DefineEeps()
    crit_kw = {'plot_dir': inputs.plot_dir,
               'diag_plot': inputs.diag_plot,
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
    info_file = os.path.join(inputs.tracks_dir, inputs.prefix,
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
    wkd = os.path.join(input_obj.tracks_dir, new_inputs.prefix)

    if input_obj.plot_dir == 'default':
        new_inputs.plot_dir = os.path.join(wkd, 'plots')
        fileIO.ensure_dir(new_inputs.plot_dir)

    if input_obj.outfile_dir == 'default':
        new_inputs.outfile_dir = os.path.join(wkd, 'match')
        fileIO.ensure_dir(new_inputs.outfile_dir)

    return new_inputs


def initialize_inputs():
    '''
    Load default inputs, the eep lists, and number of equally spaced points
    between eeps. Input file will overwrite these. These should be all possible
    input options.
    '''
    input_dict =  {'track_search_term': '*F7_*PMS',
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
                   'agb': False}
    return input_dict

if __name__ == '__main__':
    inp_obj = fileIO.InputFile(sys.argv[1], default_dict=initialize_inputs())

    if inp_obj.debug:
        import pdb
        pdb.set_trace()

    parsec2match(inp_obj)
