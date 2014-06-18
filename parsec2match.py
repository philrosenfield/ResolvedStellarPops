from __future__ import print_function
from copy import deepcopy
import pprint
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from ResolvedStellarPops.fileio import fileIO
from ResolvedStellarPops.padova_tracks.match import TracksForMatch
from ResolvedStellarPops.padova_tracks.match import MatchTracks
from ResolvedStellarPops.padova_tracks.eep.critical_point import critical_point
from ResolvedStellarPops.padova_tracks.eep.define_eep import DefineEeps

def parsec2match(input_obj):
    '''do an entire set and make the plots'''
    prefixs = set_prefixs(input_obj)

    for prefix in prefixs:
        print('Current mix: %s' % prefix)
        inps = set_outdirs(input_obj, prefix)
    
        inps = load_ptcri(inps)
        
        tfm = TracksForMatch(inps)
        if inps.from_p2m is True:
            # load from parsec2match eep file (define_eeps already ran)
            pass
        else:
            # find the parsec2match eeps for these tracks.
            tfm = define_eeps(tfm, inps)
        
        # do the match interpolation (produce match output files)
        if inps.do_interpolation is True:
            inps.flag_dict = tfm.match_interpolation(inps)

            # check the match interpolation
            mt = MatchTracks(inps)
            mt.check_tracks()
            pprint.pprint(mt.match_info)
            if inps.diag_plots is True:
                td.diag_plots()
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
        # find all dirs in tracks dir
        prefixs = [d for d in os.listdir(tracks_dir)
                   if os.path.isdir(os.path.join(tracks_dir, d))]
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
    if inputs.ptcri_file is not None:
        inputs.ptcri_file = inputs.ptcri_file
    else:
        if inputs.from_p2m is True:
            # eeps already defined
            search_term = 'p2m*%s*dat' % inputs.prefix
            print('reading ptcri from saved p2m file.')
            sandro = False
        else:
            # eeps are to be defined
            search_term = 'pt*%s*dat' % inputs.prefix
            sandro = True

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

    # Whether or not HB is happening
    hbswitch = np.unique([inputs.hb_only, inputs.hb])
    for i in range(len(hbswitch)):
        crit_kw['hb'] = hbswitch[i]
        track_str = 'tracks'
        if hbswitch[i] is True:
            track_str = 'hbtracks'

        tracks = [de.load_critical_points(track, ptcri=inputs.ptcri, **crit_kw)
                  for track in tfm.tracks]

        if inputs.from_p2m is False:
            inputs.ptcri.save_ptcri(tracks, hb=hbswitch[i])

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
                   'hb_only': False,
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
                   'match': False}
    return input_dict

if __name__ == '__main__':
    inp_obj = fileIO.InputFile(sys.argv[1], default_dict=initialize_inputs())

    if inp_obj.debug is True:
        import pdb
        pdb.set_trace()

    parsec2match(inp_obj)
