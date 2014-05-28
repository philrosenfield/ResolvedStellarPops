from __future__ import print_function
from copy import deepcopy
import sys
import os
import matplotlib.pyplot as plt
from ResolvedStellarPops.fileio import fileIO
from ResolvedStellarPops.padova_tracks.match import TracksForMatch
from ResolvedStellarPops.padova_tracks.match import MatchTracks

def do_entire_set(input_obj):
    '''
    do an entire set and make the plots
    '''
    tracks_dir = input_obj.tracks_dir
    if input_obj.prefixs == 'all':
        prefixs = [d for d in os.listdir(tracks_dir)
                   if os.path.isdir(os.path.join(tracks_dir, d))]
    else:
        prefixs = input_obj.prefixs

    del input_obj.prefixs
    assert type(prefixs) == list, 'prefixs must be a list'

    for prefix in prefixs:
        print('\n\n Current mix: %s \n\n' % prefix)
        these_inputs = set_outdirs(input_obj, prefix)
        tfm = TracksForMatch(these_inputs)
        tfm.save_ptcri(hb=these_inputs.hb)
        MatchTracks(these_inputs)
        plt.close('all')


def set_outdirs(input_obj, prefix):
    '''
    set up the directories for output and plotting
    '''
    if not hasattr(input_obj, 'tracks_dir'):
        print('No tracks_dir set, using current location')
        input_obj.tracks_dir = os.getcwd()

    new_inputs = deepcopy(input_obj)
    new_inputs.prefix = prefix
    wkd = os.path.join(input_obj.tracks_dir, new_inputs.prefix)
    if hasattr(input_obj, 'plot_dir') and input_obj.plot_dir == 'default':
        new_inputs.plot_dir = os.path.join(wkd, 'plots')
        fileIO.ensure_dir(new_inputs.plot_dir)

    if hasattr(input_obj, 'outfile_dir') and input_obj.outfile_dir == 'default':
        new_inputs.outfile_dir = os.path.join(wkd, 'match')
        fileIO.ensure_dir(new_inputs.outfile_dir)

    return new_inputs


def initialize_inputs():
    '''
    Load default inputs, the eep lists, and number of equally spaced points
    between eeps. Input file will overwrite these.
    '''
    input_dict =  {'track_search_term': '*F7_*PMS',
                   'hbtrack_search_term':'*F7_*HB',
                   'from_p2m': False,
                   'hb_only': False,
                   'masses': None,
                   'do_interpolation': True,
                   'debug': False,
                   'hb': True}
    return input_dict

if __name__ == '__main__':
    inp_obj = fileIO.InputFile(sys.argv[1], default_dict=initialize_inputs())
    # if prefix not prefixs, set the location of plots if given default.
    if hasattr(inp_obj, 'prefix'):
        inp_obj = set_outdirs(inp_obj, inp_obj.prefix)

    #import pdb
    #pdb.set_trace()

    if hasattr(inp_obj, 'prefixs'):
        do_entire_set(inp_obj)
    else:
        tfm = TracksForMatch(inp_obj)
        tfm.save_ptcri(hb=inp_obj.hb)
        MatchTracks(inp_obj)
