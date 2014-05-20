from __future__ import print_function
from copy import deepcopy
import sys
import os
import matplotlib.pyplot as plt
from fileio import fileIO
from padova_tracks.match import TracksForMatch
from padova_tracks.match import MatchTracks

def do_entire_set(inputs):
    tracks_dir = inputs.tracks_dir
    if inputs.prefixs == 'all':
        prefixs = [d for d in os.listdir(tracks_dir)
                   if os.path.isdir(os.path.join(tracks_dir, d))]
    else:
        prefixs = inputs.prefixs

    del inputs.prefixs
    assert type(prefixs) == list, 'prefixs must be a list'

    for prefix in prefixs:
        print('\n\n Current mix: %s \n\n' % prefix)
        these_inputs = set_outdirs(inputs, prefix)
        tm = TracksForMatch(these_inputs)
        tm.save_ptcri(hb=these_inputs.hb)
        MatchTracks(these_inputs)
        plt.close('all')


def set_outdirs(inputs, prefix):
    if not hasattr(inputs, 'tracks_dir'):
        print('No tracks_dir set, using current location')
        inputs.tracks_dir = os.getcwd()

    new_inputs = deepcopy(inputs)
    new_inputs.prefix = prefix
    wkd = os.path.join(inputs.tracks_dir, new_inputs.prefix)
    if hasattr(inputs, 'plot_dir') and inputs.plot_dir == 'default':
        new_inputs.plot_dir = os.path.join(wkd, 'plots')
        fileIO.ensure_dir(new_inputs.plot_dir)

    if hasattr(inputs, 'outfile_dir') and inputs.outfile_dir == 'default':
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
    inputs = fileIO.InputFile(sys.argv[1], default_dict=initialize_inputs())
    # if prefix not prefixs, set the location of plots if given default.
    if hasattr(inputs, 'prefix'):
        inputs = set_outdirs(inputs, inputs.prefix)

    import pdb
    pdb.set_trace()

    if hasattr(inputs, 'prefixs'):
        do_entire_set(inputs)
    else:
        tm = TracksForMatch(inputs)
        tm.save_ptcri(hb=inputs.hb)
        MatchTracks(inputs)
