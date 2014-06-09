from __future__ import print_function
from copy import deepcopy
import sys
import os
import matplotlib.pyplot as plt
from ResolvedStellarPops.fileio import fileIO
from ResolvedStellarPops.padova_tracks.match import TracksForMatch
from ResolvedStellarPops.padova_tracks.match import MatchTracks
from ResolvedStellarPops.padova_tracks.eep import critical_point

def parsec2match(input_obj):
    '''
    do an entire set and make the plots
    '''
    tracks_dir = input_obj.tracks_dir
    if input_obj.prefixs == 'all':
        prefixs = [d for d in os.listdir(tracks_dir)
                   if os.path.isdir(os.path.join(tracks_dir, d))]
    elif input_obj.prefixs is not None:
        prefixs = input_obj.prefixs
    else:
        if input_obj.prefix is None:
            print('nothing to do')
            return
        prefixs = [input_obj.prefix]

    # will overwrite prefix instead through the loop
    del input_obj.prefixs
    assert type(prefixs) == list, 'prefixs must be a list'

    for prefix in prefixs:
        print('Current mix: %s' % prefix)
        these_inputs = set_outdirs(input_obj, prefix)
        if these_inputs.ptcri_file is not None:
            these_inputs.ptcri_file = input_obj.ptcri_file
        else:
            if these_inputs.from_p2m is True:
                # this is the equivalent of Sandro's ptcri files, but mine.
                search_term = 'p2m*%s*dat' % these_inputs.prefix
                print('reading ptcri from saved p2m file.')
            else:
                search_term = 'pt*%s*dat' % these_inputs.prefix

        these_inputs.ptcri_file, = fileIO.get_files(these_inputs.ptcrifile_loc,
                                                    search_term)

        tfm = TracksForMatch(these_inputs)
        tfm.ptcri = critical_point.critical_point(these_inputs.ptcri_file)
        these_inputs.flag_dict = tfm.match_interpolation(these_inputs)
        #tfm.save_ptcri(hb=these_inputs.hb)
        mt = MatchTracks(these_inputs)
        mt.check_tracks()
        mt.diag_plots()
        plt.close('all')
    print('DONE')
    return

def set_outdirs(input_obj, prefix):
    '''
    set up the directories for output and plotting
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
