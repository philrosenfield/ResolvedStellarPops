import os
import socket
from matplotlib import rc
hostname = socket.gethostname()

FIGEXT = '.png'
if hostname.endswith('astro.washington.edu'):
    TRACKS_BASE ='/astro/net/angst2/philrose/tracks/'
    RUNTRILEGAL = '/astro/users/philrose/python/WXTRILEGAL/run_trilegal.py'
    BRFOLDER = '/astro/net/angst2/philrose/BRratio/'
    GALAXY_LOC = '/astro/net/angst/ben/Robert/'
    
else:
    TRACKS_BASE ='/Users/phil/research/Italy/tracks/'
    BRFOLDER = '/Users/phil/research/BRratio/'
    GALAXY_LOC = '/Users/phil/research/BRratio/data/old_method/'
    
    MODELS_LOC = os.path.join(BRFOLDER,'models')
    DATA_LOC = os.path.join(BRFOLDER,'data')
    
    TRILEGAL_RUNS = os.path.join(MODELS_LOC,'TRILEGAL_RUNS/')
    SPREAD_LOC = os.path.join(MODELS_LOC,'spread')
    
    CMD_INPUTDIR = os.path.join(TRILEGAL_RUNS,'CMD_INPUT')
    
    FAKE_LOC = os.path.join(DATA_LOC,'fakes')
    MATCH_PHOT_LOC = os.path.join(DATA_LOC,'match_phot')
    TAGGED_CMD_LOC = os.path.join(DATA_LOC,'cmd_regions','tagged_photometry')
    TABLE_DIR = os.path.join(BRFOLDER,'tables')