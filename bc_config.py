import os
try:
    BCDIR = os.environ['BCDIR']
except KeyError:
    BCDIR = None