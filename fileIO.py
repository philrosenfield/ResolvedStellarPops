import numpy as np
import os
import glob
import pyfits


def replace_ext(filename, ext):
    return '.'.join(filename.split('.')[:-1]) + ext


def read_fits(fits_table):
    return pyfits.getdata(fits_table)


def read_tagged_phot(tagged_file):
    '''
    reads an ascii photometry file that has been tagged by stage.
    '''
    cols = ['ra', 'dec', 'mag1', 'mag2', 'mag1err', 'mag2err', 'stage']
    fits = np.genfromtxt(tagged_file, names=cols)
    return fits


def ensure_file(f, mad=True):
    if not os.path.isfile(f):
        print 'there is no file', f
        if mad:
            sys.exit()


def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.isdir(d):
        os.makedirs(d)
        print 'made dirs:', d


def read_table(filename, comment_char='#', loud=False):
    if loud:
        import time
        start = time.time()

        def elapsed():
            return time.time() - start

    f = open(filename, 'r')
    lines = f.readlines()
    f.close()

    if loud:
        print '%.3fs: lines read' % (elapsed())

    col_keys = lines[0].strip().replace(comment_char, '').split()

    Ncols = len(col_keys)
    Nrows = len([l for l in lines if not l.startswith(comment_char)])

    if loud:
        print '%.3fs: Nrows %i' % (elapsed(), Nrows)
    data = np.ndarray(shape=(Nrows, Ncols), dtype=float)

    i = 0
    for line in lines:
        if line.startswith(comment_char):
            continue
        data[i] = line.strip().split()
        i += 1

    if loud:
        print '%.3fs: data filled' % (elapsed())

    tab = Table(data, col_keys, filename)
    return tab


class Table(object):
    '''
    use with read_table(filename)
    self.data_array
    self.key_dict
    self.name
    '''
    def __init__(self, data_array, col_keys, name):
        self.data_array = data_array
        self.key_dict = dict(zip(col_keys, range(len(col_keys))))
        self.base, self.name = os.path.split(name)

    def get_row(self, i):
        return self.data_array[i, :]

    def get_col(self, key):
        return self.data_array[:, self.key_dict[key]]


def get_files(src, search_string):
    '''
    returns a list of files, similar to ls src/search_string
    '''
    if not src.endswith('/'):
        src += '/'
    try:
        files = glob.glob1(src, search_string)
    except IndexError:
        print 'Can''t find %s in %s' % (search_string, src)
        sys.exit(2)
    files = [os.path.join(src, f) for f in files]
    [ensure_file(f) for f in files] 
    return files
