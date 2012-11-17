import numpy as np
import os
import glob
import pyfits

class input_file(object):
    '''
    a class to replace too many kwargs from the input file.
    does two things:
    1. sets a default dictionary (see input_defaults) as attributes
    2. unpacks the dictionary from load_input as attributes.
    '''
    def __init__(self, filename, default_dict=None):
        if default_dict is not None:
            self.set_defaults(default_dict)
        self.in_dict = load_input(filename)
        self.unpack_dict()

    def set_defaults(self, in_def):
        self.unpack_dict(udict=in_def)

    def unpack_dict(self, udict=None):
        if udict is None:
            udict = self.in_dict
        [self.__setattr__(k, v) for k, v in udict.items()]


def load_input(filename):
    '''
    reads an input file into a dictionary.
    file must have key first then value(s)
    Will make 'True' into a boolean True
    Will understand if a value is a float, string, or list, etc.
    Ignores all lines that start with #, but not with # on the same line as
    key, value.
    '''
    try:
        literal_eval
    except NameError:
        from ast import literal_eval

    d = {}
    with open(filename) as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue
            if len(line.strip()) == 0:
                continue
            key, val = line.strip().partition(' ')[0::2]
            d[key] = math_utils.is_numeric(val.replace(' ', ''))
    # do we have a list?
    for key in d.keys():
        # float
        if type(d[key]) == float:
            continue
        # list:
        temp = d[key].split(',')
        if len(temp) > 1:
            try:
                d[key] = map(float, temp)
            except:
                d[key] = temp
        # dict:
        elif len(d[key].split(':')) > 1:
            temp1 = d[key].split(':')
            d[key] = {math_utils.is_numeric(temp1[0]): math_utils.is_numeric(temp1[1])}
        else:
            val = temp[0]
            # boolean
            true = val.upper().startswith('TR')
            false = val.upper().startswith('FA')
            if true or false:
                val = literal_eval(val)
            # string
            d[key] = val
    return d

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
