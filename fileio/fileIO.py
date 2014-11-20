from __future__ import print_function
import numpy as np
import os
import glob
import sys
from pprint import pprint
import difflib
from ..utils import is_numeric


__all__ = ['InputFile', 'InputFile2', 'InputParameters', 'Table', 'ensure_dir',
           'ensure_file', 'get_files', 'get_row', 'item_from_row',
           'load_input', 'read_table', 'read_tagged_phot', 'readfile',
           'replace_ext', 'savetxt', 'get_dirs']


class InputParameters(object):
    '''
    need to make a dictionary of all the possible parameters
        (in the ex: rsp.TrilegalUtils.galaxy_input_dict())
    need to make a formatted string with dictionary printing
        (in the ex: rsp.TrilegalUtils.galaxy_input_fmt())

    example
    import ResolvedStellarPops as rsp
    inp = rsp.fileIO.input_parameters(default_dict=rsp.TrilegalUtils.galaxy_input_dict())
    send any replacement params as kwargs.
    inp.write_params('test', rsp.TrilegalUtils.galaxy_input_fmt())
    $ cat test

    use print inp to see what current values are in cmd line.
    '''
    def __init__(self, default_dict=None):
        self.possible_params(default_dict)

    def possible_params(self, default_dict=None):
        '''
        assign key as attribute name and value as attribute value from
        dictionary
        '''
        default_dict = default_dict or {}
        [self.__setattr__(k, v) for k, v in default_dict.items()]

    def update_params(self, new_dict):
        '''only overwrite attributes that already exist from dictionary'''
        [self.__setattr__(k, v) for k, v in new_dict.items() if hasattr(self, k)]

    def add_params(self, new_dict):
        '''add or overwrite attributes from dictionary'''
        [self.__setattr__(k, v) for k, v in new_dict.items()]

    def write_params(self, new_file, formatter=None):
        '''write self.__dict__ to new_file with format from formatter'''
        with open(new_file, 'w') as f:
            if formatter is not None:
                f.write(formatter % self.__dict__)
            else:
                for k in sorted(self.__dict__):
                    f.write('{0: <16} {1}\n'.format(k, str(self.__dict__[k])))

    def __str__(self):
        '''pprint self.__dict__'''
        pprint(self.__dict__)
        return ""


def savetxt(filename, data, fmt='%.4f', header=None, overwrite=False,
            loud=False):
    '''
    np.savetxt wrapper that adds header. Some versions of savetxt
    already allow this...
    '''
    if overwrite is True or not os.path.isfile(filename):
        with open(filename, 'w') as f:
            if header is not None:
                if not header.endswith('\n'):
                    header += '\n'
                f.write(header)
            np.savetxt(f, data, fmt=fmt)
        if loud:
            print('wrote', filename)
    else:
        print('error: %s exists, not overwriting' % filename)
    return

class InputFile(object):
    '''
    a class to replace too many kwargs from the input file.
    does two things:
    1. sets a default dictionary (see input_defaults) as attributes
    2. unpacks the dictionary from load_input as attributes
        (overwrites defaults).
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


class InputFile2(dict):
    '''
    a class to replace too many kwargs from the input file.
    does two things:
    1. sets a default dictionary (see input_defaults) as attributes
    2. unpacks the dictionary from load_input as attributes
        (overwrites defaults).
    '''
    def __init__(self, filename, default_dict=None):

        dict.__init__(self)

        if default_dict is not None:
            self.update(**default_dict)
        self.update(**load_input(filename))


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
            line = line.translate(None, '[]')
            key, val = line.strip().partition(' ')[0::2]
            d[key] = is_numeric(val.replace(' ', ''))
    # do we have a list?
    for key in d.keys():
        # float
        if type(d[key]) == float or type(d[key]) == int:
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
            d[key] = {is_numeric(temp1[0]): is_numeric(temp1[1])}
        else:
            val = temp[0]
            # boolean
            true = val.upper().startswith('TRUE')
            false = val.upper().startswith('FALSE')
            if true or false:
                val = literal_eval(val)
            # string
            d[key] = val
    return d


def readfile(filename, col_key_line=0, comment_char='#', string_column=None,
             string_length=16, only_keys=None):
    '''
    reads a file as a np array, uses the comment char and col_key_line
    to get the name of the columns.
    '''
    if col_key_line == 0:
        with open(filename, 'r') as f:
            line = f.readline()
        col_keys = line.replace(comment_char, '').strip().split()
    else:
        with open(filename, 'r') as f:
            lines = f.readlines()
        col_keys = lines[col_key_line].replace(comment_char, '').strip().split()
    usecols = range(len(col_keys))

    if only_keys is not None:
        usecols = list(np.sort([col_keys.index(i) for i in only_keys]))
        col_keys = list(np.array(col_keys)[usecols])

    dtype = [(c, '<f8') for c in col_keys]
    if string_column is not None:
        dtype[string_column] = (col_keys[string_column], '|S%i' % string_length)
    data = np.genfromtxt(filename, dtype=dtype, invalid_raise=False,
                         usecols=usecols, skip_header=col_key_line + 1)
    return data


def get_row(arr, index_key, index):
    '''
    send a np.array with dtype.names and choose a column item.
    For example:
    $ data.dtype.names
    ('target', 'opt_trgb', 'nopt_trgb', 'nopt_agb', 'ir_trgb',  'nir_trgb',
    'nir_agb')
    # for an array like:
    ('kkh37', 23.54, 2561.0, 147.0, 21.96, 1729.0, 151.0),
    get_row(data, 'target', 'kkh37')
    ('kkh37', 23.54, 2561.0, 147.0, 21.96, 1729.0, 151.0)
    '''
    fixed_index = difflib.get_close_matches(index.lower(), arr[index_key])
    if len(fixed_index) == 0:
        fixed_index = difflib.get_close_matches(index.upper(), arr[index_key])
    fixed_index = fixed_index[0]
    if index.lower() != fixed_index:
        if index.upper() != fixed_index:
            print('using %s instead of %s' % (fixed_index, index))
    item_key, = np.nonzero(arr[index_key] == fixed_index)
    return arr[item_key]


def item_from_row(arr, index_key, index, column_name):
    '''
    send a np.array with dtype.names and choose a column item.
    For example:
    $ data.dtype.names
    ('target', 'opt_trgb', 'nopt_trgb', 'nopt_agb', 'ir_trgb',  'nir_trgb',
    'nir_agb')
    # for an array like:
    ('kkh37', 23.54, 2561.0, 147.0, 21.96, 1729.0, 151.0),
    $ item_from_row(data, 'target', 'kkh37', 'opt_trgb')
    23.54
    '''
    row = get_row(arr, index_key, index)
    return row[column_name][0]


def replace_ext(filename, ext):
    '''
    input
    filename string with .ext
    new_ext replace ext with new ext
    eg:
    $ replace_ext('data.02.SSS.v4.dat', '.log')
    data.02.SSS.v4.log
    '''
    return split_file_extention(filename)[0] + ext


def split_file_extention(filename):
    '''
    split the filename from its extension
    '''
    return '.'.join(filename.split('.')[:-1]), filename.split('.')[-1]

def read_tagged_phot(tagged_file):
    '''
    reads an ascii photometry file that has been tagged by stage.
    '''
    cols = ['ra', 'dec', 'mag1', 'mag2', 'mag1err', 'mag2err', 'stage']
    fits = np.genfromtxt(tagged_file, names=cols)
    return fits


def ensure_file(f, mad=True):
    '''
    input
    f (string): if f is not a file will print "no file"
    optional
    mad (bool)[True]: if mad is True, will exit program.
    '''
    test = os.path.isfile(f)
    if test is False:
        print('there is no file', f)
        if mad:
            sys.exit()
    return test


def ensure_dir(f):
    '''
    will make all dirs necessary for input to be an existing directory.
    if input does not end with '/' it will add it, and then make a directory.
    '''
    if not f.endswith('/'):
        f += '/'

    d = os.path.dirname(f)
    if not os.path.isdir(d):
        os.makedirs(d)
        print('made dirs:', d)


def read_table(filename, comment_char='#', col_key_line=0):
    '''
    with open(filename) as f:
        lines = f.readlines()

    col_keys = lines[0].strip().replace(comment_char, '').split()

    comments = [l for l in lines[0:] if l.startswith(comment_char)]
    if len(comments) > 0:
        more_keys = [i for i in comments if 'information' in i]
        if len(more_keys) == 1:
            new_cols = more_keys[0].split(':')[1].strip().split()
            col_keys = list(np.concatenate((col_keys, new_cols)))

    Ncols = len(col_keys)
    Nrows = len([l for l in lines if not l.startswith(comment_char)])

    data = np.ndarray(shape=(Nrows, Ncols), dtype=float)

    i = 0
    for line in lines:
        if line.startswith(comment_char):
            continue
        data[i] = line.strip().split()
        i += 1

    tab = Table(data, col_keys, filename)
    '''
    data = readfile(filename,  col_key_line=col_key_line,
                    comment_char=comment_char)
    data = data.view(np.recarray)
    tab = Table(data, list(data.dtype.names), filename)
    return tab


class Table(object):
    '''
    use with read_table(filename)
    self.data_array
    self.key_dict
    self.name
    '''
    def __init__(self, data_array, col_keys, name):
        self.data = data_array
        self.key_dict = dict(zip(col_keys, range(len(col_keys))))
        self.base, self.name = os.path.split(name)

    def get_row(self, i):
        return self.data[i]

    def get_col(self, key):
        return self.data[key]

def get_dirs(src, criteria=None):
    """
    return a list of directories in src, optional simple cut by criteria

    Parameters
    ----------
    src : str
        abs path of directory to search in
    criteria : str
        simple if criteria in d to select within directories in src

    Returns
    -------
    dirs : abs path of directories found
    """
    dirs = [os.path.join(src, l) for l in os.listdir(src) if os.path.join(src, l)]
    if criteria is not None:
        dirs = [d for d in dirs if criteria in d]
    return dirs

def get_files(src, search_string):
    '''
    returns a list of files, similar to ls src/search_string
    '''
    if not src.endswith('/'):
        src += '/'
    try:
        files = glob.glob1(src, search_string)
    except IndexError:
        print('Can''t find %s in %s' % (search_string, src))
        sys.exit(2)
    files = [os.path.join(src, f)
             for f in files if ensure_file(os.path.join(src, f), mad=False)]
    return files
