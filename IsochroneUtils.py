import numpy as np
import sys
import os
import math_utils


class Isochrone(object):
    def __init__(self, metallicity, age, data_array, col_keys):
        self.metallicity = metallicity
        self.age = age
        self.data_array = data_array
        self.key_dict = dict(zip(col_keys, range(len(col_keys))))

    def get_row(self, i):
        return self.data_array[i, :]

    def get_col(self, key):
        return self.data_array[:, self.key_dict[key]]

    def plot_isochrone(self, col1, col2, ax=None, plt_kw={}):
        import matplotlib.pyplot as plt
        if ax is None:
            fig = plt.figure()
            ax = plt.axes()
        
        x = self.get_col(col1)
        if '-' in col2:
            # not tested...
            col2a, col2b = col2.split('-')
            y = self.get_col(col2a) - self.get_col(col2b)
        else:
            y = self.get_col(col2)    
        ax.plot(x, y, **plt_kw)

        return ax

def create_key(metallicity, age):
    '''
    create a unique dictionary key for each isochrone
    '''
    return 'ISO_%.2g_%.2g' % (metallicity, age)


def get_all_isochrones(filename):
    #first go through the file and find how many total rows and columns
    #of data you have, how many isochrones you have, and the starting
    #index & metadata of each isochrone
    #  pretend these are the values you get
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()

    start_indices = []
    metallicities = []
    ages = []
    Nrows = 0
    N_isochrones = 0
    for i in range(len(lines)):
        # The start of the column head # log age/yr
        tmp = lines[i].replace(' ', '')
        if tmp.startswith('#Z') or tmp.startswith('#log'):
            start_indices.append(i + 1)
        if not lines[i].startswith('#'):
            Nrows += 1
        if lines[i].startswith('#\tI'):
            N_isochrones += 1
            line = lines[i]
            line = line.split()
            metallicities.append(float(line[4]))
            ages.append(float(line[-2]))

    colhead = lines[start_indices[0] - 1].strip()
    colhead = colhead.replace('#', '')
    colhead = colhead.replace('/', '')
    col_keys = colhead.split()
    col_keys[0] = 'LogAge'

    Ncols = len(col_keys)

    # now go back through the file and read in all the rows and columns of data
    data = np.ndarray(shape=(Nrows, Ncols), dtype=float)
    row = 0
    for i in range(len(lines)):
        #print i
        if lines[i].startswith('#'):
            continue
        if not len(lines[i].split()) == Ncols:
            continue
        items = [math_utils.is_numeric(m) for m in lines[i].split()]
        if len(lines[i].split()) != Ncols:
            items.append('')
        try:
            data[row] = items
        except ValueError:
            err = sys.exc_info()[1]
            tb = sys.exc_info()[-1]
            logger.error("%s %s" % (tb.tb_frame.f_code.co_filename, err))
        row += 1

    IsoDict = {}

    #this will help with slicing: see below
    start_indices = np.concatenate((start_indices, [-1]))

    #create your dictionary of isochrone objects
    for i in range(N_isochrones):
        key = create_key(metallicities[i], ages[i])
        IsoDict[key] = Isochrone(metallicities[i],
                                 ages[i],
                                 data[start_indices[i]: start_indices[i + 1]],
                                 col_keys)
    return IsoDict
