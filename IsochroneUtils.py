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

    def plot_isochrone(self, col1, col2, ax=None, plt_kw={}, mag_covert_kw={},
                       photsys=None, clean=True, inds=None, reverse_x=False,
                       reverse_y=False):
        import matplotlib.pyplot as plt
        if ax is None:
            fig = plt.figure()
            ax = plt.axes()
        
        y = self.get_col(col2)    
        if len(mag_covert_kw) != 0:
            import astronomy_utils
            y = astronomy_utils.Mag2mag(y, col2, photsys, **mag_covert_kw)

        if '-' in col1:
            col1a, col1b = col1.split('-')
            x1 = self.get_col(col1a)
            x2 = self.get_col(col1b)
            if len(mag_covert_kw) != 0:
                x1 = astronomy_utils.Mag2mag(x1, col1a, photsys, **mag_covert_kw)
                x2 = astronomy_utils.Mag2mag(x2, col1b, photsys, **mag_covert_kw)
            x = x1 - x2
        else:
            x = self.get_col(col1)
        
        if clean is True:
            rollit = len(x) - np.argmax(x)
            x = np.roll(x, rollit)
            y = np.roll(y, rollit)
            x = x[1:]
            y = y[1:]

        if inds is None:
            ax.plot(x, y, **plt_kw)
        else:
            ax.plot(x[inds], y[inds], **plt_kw)

        if reverse_x is True:
            ax.set_xlim(ax.get_xlim()[::-1])

        if reverse_y is True:
            ax.set_ylim(ax.get_ylim()[::-1])

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
