import numpy as np
import sys
import os
import math_utils


class Isochrone(object):
    def __init__(self, Z, age, data_array, col_keys, Y):
        self.Z = Z
        self.age = age
        self.data_array = data_array
        self.key_dict = dict(zip(col_keys, range(len(col_keys))))
        self.Y = Y

    def get_row(self, i):
        return self.data_array[i, :]

    def get_col(self, key):
        return self.data_array[:, self.key_dict[key]]

    def plot_isochrone(self, col1, col2, ax=None, fig=None, plt_kw={},
                       mag_covert_kw={}, photsys=None, clean=True, inds=None, 
                       reverse_x=False, reverse_y=False, pms=False, xlim=None,
                       ylim=None, xdata=None, ydata=None):

        import matplotlib.pyplot as plt
        if ax is None:
            fig = plt.figure()
            ax = plt.axes()
        
        if ydata is not None:
            y = ydata
        else:
            y = self.get_col(col2)    
            if len(mag_covert_kw) != 0:
                import astronomy_utils
                y = astronomy_utils.Mag2mag(y, col2, photsys, **mag_covert_kw)
            ax.set_ylabel('$%s$' % col2, fontsize=20)
        if xdata is not None:
            x = xdata
        else:
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

            ax.set_xlabel('$%s$' % col1, fontsize=20)

        if pms is False and hasattr(self, 'stage'):
            nopms, = np.nonzero(self.get_col('stage') != 0)
        else:
            nopms = np.arange(len(y) - 1)
        
        if inds is not None:
            inds = list(set(inds) & set(nopms))
        else:
            inds = nopms

        x = x[inds]
        y = y[inds]

        if clean is True:
            isep = np.argmax(np.diff(x, 2))
            pl,  = ax.plot(x[:isep-1], y[:isep-1], **plt_kw)
            plt_kw['color'] = pl.get_color()
            plt_kw['label'] = ''
            pl,  = ax.plot(x[isep+1:], y[isep+1:], **plt_kw)

        if reverse_x is True:
            ax.set_xlim(ax.get_xlim()[::-1])

        if reverse_y is True:
            ax.set_ylim(ax.get_ylim()[::-1])
        if xlim is not None:
            ax.set_xlim(xlim)

        if ylim is not None:
            ax.set_ylim(ylim)

        ax.tick_params(labelsize=16)

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
    Ys = []
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
            Ys.append(float(line[7]))
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
                                 col_keys, Ys[i])
    return IsoDict
