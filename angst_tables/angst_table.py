from __future__ import print_function
import numpy as np
import os
import difflib

from .helpers import deprecated, flatten_dict

#config
TABLE_DIR = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'tables')


__all__ = ['AngstTables']


class AngstTables(object):
    """
    """
    def __init__(self):
        self.table5 = read_angst_tab5()
        self.table4 = read_angst_tab4()

        for i in [1, 2, 3]:
            self.__setattr__('snap_tab%i' % i, read_snap(table='table%i' % i))

        self.targets = np.unique(np.concatenate((self.table4['target'],
                                                 self.table5['target'])))
        self.load_data()

    def get_item(self, target, item, extra_key=None, case=True):
        '''
        The problem with writing basic necessity code before I got the hang
        of python is that I need to write shitty wrappers now...
        '''
        target = self.correct_target(target)

        table_row = self.__getattribute__(target)

        if item in table_row:
            return table_row[item]

        flat_table_row = flatten_dict(table_row)

        if item in flat_table_row:
            return flat_table_row[item]

        if case is True:
            match = [ k for k in flat_table_row if item in k ]
        else:
            match = [ k for k in flat_table_row if item.lower() in k.lower() ]

        if (len(match) > 1) & (extra_key is not None):
            if case is True:
                match = [ k for k in match if extra_key in k ]
            else:
                match = [ k for k in match if extra_key.lower() in k.lower() ]

        if len(match) > 1:
            val = [ flat_table_row[mk] for mk in match ]
        else:
            val = flat_table_row[match]

        return val, match

    def load_data(self):
        '''
        loads table 5 and table 4 with target names as attributes
        and filter-specific data in a dictionary.
        '''
        # angst table 4:
        subdict_keys = ['Nstars', 'exposure_time', '50_completeness_mag']
        replace_keys = {'50_completeness_mag': '50_completeness'}
        break_key = 'filter'

        for row in self.table4:
            target = row['target'].replace('-', '_')
            row_dict = dict(zip(row.dtype.names, row))
            filter = row_dict['filter']
            target_dict = split_dictionary(row_dict, break_key, filter, *subdict_keys, **replace_keys)

            if not hasattr(self, target):
                self.__setattr__('%s' % target, target_dict)
            else:
                self.__dict__[target].update(target_dict)

        # angst table 5
        subdict_keys = ['Nstars', 'Av', 'mean_color', 'mTRGB_raw', 'mTRGB',
                        'mTRGB_err', 'dmod', 'dist_Mpc', 'dist_Mpc_err']
        break_key = 'filters'

        for row in self.table5:
            target = row['target'].replace('-', '_')
            row_dict = dict(zip(row.dtype.names, row))
            filters = row_dict['filters']
            target_dict = split_dictionary(row_dict, break_key, filters, *subdict_keys)

            if not hasattr(self, target):
                self.__setattr__('%s' % target, target_dict)
            else:
                self.__dict__[target].update(target_dict)

    def correct_target(self, target):
        '''
        NOT FINISHED
        '''
        target = target.upper().replace('-', '_')
        if '404' in target:
            target = 'NGC404_DEEP'
        return target


    def get_tab5_trgb_av_dmod(self, target, filters=None):
        '''
        backward compatibility to my old codes.
        it's a bit crap.

        since trgb is F814W, Av is V, the exact field doesn't matter
        for my batch use.

        If the target isn't in table 5, I find the closest match to
        the target string and grab those filters. All that is stored
        locally.
        '''
        target = target.upper().replace('-', '_')
        if filters is not None and 'F160W' in filters:
            return self.get_snap_trgb_av_dmod(target)

        k = [k for k in self.__dict__[target].keys() if ',' in k]
        try:
            datum = self.__dict__[target][k]
        except:
            otarget = target
            target = target.replace('_', '-').split('WIDE')[0]
            target = difflib.get_close_matches(target, self.table5['target'])[0]
            print('%s using %s' % (otarget, target))
            filters = [k for k in
                       self.__dict__[target.replace('-', '_')].keys()
                       if ',' in k][0]
            datum = self.__dict__[target.replace('-', '_')][filters]

        trgb = datum['mTRGB']
        av = datum['Av']
        dmod = datum['dmod']
        return trgb, av, dmod

    @deprecated
    def get_50compmag(self, target, filter):
        '''
        backward compatibility to my old codes.
        input target,filter: get 50% comp.
        '''
        target = target.upper().replace('-', '_').replace('C_', 'C')

        if 'F160W' in filter or 'F110W' in filter:
            return self.get_snap_50compmag(target, filter)
        try:
            datum = self.__dict__[target][filter]
        except KeyError:
            target = difflib.get_close_matches(target, self.table5['target'])[0]
            datum = self.__dict__[target][filter]
        return datum['50_completeness']


    def get_snap_trgb_av_dmod(self, otarget):
        try:
            target = self[snap_tab3['target']]
        except:
            target = difflib.get_close_matches(otarget.upper(), self.snap_tab3['target'])[0]
        if target != otarget:
            print('get_snap_trgb_av_dmod: using {}, not {}'.format(target, otarget))
        ind, = np.nonzero(self.snap_tab3['target'] == target)
        mTRGB, = self.snap_tab3['mTRGB_raw'][ind]
        dmod, = self.snap_tab3['dmod'][ind]
        Av, = self.snap_tab3['Av'][ind]
        return mTRGB, Av, dmod

    def get_snap_50compmag(self, target, filter):
        target = difflib.get_close_matches(target, self.snap_tab2['target'])[0]
        ind, = np.nonzero(self.snap_tab2['target'] == target)
        return self.snap_tab2['50_completeness_%s' % filter][ind][0]


def split_dictionary(rawdict, break_key, subdictname,
                     *subdict_keys, **replace_keys):
    '''
    splits dictionary into two based on a key. The sub dictionary takes on
    values from the main dictionary. Also allows to replace keys in sub
    dictionary.
    INPUT
    rawdict: entire dictionary
    break_key: key of rawdict to make subdict (and then remove)
    subdictname: val of newdict that will be key for subdict
    subdict_keys: *keys of rawdict to put in subdict
    replace_keys: **{old key:new key} keys of new subdict to change
    OUTPUT
    newdict
    '''
    maindict = rawdict.copy()
    tempdict = rawdict.copy()
    [maindict.pop(k) for k in subdict_keys]
    [tempdict.pop(k) for k in rawdict.keys() if not k in subdict_keys]
    try:
        [d.pop(break_key) for d in (maindict, tempdict)]
    except KeyError:
        pass
    subdict = tempdict.copy()

    for kold, knew in replace_keys.items():
        subdict[knew] = tempdict[kold]
        subdict.pop(kold)

    newdict = maindict.copy()
    newdict[subdictname] = subdict
    return newdict


def read_snap(table=None):
    assert table in ['table1', 'table2', 'table3'], \
        'table must be table1, table2 or table3'

    if table == 'table1':
        dtype = [('Galaxy', '|U9'), ('AltNames', '|U18'), ('ra', '|U12'),
                 ('dec', '|U12'), ('diam', '<f8'), ('Bt', '<f8'), ('Av', '<f8'),
                 ('dmod', '<f8'), ('T', '<f8'), ('W50', '<f8'), ('Group', '|U8')]

    if table == 'table2':
        dtype = [('catalog name', '|U8'), ('target', '|U18'),
                 ('ObsDate', '|U21'), ('Nstars', '<f8'), ('Uigma_max ', '<f8'),
                 ('Uigma_min', '<f8'), ('.50_completeness_F110W', '<f8'),
                 ('.50_completeness_F160W', '<f8'), ('opt propid', '|U10'),
                 ('opt filters ', '|U19')]

    if table == 'table3':
        dtype = [('catalog name', '|U10'), ('target', '|U17'), ('dmod', '<f8'),
                 ('Av', '<f8'), ('Nstars', '<f8'), ('mean_color', '<f8'),
                 ('mTRGB_raw', '<f8'), ('mTRGB_F160W', '<f8'),
                 ('mTRGB_F160W_err', '<f8'), ('MTRGB_F160W', '<f8'),
                 ('MTRGB_F160W_err', '<f8')]

    table = os.path.join(TABLE_DIR, 'snap_%s.tex' % table)
    return np.genfromtxt(table, delimiter='&', dtype=dtype, autostrip=1)


def read_angst_tab5():
    dtype = [('catalog name', '|U10'), ('target', '|U23'), ('filters', '|U11'),
             ('Nstars', '<f8'), ('Av', '<f8'), ('mean_color', '<f8'),
             ('MTRGB_F814W', '<f8'), ('mTRGB_raw', '<f8'), ('mTRGB', '<f8'),
             ('mTRGB_err', '<f8'), ('dmod', '<f8'), ('dist_Mpc', '<f8'),
             ('dist_Mpc_err', '<f8')]

    table = os.path.join(TABLE_DIR, 'angst_tab5.tex')
    tab5 = np.genfromtxt(table, delimiter='&', dtype=dtype, autostrip=1)
    return tab5


def read_angst_tab4():
    dtype = [('catalog name', '|U14'), ('propid', '<f8'), ('target', '|U19'),
             ('camera', '|U5'), ('filter', 'U5'), ('exposure time', '<f8'),
             ('Nstars', '<f8'), ('50_completeness_mag', '<f8')]
    table = os.path.join(TABLE_DIR, 'angst_tab4.tex')
    tab4 = np.genfromtxt(table, delimiter='&', dtype=dtype, autostrip=1)
    return tab4


def cleanup_target(target):
    '''
    table 4 and table 5 call galaxies different things.
    table 5 is only to get dmod, av, and trgb (in 814) so it's
    not a problem to use another field in the same galaxy to grab the data
    it will be the same within errors.
    '''
    if 'wide' in target:
        target = target[:-1]
    if 'field' in target:
        target = '-'.join(target.split('-')[:-1])
    if 'c-' in target:
        target = target.replace('c-', 'c')
    if 'c0' in target:
        target = target.replace('c0', 'c')
    if target.split('-')[-1].isdigit():
        target = target[:-2]
    return target
