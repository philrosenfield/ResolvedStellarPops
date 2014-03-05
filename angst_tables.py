import numpy as np
import os
import traceback
import difflib

TABLE_DIR = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'tables')

class AngstTables(object):
    def __init__(self):
        self.table5 = read_angst_tab5()
        self.table4 = read_angst_tab4()
        [self.__setattr__('snap_tab%i' % i,
                          read_snap(table='table%i' % i)) for i in [1, 2, 3]]
        self.targets = np.unique(np.concatenate((self.table4['target'],
                                                 self.table5['target'])))
        self.load_data()

    def get_item(self, target, item, extra_key=None):
        '''
        The problem with writing basic necessity code before I got the hang
        of python is that I need to write shitty wrappers now...
        '''
        target = self.correct_target(target)
        table_row = self.__getattribute__(target)
        if not item in table_row.keys():
            keys, vals = zip(*[(k, [v for l, v in tvals.items() if item == l])
                for k, tvals in table_row.items() if type(tvals) is dict])
            not_empty, = np.nonzero(vals)
            keys = np.array(keys)[not_empty]
            vals = np.concatenate(vals)
            if len(vals) > 1 and extra_key is None:
                print '%s is ambiguous, please provide extra_key.' % item
                print keys
                return vals
            elif len(vals) > 1 and extra_key is not None:
                val = vals[keys==extra_key][0]
            else:
                val = vals[0]
        else:
            val = table_row[item]
        return val

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
            target_dict = split_dictionary(row_dict, break_key, filter,
                                           *subdict_keys, **replace_keys)

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
            target_dict = split_dictionary(row_dict, break_key, filters,
                                           *subdict_keys)

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

    def get_tab5_trgb_av_dmod(self, target, filters):
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
        if 'F160W' in filters:
            return self.get_snap_trgb_av_dmod(target)
        try:
            datum = self.__dict__[target][filters]
        except KeyError:
            #print traceback.print_exc()
            #print '%s not found' % target
            otarget = target
            target = target.replace('_', '-').split('WIDE')[0]
            target = difflib.get_close_matches(target,
                                               self.table5['target'])[0]
            print '%s using %s' % (otarget, target)
            filters = [k for k in
                       self.__dict__[target.replace('-', '_')].keys()
                       if ',' in k][0]
            datum = self.__dict__[target.replace('-', '_')][filters]
        try:
            trgb = datum['mTRGB']
            av = datum['Av']
            dmod = datum['dmod']
        except KeyError:
            print traceback.print_exc()
        return trgb, av, dmod

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
            print traceback.print_exc()
            print '%s not found' % target
            #target = target.replace('_', '-')
            target = difflib.get_close_matches(target,
                                               self.table5['target'])[0]
            datum = self.__dict__[target][filter]
        return datum['50_completeness']

    def get_snap_trgb_av_dmod(self, target):
        target = difflib.get_close_matches(target, self.snap_tab3['target'])[0]
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
        dtype = [('Galaxy', '|S9'), ('AltNames', '|S18'), ('ra', '|S12'),
                 ('dec', '|S12'), ('diam', '<f8'), ('Bt', '<f8'), ('Av', '<f8'),
                 ('dmod', '<f8'), ('T', '<f8'), ('W50', '<f8'), ('Group', '|S8')]

    if table == 'table2':
        dtype = [('catalog name', '|S8'), ('target', '|S18'),
                 ('ObsDate', '|S21'), ('Nstars', '<f8'), ('Sigma_max ', '<f8'),
                 ('Sigma_min', '<f8'), ('.50_completeness_F110W', '<f8'),
                 ('.50_completeness_F160W', '<f8'), ('opt propid', '|S10'),
                 ('opt filters ', '|S19')]

    if table == 'table3':
        dtype = [('catalog name', '|S10'), ('target', '|S17'), ('dmod', '<f8'),
                 ('Av', '<f8'), ('Nstars', '<f8'), ('mean_color', '<f8'),
                 ('mTRGB_raw', '<f8'), ('mTRGB_F160W', '<f8'),
                 ('mTRGB_F160W_err', '<f8'), ('MTRGB_F160W', '<f8'),
                 ('MTRGB_F160W_err', '<f8')]

    table = os.path.join(TABLE_DIR, 'snap_%s.tex' % table)
    return np.genfromtxt(table, delimiter='&', dtype=dtype, autostrip=1)


def read_angst_tab5():
    dtype = [('catalog name', '|S10'), ('target', '|S23'), ('filters', '|S11'),
             ('Nstars', '<f8'), ('Av', '<f8'), ('mean_color', '<f8'),
             ('MTRGB_F814W', '<f8'), ('mTRGB_raw', '<f8'), ('mTRGB', '<f8'),
             ('mTRGB_err', '<f8'), ('dmod', '<f8'), ('dist_Mpc', '<f8'),
             ('dist_Mpc_err', '<f8')]

    table = os.path.join(TABLE_DIR, 'angst_tab5.tex')
    tab5 = np.genfromtxt(table, delimiter='&', dtype=dtype, autostrip=1)
    return tab5


def read_angst_tab4():
    dtype = [('catalog name', '|S14'), ('propid', '<f8'), ('target', '|S19'),
             ('camera', '|S5'), ('filter', 'S5'), ('exposure time', '<f8'),
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
