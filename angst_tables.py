import numpy as np
import os
from BRparams import *
import traceback
import re
import difflib
import logging
logger = logging.getLogger()

class AngstTables(object):
    def __init__(self):
        self.table5 = read_angst_tab5()
        self.table4 = read_angst_tab4()
        self.targets = np.unique(np.concatenate((self.table4['target'],
                                                 self.table5['target'])))
        AngstTables.load_data(self)
        
    def load_data(self):
        '''
        loads table 5 and table 4 with target names as attributes
        and filter-specific data in a dictionary.
        '''
        # angst table 4:
        subdict_keys = ['Nstars','exposure_time','50_completeness_mag']
        replace_keys = {'50_completeness_mag':'50_completeness'}
        
        break_key = 'filter'
        targets = np.unique([t.replace('-','_') for t in self.table4['target']])
        
        for row in self.table4:
            target = row['target'].replace('-','_')
            row_dict = dict(zip(row.dtype.names,row))
            filter = row_dict['filter']
            target_dict = split_dictionary(row_dict,break_key,filter,
                                           *subdict_keys,**replace_keys)
            
            if not hasattr(self,target):
                self.__setattr__('%s'%target,target_dict)
            else:
                self.__dict__[target].update(target_dict)
        
        # angst table 5
        subdict_keys = ['Nstars','Av','mean_color','mTRGB_raw','mTRGB',
                        'mTRGB_err','dmod','dist_Mpc','dist_Mpc_err']
        break_key = 'filters'
        
        for row in self.table5:
            target = row['target'].replace('-','_')
            row_dict = dict(zip(row.dtype.names,row))
            filters = row_dict['filters']
            target_dict = split_dictionary(row_dict,break_key,filters,
                                           *subdict_keys)
    
            if not hasattr(self,target):
                self.__setattr__('%s'%target,target_dict)
            else:
                self.__dict__[target].update(target_dict)
    
    def correct_target(target):
        '''
        NOT FINISHED
        '''
        pass
    
    def get_tab5_trgb_av_dmod(self,target,filters):
        '''
        backward compatibility to my old codes. 
        
        it's a bit crap. 
        
        since trgb is F814W, Av is V, the exact field doesn't matter
        for my batch use. 
        
        If the target isn't in table 5, I find the closest match to
        the target string and grab those filters. All that is stored 
        locally.
        '''
        target = target.upper().replace('-','_')
        try:
            datum = self.__dict__[target][filters]
        except KeyError,err:
            print traceback.print_exc()
            print '%s not found'%target
            target = target.replace('_','-')
            target = difflib.get_close_matches(target, self.table5['target'])[0]
            print 'using %s'%target
            filters = [k for k in self.__dict__[target.replace('-','_')].keys() if ',' in k][0]
            datum = self.__dict__[target.replace('-','_')][filters]
        try:
            trgb = datum['mTRGB']
            av = datum['Av']
            dmod = datum['dmod']
        except KeyError,err:
            print 'fuck.'
        return trgb,av,dmod

    def get_50compmag(self,target,filter):
        '''
        backward compatibility to my old codes. 
        input target,filter: get 50% comp. 
        '''
        target = target.upper().replace('-','_')
        try:
            datum = self.__dict__[target][filter]
        except KeyError,err:
            logger.error(traceback.print_exc())
            logger.error('keys available: {}'.format(self.__dict__[target].keys()))
            return -1,-1,-1
        return datum['50_completeness']

def split_dictionary(rawdict,break_key,subdictname,*subdict_keys,**replace_keys):
    '''
    splits dictionary into two based on a key. The sub dictionary takes on values
    from the main dictionary. Also allows to replace keys in sub dictionary.
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
        [d.pop(break_key) for d in (maindict,tempdict)]
    except KeyError:
        pass    
    subdict = tempdict.copy()
    
    for kold,knew in replace_keys.items():
        subdict[knew] = tempdict[kold]
        subdict.pop(kold)
    
    newdict = maindict.copy()
    newdict[subdictname] = subdict
    return newdict

def read_angst_tab5():
    dtype = [('catalog name', '|S10'), ('target','|S23'),('filters','|S11'),('Nstars','<f8'),('Av','<f8'),('mean_color','<f8'),('MTRGB_F814W','<f8'),('mTRGB_raw','<f8'),('mTRGB','<f8'),('mTRGB_err','<f8'),('dmod','<f8'),('dist_Mpc','<f8'),('dist_Mpc_err','<f8')]
    table = os.path.join(TABLE_DIR,'tab5.tex')
    tab5 = np.genfromtxt(table,delimiter='&',dtype=dtype,autostrip=1)
    return tab5

def read_angst_tab4():
    dtype=[('catalog name', '|S14'), ('propid', '<f8'), ('target', '|S19'), ('camera', '|S5'), ('filter', 'S5'), ('exposure time', '<f8'), ('Nstars', '<f8'), ('50_completeness_mag', '<f8')]
    table = os.path.join(TABLE_DIR,'tab4.tex')
    tab4 = np.genfromtxt(table,delimiter='&',dtype=dtype,autostrip=1)
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
        target = target.replace('c-','c')
    if 'c0' in target:
        target = target.replace('c0','c')
    if target.split('-')[-1].isdigit():
        target = target[:-2]
    return target
    