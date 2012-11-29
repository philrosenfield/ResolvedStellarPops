import ResolvedStellarPops as rsp
import numpy as np
import os
import logging
logger = logging.getLogger()
if logger.name == 'root':
    rsp.fileIO.setup_logging()


class IsotrackSets(object):
    '''
    finds all isochrone sets for use in trilegal.
    mix is by default CAF09, but basically whatever the isochrone sets starts
    with. picker can be set to be another string to look for, ENV or ENV0.5.
    both mix and picker are sent to get_isosets.
    '''
    def __init__(self, locations='default', mix='', picker=''):
        self.mix = mix
        self.picker = picker
        if locations.lower() == 'default':
            self.isochrone_location = self.default_locations('isochrones',
                                                            load=True)

            self.cmd_input_location = self.default_locations('cmd_input',
                                                             load=True)
        
    def get_em_all(self, item, location):
        if item == 'isochrones':
            initial = self.mix
        if item == 'cmd_input':
            initial = item
        search_string = '%s*%s*dat' % (initial, self.picker)
        dir_list = rsp.fileIO.get_files(location, search_string)
        assert len(dir_list) > 0, '%s found nothing' % search_string
        self.__setattr__(item, dir_list)
        return

    def short_name(self, item):
        short_names = [os.path.split(s)[1].replace('.dat','')
                       for s in self.__getattr__(item)]
        
    def sort_em(self):
        iso_names = self.short_name('isochrones')
        cmd_names = self.short_name('cmd_input')
        # not finished... might not need to be...
        
    def default_locations(self, item, load=False):
        '''
        so far just a pointer to isotrack, which is an environmental variable
        for me, but trilegal root is really the only necessary one.
        
        future: pass 'isochrone_location' to default_locations and get the
        abs path back. Would work for any default locations on the list.
        
        this could be input file ready...
        '''
        trilegal = os.environ['TRILEGAL_ROOT']
        if item == 'isochrones':
            location = os.path.join(trilegal, 'isotrack', 'parsec')
        elif item == 'cmd_input':
            from BRparams import CMD_INPUTDIR
            location = CMD_INPUTDIR
        else:
            print '%s not recongized as input.' % item
            return 0
        assert os.path.isdir(location)
        if load is True:
            self.get_em_all(item, location)
        return location