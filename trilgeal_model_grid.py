import TrilegalUtils
import Galaxies
import fileIO
import numpy as np
import itertools
import sys
import os

class model_grid(object):
    def __init__(self,
                 cmd_input=None,
                 photsys=None,
                 filter=None,
                 object_mass=1e6,
                 dlogt=0.05,
                 logtrange=(6.6, 13.15),
                 logzrange=(0.0006, 0.01),
                 dlogz=0.0004,
                 location=None,
                 sfr_pref='sfr',
                 out_pref='burst',
                 inp_pref='input',
                 over_write=False):
        
        assert cmd_input is not None, 'need cmd_input file'
        assert photsys is not None, 'need photsys'
        assert filter is not None, 'need filter for trilegal galaxy input'
        self.__mix_model(cmd_input)
        self.photsys = photsys
        self.filter = filter
        self.dlogt = dlogt
        self.dlogz = dlogz
        self.logtrange = logtrange
        self.logzrange = logzrange
        self.object_mass = object_mass
        self.sfr_pref = sfr_pref
        self.out_pref = out_pref
        self.inp_pref = inp_pref
        self.over_write = over_write

        if location is None:
            location = os.getcwd()
        else:
            fileIO.ensure_dir(location)
        self.location = location

    def write_sfh_file(self, filename, to, tf, z):
        '''
        single burst from to to tf with constant z.
        '''
        offset = 0.0001

        age = 10 ** np.array([to, to + offset, tf, tf + offset])
        sfr = np.array([0, 1, 1, 0])

        if type(z) != np.array:
            # it should always be one value, but perhaps some day I'll
            # have a reason to vary it?
            z = np.repeat(z, sfr.size)

        data = np.column_stack((age, sfr, z))

        np.savetxt(filename, data, fmt='%.4g')
        return

    def __mix_model(self, cmd_input):
        '''
        separate mix and model from string. If I knew regex, this could
        be in trilegal utils and not needed. sigh.
        '''
        self.cmd_input = cmd_input
        string = os.path.split(cmd_input)[1].replace('.dat','').split('_')
        self.mix = string[2]
        self.model = '_'.join(string[3:])
        return 

    def filename_fmt(self, pref, to, tf, z):
        '''
        form and content almost separated!
        '''
        filename_fmt = '%s_%s_%s_%.2f_%.2f_%.4f_%s.dat'
        filename = filename_fmt % (pref, self.mix, self.model, to, tf, z,
                                   self.photsys)
        return os.path.join(self.location, filename)

    def make_galaxy_input(self, sfr_file, galaxy_input, galaxy_inkw={}):
        '''
        makes galaxy_input file forcing only sfr_file and object_mass
        to be changed from default. Any other changes passed as galaxy_input_kw
        dictionary. 
        '''
        gal_inppars = fileIO.input_parameters(TrilegalUtils.galaxy_input_dict())
        (mag_num, mag_file) = TrilegalUtils.find_photsys_number(self.photsys, 
                                                                self.filter)
        default_kw = {'object_sfr_file': sfr_file,
                      'object_mass': self.object_mass,
                      'photsys': self.photsys,
                      'mag_num': mag_num,
                      'file_mag': mag_file}

        kwargs = dict(galaxy_inkw.items() + default_kw.items())
        gal_inppars.add_params(kwargs)
        gal_inppars.write_params(galaxy_input, TrilegalUtils.galaxy_input_fmt())
        return

    def make_grid(self):
        '''
        go through each age, metallicity step and make a single age
        cmd
        '''
        here = os.getcwd()
        os.chdir(self.location)
        ages = np.arange(*self.logtrange, step=self.dlogt)
        zs = np.arange(*self.logzrange, step=self.dlogz)

        for age, z in itertools.product(ages, zs):
            to = age
            tf = age + self.dlogt
            print 'now doing %.2f-%.2f, %.4f' % (to, tf, z)
            # set up filenames TODO: make prefixes kwargs

            sfh_file = self.filename_fmt(self.sfr_pref,to, tf, z)
            galaxy_input = self.filename_fmt(self.inp_pref, to, tf, z)
            output =  self.filename_fmt(self.out_pref, to, tf, z)

            # write files
            if self.over_write is False and os.path.isfile(output):
                print 'not overwriting %s' % output
            else:
                self.write_sfh_file(sfh_file, to, tf, z)
                self.make_galaxy_input(sfh_file, galaxy_input)

                # run trilegal

                TrilegalUtils.run_trilegal(self.cmd_input, galaxy_input, output)
        os.chdir(here)

    def get_grid(self, search_string):
        sub_grid = fileIO.get_file(self.location, search_string)

    def load_grid(self):
        grid = fileIO.get_files(self.location, '%s*dat' % self.out_pref)
        '''
        for file in grid:
           if os.path.isfile(file):
               if os.path.getsize(file) < 1:
                   print 'rm',file
        '''
        self.grid = grid
        return

    def split_on_val(self, string, val):
        return float(os.path.split(string)[1].split('_')[val])    

    def key_map(self, key):
        possible_keys = ['pref', 'mix', 'model1', 'model2', 'to', 'tf', 'z', 'photsys']
        return possible_keys.index(key)
        
    def grid_values(self, *keys):
        '''
        vals are strings like 
        '''
        if not hasattr(self, 'grid'):
            self.load_grid()
        vals = [self.key_map(key) for key in keys]
        grid_dict = {}
        for k,v in zip(keys, vals):
            grid_dict[k] = [self.split_on_val(g, v) for g in self.grid]
            self.__setattr__('grid_%ss' % k, np.unique(grid_dict[k]))
            
    def filter_grid(self, younger=None, older=None, metal_rich=None,
                    metal_poor=None, z=None):
        if not hasattr(self, 'grid'):
            self.load_grid()
        sub_grid = self.grid[:]

        if older is not None and younger is not None: 
            assert older < younger

        if z is not None:
            sub_grid = filter((lambda c: self.split_on_val(c, 6) == z), sub_grid)
            if len(sub_grid) == 0:
                print 'z=%.4f not found' % z
        if older is not None:
            s_grid = filter((lambda c: self.split_on_val(c, 4) >= older), sub_grid)
            if len(s_grid) == 0:
                if not hasattr(self, 'grid_tos'):
                    self.grid_values('to')
                    maxage = np.max(self.grid_tos)
                    print 'lage >= %.1f not found assinging %.1f' % (older, maxage)
                    s_grid = filter((lambda c: self.split_on_val(c, 4) == maxage), sub_grid)
                    print s_grid
            sub_grid = s_grid
        
        if younger is not None:
            if len(sub_grid) == 0:
                print 'no combo'
                return
            sub_grid = filter((lambda c: self.split_on_val(c, 5) <= younger), sub_grid)
            if len(sub_grid) == 0:
                print sub_grid
                print 'lage < %.1f not found' % younger

        '''
        if metal_rich is not None:
            sub_grid = filter((lambda c: split_on_val(c, 6) > metal_rich), sub_grid)
        if metal_poor is not None:
            sub_grid = filter((lambda c: split_on_val(c, 6) < metal_poor), sub_grid)    
        '''
        return sub_grid
'''
class test_grid(model_grid):
    def __init__(self, filter1, filter2, kwargs={}):
        model_grid.__init__(self, **kwargs)
        self.load_grid(filter1='F555W', filter2='F814W')
        # doesn't work but could do this:
        cmds = rsp.fileIO.get_files(os.getcwd(), 'burst*dat')
        sgals = [rsp.Galaxies.simgalaxy(c, filter1='F555W', filter2='F814W', photsys='wfpc2') for c in cmds]
        sgal.plot_cmd(sgal.Color, sgal.Mag2)
'''

class match_sfh(object):
    def __init__(self, sfhfile):
        self.data = np.genfromtxt(sfhfile, skip_header=2, skip_footer=2, unpack=True)
        (to, tf, sfr, nstars, logz, dmod) = self.data
        self.agei = to
        self.agef = tf
        self.sfr = sfr
        self.nstars = nstars
        self.logz = logz
        with open(sfhfile, 'r') as f:
            header = f.readline()
        key_val = header.replace('#','').replace(',').strip().split()
        for kv in key_val:
            (k, v) = kv.split('=')
            self.__setattr__(k, v)
  
  
class sf_stitcher(TrilegalUtils.trilegal_sfh, model_grid):
    def __init__(self, sfr_file, galaxy_input=False, indict={}):
        model_grid.__init__(self, **indict)
        TrilegalUtils.trilegal_sfh.__init__(self, sfr_file, galaxy_input=False)
        
    def join_sfr(sfr_files):
        pass


    def build_sfh(self):
        if max(self.age) > 12.:
            self.lage = np.round(np.log10(self.age), 2)
        else:
            self.lage = np.round(self.age, 2)
            self.age = 10**self.lage

        assert np.diff(self.lage[0::2][:2]) > self.dlogt, 'time steps too small in sfr_file.'
        if not hasattr(self, 'grid'):
            self.load_grid()

        sinds, = np.nonzero(self.sfr)
        to = self.lage[sinds][0::2]
        tf = self.lage[sinds][1::2]
        sfr = self.sfr[sinds][0::2]
        z = np.round([(self.z[i]+self.z[i+1])/2. for i in sinds[0][0::2]], 4)
        # there are quite a few repeated values...
        # where to, tf and z are unique the multiplicative factors are arbs.
        lixo, linds = np.unique(to*2.312+tf*32.221+z*123.111, return_index=True)
        # checked, and all the sfrs are the same, it doesn't matter to trilegal
        # because galaxy mass is normed, but will slow down this code if we 
        # include them. To check:
        # rixo, rinds = np.unique(to+tf+z, return_inverse=True)
        to = to[linds]
        tf = tf[linds]
        sfr = sfr[linds]
        z = z[linds]

        sub_grid_files = np.concatenate([self.filter_grid(younger=tf[i],
                                                          older=to[i],
                                                          z=z[i])
                                         for i in range(len(to))])

        sgals = [Galaxies.simgalaxy(sgf, self.filter1, self.filter2,
                                    photsys=self.photsys)
                                    for sgf in sub_grid_files]

        self.grid_ages = np.concatenate([np.unique(sgal.data.get_col('logAge')) for sgal in sgals])
        
    
if __name__ == '__main__':
    input_file = sys.argv[1]
    indict = fileIO.load_input(input_file)
    fileIO.ensure_file(indict['cmd_input'])
    mg = model_grid(**indict)
    import pdb
    pdb.set_trace()
    mg.make_grid()