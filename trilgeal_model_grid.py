import TrilegalUtils
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
                 inp_pref='input'):
        
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
        form and content separated!
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

        for age, z in itertools.product(ages[:-1], zs):
            to = age
            tf = age + self.dlogt

            # set up filenames TODO: make prefixes kwargs

            sfh_file = self.filename_fmt(self.sfr_pref,to, tf, z)
            galaxy_input = self.filename_fmt(self.inp_pref, to, tf, z)
            output =  self.filename_fmt(self.out_pref, to, tf, z)

            # write files

            self.write_sfh_file(sfh_file, to, tf, z)
            self.make_galaxy_input(sfh_file, galaxy_input)

            # run trilegal

            TrilegalUtils.run_trilegal(self.cmd_input, galaxy_input, output)
        os.chdir(here)

    def get_grid(self, search_string):
        sub_grid = fileIO.get_file(self.location, search_string)

    def load_grid(self):
        grid = fileIO.get_files(self.location, '%s*dat' % self.out_pref)
        self.grid = grid
        return
    
    def filter_grid(self, younger=None, older=None, metal_rich=None, metal_poor=None):
        sub_grid = self.grid[:]

        if older is not None:
            sub_grid = filter((lambda c: os.path.split(c)[1].split('_')[4] > older), sub_grid)
        if younger is not None:
            sub_grid = filter((lambda c: os.path.split(c)[1].split('_')[5] > younger), sub_grid)
        if metal_rich is not None:
            sub_grid = filter((lambda c: os.path.split(c)[1].split('_')[6] > metal_rich), sub_grid)
        if metal_poor is not None:
            sub_grid = filter((lambda c: os.path.split(c)[1].split('_')[6] < metal_poor), sub_grid)    
        
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

class sf_stitcher(TrilegalUtils.trilegal_sfh, model_grid):
    def __init__(self, sfr_file, galaxy_input=False, indict={}):
        model_grid.__init__(self, **indict)
        TrilegalUtils.trilegal_sfh.__init__(self, sfr_file, galaxy_input=False)

    def join_sfr(sfr_files):
        pass

    def build_sfh(self):
        if max(self.age) > 12.:
            self.lage = np.log10(self.age)
        else:
            self.lage = self.age
            self.age = 10**self.lage
        import pdb
        pdb.set_trace()
        
        assert np.diff(lage) > self.dlogt, 'time steps too small in sfr_file.'
        if not hasattr(self, 'grid'):
            self.load_grid()
        
        sub_grid = [c for c in self.grid if float(os.path.split(c)[1].split('_')[4]) > lage[0]]
        sub_grid = [c for c in sub_grid if float(os.path.split(c)[1].split('_')[5]) < lage[-1]]
        sgals = [rsp.Galaxies.simgalaxy(c, filter1=filter1, filter2=filter2,
                 photsys=self.photsys) for c in cmds]

        infile_dlogt = np.round(np.diff(lage), 2)
        
        sfr_files = [self.filename_fmt(self.out_pref, self.lage[i],
                     self.lage[i+1], self.z[i]) for i in range(len(self.lage) - 1)]
        [fileIO.ensure_file(s) for s in sfr_files]
        #sfr_files = [fileIO.get_files(self.location, sfr_string) for sfr_string in sfr_strings]
        print sfr_files

if __name__ == '__main__':
    input_file = sys.argv[1]
    indict = fileIO.load_input(input_file)
    fileIO.ensure_file(indict['cmd_input'])
    mg = model_grid(**indict)
    import pdb
    pdb.set_trace()
    mg.make_grid()