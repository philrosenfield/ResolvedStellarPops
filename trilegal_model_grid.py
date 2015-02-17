import argparse
import ResolvedStellarPops as rsp
import numpy as np
import itertools
import sys
import os
import logging
from IPython import parallel
import time

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def example_inputfile():
    return """
# input file for ResolvedStellarPops.trilegal_model_grid
# python ~/research/python/ResolvedStellarPops/trilegal_model_grid.py model_grid.inp
# notes: cmd_input should be abs path
#        object_mass should be large enough to well sample the IMF.
#        agb tracks used for Nell are MAR13 (important info if remaking cmd_input)
cmd_input       /home/rosenfield/research/padova_apps/trilegal_1.5/cmd_input_CAF09_M4.dat
photsys         phat
filter          F814W
object_mass     1e7
dlogt           0.01
dlogz           0.002
# if it's > 10.1, it will be called 10.1...
logtrange       9.96, 10.05
logzrange       0.01, 0.032
location        /home/rosenfield/research/stel_evo/NellGrid/tri_grid_etaM6/
sfr_pref        sfr
out_pref        burst_mloss
inp_pref        input
over_write      False
"""

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
                 over_write=False,
                 **kwargs):
        '''
        kwargs are not used here, just left to pass to other objects.
        '''

        assert cmd_input is not None, 'need cmd_input file'
        assert photsys is not None, 'need photsys'
        assert filter is not None, 'need filter for rsp.trilegal galaxy input'

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
        self.zs = kwargs.get('zs')

        if location is None:
            location = os.getcwd()
        else:
            rsp.fileio.ensure_dir(location)
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
        be in rsp.trilegal utils and not needed. sigh.
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
        first_dict = {'filter1': self.filter,
                      'photsys': self.photsys}
        gal_inppars = rsp.fileio.InputParameters(rsp.trilegal.galaxy_input_dict(**first_dict))
        (mag_num, mag_file) = rsp.trilegal.find_photsys_number(self.photsys,
                                                                self.filter)
        default_kw = {'object_sfr_file': sfr_file,
                      'object_mass': self.object_mass,
                      'photsys': self.photsys,
                      'mag_num': mag_num,
                      'file_mag': mag_file}

        kwargs = dict(default_kw.items() + galaxy_inkw.items())
        gal_inppars.add_params(kwargs)
        gal_inppars.write_params(galaxy_input, rsp.trilegal.galaxy_input_fmt())
        return

    def run_parallel(self, dry_run=False, nproc=8, start=30,
                     timeout=45):
        """parallelize make_grid"""
        def setup_parallel():
            """I would love a better way to do this."""
            clients = parallel.Client()
            clients.block = False
            clients[:].use_dill()
            clients[:].execute('import ResolvedStellarPops as rsp')
            clients[:].execute('import numpy as np')
            clients[:].execute('import os')
            clients[:].execute('import logging')
            clients[:]['logger'] = logger
            return clients

        # check for clusters.
        try:
            clients = parallel.Client()
        except IOError:
            logger.debug('Starting ipcluster... waiting {} s for spin up'.format(start))
            os.system('ipcluster start --n={} &'.format(nproc))
            time.sleep(start)

        # find looping parameters. How many sets of calls to the max number of
        # processors
        ncalls = len(self.galaxy_inputs)
        niters = np.ceil(ncalls / float(nproc))
        sets = np.arange(niters * nproc, dtype=int).reshape(niters, nproc)
        logger.info('{} calls {} sets'.format(ncalls, niters))

        # in case it takes more than 45 s to spin up clusters, set up as
        # late as possible
        clients = setup_parallel()
        logger.debug('ready to go!')
        for j, iset in enumerate(sets):
            # don't use not needed procs
            iset = iset[iset < ncalls]
            # parallel call to run
            res = [clients[i].apply(rsp.trilegal.run_trilegal,
                                    self.cmd_input, self.galaxy_inputs[i],
                                    self.outputs[i],)
                   for i in range(len(iset))]
            logger.debug('waiting on set {} of {}'.format(j, niters))
            while False in [r.ready() for r in res]:
                time.sleep(1)
            logger.info('set {} complete'.format(j))

        return

    def make_grid(self, ages=None, zs=None, run_trilegal=True, galaxy_inkw={},
                  over_write=False):
        '''
        go through each age, metallicity step and make the files/filenames for
        trilgal to make a simple (single) stellar population
        '''
        if ages is None:
            ages = np.arange(*self.logtrange, step=self.dlogt)
        if zs is None:
            zs = np.arange(*self.logzrange, step=self.dlogz)
        if type(zs) == float:
            zs = [zs]

        self.sfh_files = []
        self.galaxy_inputs = []
        self.outputs = []
        for age, z in itertools.product(ages, zs):
            to = age
            tf = age + self.dlogt
            obj_mass = galaxy_inkw.get('object_mass', self.object_mass)

            sfh_file = self.filename_fmt(self.sfr_pref, to, tf, z)
            galaxy_input = self.filename_fmt(self.inp_pref, to, tf, z)
            output =  self.filename_fmt(self.out_pref, to, tf, z)

            # write files
            if self.over_write is False and os.path.isfile(output):
                logger.warning('not overwriting %s' % output)
            else:
                self.write_sfh_file(sfh_file, to, tf, z)
                self.make_galaxy_input(sfh_file, galaxy_input,
                                       galaxy_inkw=galaxy_inkw)
                self.sfh_files.append(os.path.join(self.location, sfh_file))
                self.galaxy_inputs.append(os.path.join(self.location, galaxy_input))
                self.outputs.append(os.path.join(self.location, output))
                ## left commented out -- this would be to run non-parallel
                #if run_trilegal is True:
                #    rsp.trilegal.run_trilegal(self.cmd_input, galaxy_input,
                #                              output)

    def get_grid(self, search_string):
        sub_grid = rsp.fileio.get_files(self.location, search_string)

    def load_grid(self, check_empty=False):
        grid = rsp.fileio.get_files(self.location, '%s*dat' % self.out_pref)

        if check_empty is True:
            # this was happening when I tried to cancel a run mid way, and
            # it still wrote files, just empty ones.
            for filename in grid:
                if os.path.isfile(filename):
                    if os.path.getsize(filename) < 1:
                        print 'rm', filename
                else:
                    print filename, 'does not exist'

        self.grid = grid
        return

    def delete_columns_from_files(self, keep_cols='default', del_cols=None,
                                  fmt='%.4f'):
        '''
        the idea here is to save space on the disk, and save space in memory
        when loading many files, so I'm taking away lots of extra filters and
        other mostly useless info. Some useless info is saved because simgalaxy
        uses it, e.g., dmod. In case I want more filters, I'm keeping log l,
        te, g, and mbol.

        another option is to make the files all binary too.

        this will keep only columns on the keep_cols list,
        right now it only works if it's default.

        del_cols not implemented yet, I don't know how general this should be
        living here.
        '''
        import pdb; pdb.set_trace()
        if not hasattr(self, 'grid'):
            self.load_grid()
        
        if 'acs' in self.photsys:
            logger.error('delete_columns_from_files: ',
                         'must add F475W to cols list as default')
            return

        if keep_cols == 'default':
            cols = ['logAge', '[M/H]', 'm_ini', 'logL', 'logTe', 'logg', 'm-M0',
                    'Av', 'm2/m1', 'mbol', 'F555W', 'F606W', 'F814W', 'stage']

            fmt = '%.2f %.2f %.5f %.3f %.3f %.3f %.2f %.3f %.2f %.3f %.3f %.3f %.3f %i'

        for filename in self.grid:
            file_cols = open(filename).readline().replace('#', '').strip().split()
            if len(file_cols) == len(cols):
                continue
            #tab = rsp.fileio.read_table(filename)
            #col_vals = [tab.key_dict[c]
            #            for c in cols if c in tab.key_dict.keys()]
            cols_save = [i for i, c in enumerate(file_cols) if c in cols]
            new_tab = np.genfromtxt(filename, usecols=cols_save)
            #new_tab = np.delete(tab.data, vals, axis=1)
            rsp.fileio.savetxt(filename+'test', new_tab, fmt=fmt,
                               header= '# %s\n' % (' '.join(cols)))


def main(argv):
    parser = argparse.ArgumentParser(description="Make a large number of sfh bursts with TRILEGAL")

    parser.add_argument('-v', '--pdb', action='store_true',
                        help='verbose mode')

    parser.add_argument('name', type=str,
                        help='input file e.g., {}'.format(example_inputfile()))

    args = parser.parse_args(argv)

    # set up logging
    handler = logging.FileHandler('{}.log'.format(args.name))
    if args.pdb:
        handler.setLevel(logging.DEBUG)
    else:
        handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    indict = rsp.fileio.load_input(args.name)
    
    if not 'cmd_input' in indict.keys():
        cmd_input_src = indict['cmd_input_src']
        cmd_input_fmt = indict['cmd_input_fmt']
        cmd_inputs = rsp.fileio.get_files(cmd_input_src, cmd_input_fmt)
    else:
        cmd_inputs = [indict['cmd_input']]

    base = indict['location']
    for cmd_input in cmd_inputs:
        indict['cmd_input'] = cmd_input
        rsp.fileio.ensure_file(indict['cmd_input'])
        # adding a directory in location for the sims.
        location = os.path.split(cmd_input.replace('.dat', '').replace('cmd_input_', ''))[1]
        indict['location'] = os.path.join(base, location)

        mg = model_grid(**indict)
        mg.make_grid(ages=indict.get('ages'), zs=indict.get('zs'),
                     galaxy_inkw={'filter1': indict.get('filter')})
        mg.run_parallel(nproc=indict.get('nproc', 8))
        
        clean_up = indict.get('clean_up', False)
        if clean_up:
            logger.info('now cleaning up files')
            mg.delete_columns_from_files()


if __name__ == '__main__':
    main(sys.argv[1:])