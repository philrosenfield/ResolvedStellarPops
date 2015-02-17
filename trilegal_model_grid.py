import argparse
import ResolvedStellarPops as rsp
import numpy as np
import itertools
import sys
import os
import logging
from IPython import parallel

logger = logging.getLogger()

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
        go through each age, metallicity step and make a single age
        cmd
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
            logger.info('now doing %.2f-%.2f, %.4f %g' % (to, tf, z, obj_mass))
            # set up filenames TODO: make prefixes kwargs

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
                #if run_trilegal is True:
                #    if os.path.isfile(output) and over_write is False:
                #        print 'not over writting %s' % output
                #    else:
                #        rsp.trilegal.run_trilegal(self.cmd_input,
                #                                  galaxy_input, output)

    def get_grid(self, search_string):
        sub_grid = rsp.fileio.get_files(self.location, search_string)

    def load_grid(self, check_empty=False):
        grid = rsp.fileio.get_files(self.location, '%s*dat' % self.out_pref)

        if check_empty is True:
            # this was happening when I tried to cancel a run mid way, and
            # it still wrote files, just empty ones.
            for file in grid:
                if os.path.isfile(file):
                    if os.path.getsize(file) < 1:
                        print 'rm', file
                else:
                    print file, 'does not exist'

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
        if not hasattr(self, 'grid'):
            self.load_grid()
        if 'acs' in self.photsys:
            print 'delete_columns_from_files: must add F475W to cols list as default'
            return

        if keep_cols == 'default':
            cols = ['logAge', '[M/H]', 'm_ini', 'logL', 'logTe', 'logg', 'm-M0',
                    'Av', 'm2/m1', 'mbol', 'F555W', 'F606W', 'F814W', 'stage']
            fmt = '%.2f %.2f %.5f %.3f %.3f %.3f %.2f %.3f %.2f %.3f %.3f %.3f %.3f %i'

        for file in self.grid:
            tab = rsp.fileio.read_table(file)
            if len(tab.key_dict.keys()) == len(cols):
                continue
            col_vals = [tab.key_dict[c] for c in cols if c in tab.key_dict.keys()]
            vals = [c for c in range(len(tab.key_dict)) if not c in col_vals]
            new_tab = np.delete(tab.data_array, vals, axis=1)
            rsp.fileio.savetxt(file, new_tab, fmt=fmt, header= '# %s\n' % (' '.join(cols)))

    def split_on_val(self, string, val):
        return float(os.path.split(string)[1].split('_')[val])

    def split_on_key(self, string, key):
        val = self.key_map(key)
        return self.split_on_val(string, val)

    def key_map(self, key):
        possible_keys = ['pref', 'mix', 'model1', 'model2', 'to', 'tf', 'z', 'photsys']
        return possible_keys.index(key)

    def grid_values(self, *keys):
        '''
        vals are strings like to, tf etc. This probably won't need to be used.
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
            sub_grid = filter((lambda c: self.split_on_val(c, 5) <= younger),
                              sub_grid)
            if len(sub_grid) == 0:
                print sub_grid
                print 'lage < %.1f not found' % younger

        return sub_grid


class sf_stitcher(rsp.trilegal.trilegal_sfh, model_grid):
    '''
    inherits from rsp.trilegal sfh and model_grid.
    '''
    def __init__(self, sfr_file, filter1, filter2, galaxy_input=False,
                 indict={}):
        model_grid.__init__(self, **indict)
        rsp.trilegal.trilegal_sfh.__init__(self, sfr_file,
                                            galaxy_input=galaxy_input)
        # put back into msun/year (see MatchUtils.process_sfh or something)
        self.sfr = self.sfr * 1e-3
        self.sfr_file = sfr_file
        self.filter1 = filter1
        self.filter2 = filter2
        self.load_grid()

    def build_model_cmd(self, match_bg, sfr_arr=None, tol=0.25, min_stars=10,
                        over_write=False, spread_old_ages='', age_adjust_limit=13.):
        '''
        1. randomly select x number of stars from cmd in each bin
        2. write to a match bg file

        sf_frac = sfr_match [Msun/yr] / grid_sfr [Msun/yr] = the fraction of
        mass needed to extract from grid cmd.
        sf_frac * tot_mass = the fraction of total mass needed from the sim
        galaxy as a first guess, pull out ave mass * fraction number of stars.

        if sfr_arr is none, will use the sfr_file's sfr i.e, initialization.
        '''
        try:
            random
        except NameError:
            import random

        # hack, only the youngest sf should be adjusted by emcee.
        (sfr_inds, ) = np.nonzero(self.match_sfr[0] <= age_adjust_limit)
        sfr_adj = self.sfr_arr[sfr_inds]
        # add the inds to adjust the sfr to the sf stitcher.
        self.sfr_inds = sfr_inds

        if not hasattr(self, 'sgals'):
            # was this called before build_sfh?
            logger.info('first running build sfh')
            self.build_sfh()

        if sfr_arr is None:
            sfr_arr = self.sfr_arr
        else:
            # over write sfr_arr with the new sfr_arr, the original sfr_arr
            # will still be self.match_sfr[2] (with the rest of the sfh_file)
            self.sfr_arr = sfr_arr

        split_on_age = False
        these_gals = self.sgals.rsp.galaxies

        if hasattr(self, 'sfr_inds'):
            if not os.path.isfile(spread_old_ages) or over_write is True:
                split_on_age = True
                logging.info('writing old age bins to file %s' % spread_old_ages)
                obg = open(spread_old_ages, 'w')
            else:
                logging.info('using a subset galaxy bins')
                these_gals = self.sgals.rsp.galaxies[self.sfr_inds]
                assert len(these_gals) == len(sfr_arr), \
                    'sfr bins and grid are different lengths'

        mbg = open(match_bg, 'w')
        #mbg.write('# %s %s\n' % (self.filter1, self.filter2))
        tofile = mbg

        for i, sgal in enumerate(these_gals):
            logging.debug('build cmd loop... %i of %i' % (i, len(these_gals) - 1))
            if not hasattr(sgal, 'grid_sfr'):
                sgal.burst_duration()
                mass = sgal.data.get_col('m_ini')[sgal.rec]
                tot_mass = np.sum(mass)
                # the mass of stars formed per year in the grid
                grid_sfr = tot_mass / sgal.burst_length
                sgal.grid_sfr = grid_sfr
                sgal.mass = mass
            else:
                grid_sfr = sgal.grid_sfr

            # the ratio of mass formed in the sfr_file to the grid per year
            sf_frac = sfr_arr[i] / grid_sfr

            frac_mass = sfr_arr[i] * sgal.burst_length

            if frac_mass < 1:
                logging.info('less than 1 msun...')
                logging.info('sf_mass grid_sfr match_sfr')
                logging.info('%.2f %.2g %.2g' % (frac_mass, grid_sfr,
                                                 sfr_arr[i]))
                continue
            nstars = len(sgal.mass)
            ind_arr = range(nstars)

            # the guess assumes 1. Msun is the average mass. Works ok.
            nstars_guess = int(np.round(sf_frac * nstars))
            predict_ratio = 99.
            broke = 0
            max_try = 0
            while abs(predict_ratio) > tol:
                max_try += 1
                if max_try > 5:
                    logger.info('too may interations... skipping')
                    broke = 1
                    break

                if nstars_guess < min_stars:
                    logger.debug('fewer than %i stars... skipping.' % min_stars)
                    broke = 1
                    break

                if nstars_guess > nstars:
                    # randomly select the guess number but trick
                    # random sample into doing a shuffle and a sample to get
                    # all the stars. I.e.,
                    # nstars_guess = nstars * how_big + how_extra
                    # this is ok*, because we bin the stars into a hess diagram
                    # *as long as the IMF is well sampled in sgal.data
                    how_big = nstars_guess / nstars
                    how_extra = nstars_guess % nstars
                    rand_inds = np.array([random.sample(ind_arr, nstars) for i in range(how_big)])
                    rand_inds = np.squeeze(np.append(rand_inds, random.sample(ind_arr, how_extra)))
                    rand_inds = map(int, rand_inds)
                else:
                    rand_inds = random.sample(ind_arr, nstars_guess)
                guess_mass = np.sum(np.array(sgal.mass)[rand_inds])
                predict_ratio = 1. - frac_mass / guess_mass
                nstars_guess += (predict_ratio * nstars_guess)
                nstars_guess = int(np.round(nstars_guess))

            if broke == 0:
                '''
                if I skip match's stats package, I could add up the bins here.
                # bin up hess
                mbinsize = np.diff(hess_kw['mbin'])[0]
                sgal.make_hess(mbinsize, useasts=True, slice_inds=rand_inds,
                               **hess_kw)
                try:
                    sup_hess += sgal.hess[2]
                except NameError:
                    sup_hess = sgal.hess[2]
                '''
                if split_on_age is True:
                    if i in self.sfr_inds:
                        tofile = mbg
                    else:
                        tofile = obg

                tosave = np.column_stack((sgal.ast_mag1[sgal.rec][rand_inds],
                                          sgal.ast_mag2[sgal.rec][rand_inds]))

                np.savetxt(tofile, tosave, fmt='%.4f %.4f')

            else:
                logger.debug('to: %.1f tf: %.1f sfr: %.4g' % (self.match_sfr[0][i], self.match_sfr[1][i], sfr_arr[i]))

        mbg.close()
        try:
            obg.close()
        except NameError:
            pass

        logger.info('build_model_cmd wrote %s' % match_bg)

    def check_grid(self, object_mass=None, run_trilegal=True,
                   max_sfr_inc_frac=0.2, sfr_arr=None):
        '''
        why is this here, not in model grid class? Because I want it to be
        limited to a sfr file, so I don't go crazy with sim rsp.galaxies sizes.

        this will run rsp.trilegal at a higher mass to make sure there is at least
        as much sf in a time bin as there is estimated in the sfr file.

        this should be tricked into using not only the processed sfr file
        (emcee prior) but also the new sfr array that will be passed. Yeah,
        it will slow down the works.

        if sfr_arr is None, will use initial sfr from sfr_file.
        '''
        if not hasattr(self, 'sgals'):
            print 'building sfh'
            self.build_sfh()

        extra = 1.
        if sfr_arr is None:
            sfr_arr = self.match_sfr[2]
            extra += max_sfr_inc_frac

        self.sgals.sum_attr('m_ini')
        self.grid_sfr = []
        ages = np.array([])
        zs = np.array([])
        for i, sgal in enumerate(self.sgals.rsp.galaxies):
            sgal.burst_duration()
            grid_sfr = sgal.sum_m_ini / sgal.burst_length
            self.grid_sfr.append(grid_sfr)  # Msun/year
            if sfr_arr[i] * extra > grid_sfr:
                ages = np.append(ages, self.split_on_key(sgal.name, 'to'))
                zs = np.append(zs, self.split_on_key(sgal.name, 'z'))
                self.over_write = True
        if len(ages) == 0:
            print 'you\'re golden!'
            return
        if object_mass is None:
            object_mass = self.object_mass
        new_objmass = object_mass * 10
        assert new_objmass < 5e7, 'obj mass is getting out of hand'
        galaxy_inkw = {'object_mass': new_objmass}
        here = os.getcwd()
        os.chdir(self.location)
        i = -1
        for age, z in zip(ages,zs):
            i += 1
            to = age
            tf = age + self.dlogt
            sfh_file = self.filename_fmt(self.sfr_pref,to, tf, z)
            galaxy_input = self.filename_fmt(self.inp_pref, to, tf, z)
            output =  self.filename_fmt(self.out_pref, to, tf, z)
            print 'now doing %.2f-%.2f, %.4f %g' % (to, tf, z, new_objmass)
            self.write_sfh_file(sfh_file, to, tf, z)
            self.make_galaxy_input(sfh_file, galaxy_input,
                                   galaxy_inkw=galaxy_inkw)
            # run rsp.trilegal
            if run_trilegal is True:
                rsp.trilegal.run_trilegal(self.cmd_input, galaxy_input,
                                           output)
            else:
                print to, tf, self.grid_sfr[i], sfr_arr[i]
        os.chdir(here)

    def build_sfh(self):
        '''

        '''
        if hasattr(self, 'sgals'):
            return
        if max(self.age) > 12.:
            self.lage = np.round(np.log10(self.age), 2)
        else:
            self.lage = np.round(self.age, 2)
            self.age = 10**self.lage

        assert np.diff(self.lage[0::2][:2]) > self.dlogt, \
            'time steps too small in sfr_file.'

        # sfr is series of step functions get only values
        sinds, = np.nonzero(np.round(self.sfr, 6))
        to = self.lage[sinds][0::4]
        tf = self.lage[sinds][1::4]
        sfr = self.sfr[sinds][0::4]
        z = np.round([(self.z[i] + self.z[i + 1]) / 2. for i in sinds[0:
                     :4]], 4)

        self.grid_values('to','tf')
        maxagebin = np.max(self.grid_tos)
        (ainds, ) = np.nonzero(to < maxagebin)

        to = to[ainds]
        tf = tf[ainds]
        sfr = sfr[ainds]
        z = z[ainds]
        sub_grid_files = \
            np.concatenate([self.filter_grid(younger=tf[i],
                            older=to[i], z=z[i]) for i in
                            range(len(to))])

        # no joining or interpolation so must have 1:1 match for proper
        # sfr scaling of the grid cmd.

        assert len(sfr) == len(sub_grid_files), \
            'sfr and grid length is not the same'

        # load all grid files as sim rsp.galaxies.

        sgals = [rsp.galaxies.simgalaxy(sgf, self.filter1, self.filter2,
                 photsys=self.photsys) for sgf in sub_grid_files]
        self.sgals = rsp.galaxies.rsp.galaxies(sgals)

        # not sure why this gets unsorted, but best to be careful...

        self.sort_on_key('to')
        self.match_sfr = np.array([to[self.sort_inds], tf[self.sort_inds],
                                   sfr[self.sort_inds], z[self.sort_inds]])
        self.sfr_arr = self.match_sfr[2]

    def sort_on_key(self, key):
        '''
        sorts on some _ split key value, key to val map is defined in key_map.
        there should be some way to combine that with fileformat but I'm
        trying to go quickly...
        ex:
        self.sort_on_key('to')
        '''

        assert hasattr(self, 'sgals'), 'need sgals initialized'

        names = [self.sgals.rsp.galaxies[i].name for i in
                 range(len(self.sgals.rsp.galaxies))]
        val = self.key_map(key)
        sinds = [names.index(x) for x in sorted(names, key=lambda n: \
                 float(n.split('_')[val]))]
        self.sgals.galaxy_names = np.array(names)[sinds]
        self.sgals.rsp.galaxies = np.array(self.sgals.rsp.galaxies)[sinds]
        self.sort_inds = sinds


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
        
        clean_up = indict.get('clean_up', True)
        if clean_up:
            logger.info('now cleaning up files')
            mg.delete_columns_from_files()


if __name__ == '__main__':
    main(sys.argv[1:])