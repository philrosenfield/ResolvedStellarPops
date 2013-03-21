import matplotlib.pyplot as plt
import numpy as np
import math_utils
import os
import logging
logger = logging.getLogger()
import fileIO
import graphics


class HRD(object):
    def __init__(self, age, logl, logte, mass, eep_list, ieep):
        self.age = np.array(age)
        self.logl = np.array(logl)
        self.logte = np.array(logte)
        self.mass = mass
        self.eep_dict = dict(zip(eep_list, np.array(ieep)))


class PadovaIsoch(object):
    def __init__(self, ptcri_file):
        self.ptcri_file = ptcri_file
        self.read_padova_isoch()

    def read_padova_isoch(self):
        '''
        returns age, logl, logte.
        '''
        with open(self.ptcri_file, 'r') as p:
            lines = np.array(p.readlines())

        # keep the space because !M exists in the files.
        inds, masses = zip(*[(i, float(l.split(' M=')[1].split()[0])) for
                             (i, l) in enumerate(lines) if ' M=' in l])

        last = [i for (i, l) in enumerate(lines) if '!M' in l][0] - 2
        inds = np.append(inds, last)
        self.masses = np.array(masses)
        for i in range(len(inds) - 1):
            mass = masses[i]
            isoc_inds = np.arange(inds[i], inds[i+1])
            age, logl, logte = zip(*[map(float, l[:35].split())
                                     for l in lines[isoc_inds]])
            eep_list, ieep = zip(*[l[35:].split() for l in lines[isoc_inds]
                                   if len(l[35:].split()) == 2])
            ieep = map(int, ieep)
            self.__setattr__('iso_m%s' % self.strmass(mass),
                             HRD(age, logl, logte, mass, eep_list, ieep))

        return

    def strmass(self, mass):
        return ('%.3f' % mass).replace('.', '_')

    def plot_padova_isoch(self, figname=None, masses=None):
        if figname is None:
            figname = fileIO.replace_ext(self.ptcri_file, '.png')
        figtitle = os.path.split(self.ptcri_file)[1].replace('_', '\_')

        if masses is None:
            masses = self.masses

        max = np.max([len(self.__dict__['iso_m%s' % self.strmass(m)].eep_dict.values()) for m in masses])
        cols = graphics.GraphicsUtils.discrete_colors(max)
        ax = plt.axes()
        for j, mass in enumerate(masses):
            isoch = self.__dict__['iso_m%s' % self.strmass(mass)]
            logte = isoch.logte
            logl = isoch.logl
            ptinds = np.array(isoch.eep_dict.values())

            ax.plot(logte, logl, color='black', alpha=0.2)
            if len(masses) == 1:
                ax.plot(logte[ptinds], logl[ptinds], '.', color='blue', alpha=0.3)

            sinds = np.argsort(ptinds)
            ptinds = ptinds[sinds]
            labels = np.array(isoch.eep_dict.keys())[sinds]
            for i in range(len(ptinds)):
                if j == 0:
                    ax.plot(logte[ptinds[i]], logl[ptinds[i]], 'o',
                            color=cols[i], label='%s' % labels[i].replace('_', '\_'))
                else:
                    ax.plot(logte[ptinds[i]], logl[ptinds[i]], 'o',
                            color=cols[i])

        ax.set_xlim(4.6, 3.5)
        ax.set_xlabel('$LOG\ TE$', fontsize=20)
        ax.set_ylabel('$LOG\ L$', fontsize=20)
        ax.set_title(r'$\textrm{%s}$' % figtitle, fontsize=16)
        plt.legend()
        plt.savefig(figname)


def write_trilegal_sim(sgal, outfile, slice_inds=None):
    '''
    writes trilegal sim to outfile.
    '''
    header = sgal.get_header()

    with open(outfile, 'w') as f:
        if not header.endswith('\n'):
            header += '\n'
        f.write('%s' % header)
        np.savetxt(f, sgal.data.data_array, fmt='%10.5f')
    return


def write_spread(sgal, outfile, overwrite=False, slice_inds=None):
    if overwrite or not os.path.isfile(outfile):
        # ast corrected filters are filter_cor, this may be changed... anyway
        # need someway to check if ast corrections were made.
        cors = [c for c in sgal.data.key_dict.keys() if '_cor' in c]
        if len(cors) == 0:
            logger.error('can not make spread file without ast_corrections')
            return -1

        filt1, filt2 = np.sort(cors)
        if hasattr(sgal, 'ast_mag1'):
            # this isn't only a trilegal catalog, it's already been corrected
            # with asts, and sliced for only recovered asts. see simgalaxy.
            cor_mag1 = sgal.ast_mag1[sgal.rec]
            cor_mag2 = sgal.ast_mag2[sgal.rec]
        else:
            # it's a trilegal catalog, now with ast corrections, though was not
            # loaded with them, perhaps they were just written to a new file.
            cor_mag1_full = sgal.data.get_col(filt1)
            cor_mag2_full = sgal.data.get_col(filt2)
            rec1, = np.nonzero(np.abs(cor_mag1_full) < 90.)
            rec2, = np.nonzero(np.abs(cor_mag2_full) < 90.)
            if slice_inds is not None:
                rec = list(set(rec1) & set(rec2) & set(slice_inds))
            else:
                rec = list(set(rec1) & set(rec2))
            cor_mag1 = cor_mag1_full[rec]
            cor_mag2 = cor_mag2_full[rec]

        cor_mags = np.column_stack((cor_mag1, cor_mag2))

        with open(outfile, 'w') as f:
            f.write('# %s %s \n' % (filt1, filt2))
            np.savetxt(f, cor_mags, fmt='%10.5f')
        logger.info('wrote %s' % outfile)
    else:
        logger.warning('%s exists, send overwrite=True arg to overwrite' % outfile)
    return outfile


def change_galaxy_input(galaxy_input, **kwargs):
    '''
    if no kwargs are given, will write None as object_mass and object_sfr_file.
    see galaxy_input_dict()-
    '''
    input_pars = fileIO.input_parameters(galaxy_input_dict())
    input_pars.add_params(kwargs)
    input_pars.write_params(galaxy_input, galaxy_input_fmt())


def find_mag_num(file_mag, filter1):
    file_mag = os.path.join(os.environ['TRILEGAL_ROOT'], file_mag)
    with open(file_mag, 'r') as f:
        line = f.readlines()[1].strip().split()
    try:
        return line.index(filter1)
    except ValueError:
        print '%s not found in %s.' (filter1, file_mag)


def galaxy_input_dict():
    return {'coord_kind': 1,
            'coord1': 0.0,
            'coord2': 0.0,
            'field_area': 1.0,
            'file_mag': 'tab_mag_odfnew/tab_mag_wfpc2.dat',
            'mag_num': 16,
            'mag_limit_val': 5,
            'mag_resolution': 0.1,
            'r_sun': 8500.0,
            'z_sun': 24.2,
            'file_imf': 'tab_imf/imf_chabrier_lognormal.dat',
            'binary_kind': 0,
            'binary_frac': 0.0,
            'binary_mrinf': 0.7,
            'binary_mrsup': 1.0,
            'extinction_kind': 0,
            'extinction_rho_sun': 0.00015,
            'extinction_infty': 0.045756,
            'extinction_infty_disp': 0.0,
            'extinction_h_r': 100000.0,
            'extinction_h_z': 110.0,
            'thindisk_kind': 0,
            'thindisk_rho_sun': 59.0,
            'thindisk_h_r': 2800.0,
            'thindisk_r_min': 0.0,
            'thindisk_r_max': 15000.0,
            'thindisk_h_z0': 95.0,
            'thindisk_hz_tau0': 4400000000.0,
            'thindisk_hz_alpha': 1.6666,
            'thindisk_sfr_file': 'tab_sfr/file_sfr_thindisk_mod.dat',
            'thindisk_sfr_mult_factorA': 0.8,
            'thindisk_sfr_mult_factorB': 0.0,
            'thickdisk_kind': 0,
            'rho_thickdisk_sun': 0.0015,
            'thickdisk_h_r': 2800.0,
            'thickdisk_r_min': 0.0,
            'thickdisk_r_max': 15000.0,
            'thickdisk_h_z': 800.0,
            'thickdisk_sfr_file': 'tab_sfr/file_sfr_thickdisk.dat',
            'thickdisk_sfr_mult_factorA': 1.0,
            'thickdisk_sfr_mult_factorB': 0.0,
            'halo_kind': 0,
            'halo_rho_sun': 0.00015,
            'halo_r_eff': 2800.0,
            'halo_q': 0.65,
            'halo_sfr_file': 'tab_sfr/file_sfr_halo.dat',
            'halo_sfr_mult_factorA': 1.0,
            'halo_sfr_mult_factorB': 0.0,
            'bulge_kind': 0,
            'bulge_rho_central': 76.0,
            'bulge_am': 1900.0,
            'bulge_a0': 100.0,
            'bulge_eta': 0.5,
            'bulge_csi': 0.6,
            'bulge_phi': 20.0,
            'bulge_cutoffmass': 0.8,
            'bulge_sfr_file': 'tab_sfr/file_sfr_bulge.dat',
            'bulge_sfr_mult_factorA': 1.0,
            'bulge_sfr_mult_factorB': 0.0,
            'object_kind': 1,
            'object_mass': None,
            'object_dist': 10.,
            'object_avkind': 1,
            'object_av': 0.0,
            'object_cutoffmass': 0.8,
            'object_sfr_file': None,
            'object_sfr_mult_factorA': 1.0,
            'object_sfr_mult_factorB': 0.0,
            'output_file_type': 1}


def galaxy_input_fmt():
    fmt = \
        """%(coord_kind)i %(coord1).1f %(coord2).1f %(field_area).1f # 1: galactic l, b (deg), field_area (deg2) # 2: ra dec in ore ( gradi 0..24)

%(file_mag)s  # kind_mag, file_mag
%(mag_num)i %(mag_limit_val)i %(mag_resolution).1f # Magnitude: num, limiting value, resolution

%(r_sun).1f %(z_sun).1f # r_sun, z_sun: sun radius and height on disk (in pc)

%(file_imf)s # file_imf
%(binary_kind)i # binary_kind: 0=none, 1=yes
%(binary_frac).1f # binary_frac: binary fraction
%(binary_mrinf).1f %(binary_mrsup).1f  # binary_mrinf, binary_mrsup: limits of mass ratios if binary_kind=1

%(extinction_kind)i  # extinction kind: 0=none, 1=exp with local calibration, 2=exp with calibration at infty
%(extinction_rho_sun)f  # extinction_rho_sun: local extinction density Av, in mag/pc
%(extinction_infty)f %(extinction_infty_disp).1f  # extinction_infty: extinction Av at infinity in mag, dispersion
%(extinction_h_r).1f %(extinction_h_z).1f  # extinction_h_r, extinction_h_z: radial and vertical scales

%(thindisk_kind)i  # thindisk kind: 0=none, 1=z_exp, 2=z_sech, 3=z_sech2
%(thindisk_rho_sun).1f  # thindisk_rho_sun: local thindisk surface density, in stars formed/pc2
%(thindisk_h_r).1f %(thindisk_r_min).1f %(thindisk_r_max).1f  # thindisk_h_r, thindisk_r_min,max: radial scale, truncation radii
%(thindisk_h_z0).1f %(thindisk_hz_tau0).1f %(thindisk_hz_alpha)%f # thindisk_h_z0, thindisk_hz_tau0, thindisk_hz_alpha: height now, increase time, exponent
%(thindisk_sfr_file)s %(thindisk_sfr_mult_factorA).1f %(thindisk_sfr_mult_factorB).1f # File with (t, SFR, Z), factors A, B (from A*t + B)

%(thickdisk_kind)i # thickdisk kind: 0=none, 1=z_exp, 2=z_sech2, 3=z_sech2
%(rho_thickdisk_sun)f  # rho_thickdisk_sun: local thickdisk volume density, in stars formed/pc3
%(thickdisk_h_r).1f %(thickdisk_r_min).1f %(thickdisk_r_max).1f # thickdisk_h_r, thickdisk_r_min,max: radial scale, truncation radii
%(thickdisk_h_z).1f # thickdisk_h_z: scale heigth (a single value)
%(thickdisk_sfr_file)s %(thickdisk_sfr_mult_factorA).1f %(thickdisk_sfr_mult_factorB).1f  # File with (t, SFR, Z), factors A, B

%(halo_kind)i # halo kind: 0=none, 1=1/r^4 cf Young 76, 2=oblate cf Gilmore
%(halo_rho_sun)f # 0.0001731 0.0001154 halo_rho_sun: local halo volume density, to be done later: 0.001 for 1
%(halo_r_eff).1f %(halo_q).2f #  halo_r_eff, halo_q: effective radius on plane (about r_sun/3.0), and oblateness
%(halo_sfr_file)s %(halo_sfr_mult_factorA).1f %(halo_sfr_mult_factorB).1f # File with (t, SFR, Z), factors A, B

%(bulge_kind)i  # bulge kind: 0=none, 1=cf. Bahcall 86, 2=cf. Binney et al. 97
%(bulge_rho_central).1f # bulge_rho_central: central bulge volume density, unrelated to solar position
%(bulge_am).1f %(bulge_a0).1f #  bulge_am, bulge_a0: scale length and truncation scale length
%(bulge_eta).1f %(bulge_csi).1f %(bulge_phi).1f #  bulge_eta, bulge_csi, bulge_phi0: y/x and z/x axial ratios, angle major-axis sun-centre-line (deg)
%(bulge_cutoffmass).1f # bulge_cutoffmass: (Msun) masses lower than this will be ignored
%(bulge_sfr_file)s %(bulge_sfr_mult_factorA).1f %(bulge_sfr_mult_factorB).1f # File with (t, SFR, Z), factors A, B

%(object_kind)i # object kind: 0=none, 1=at fixed distance
%(object_mass)g %(object_dist).1f # object_mass, object_dist: total mass inside field, distance
%(object_avkind)i %(object_av).1f # object_avkind, object_av: Av added to foregroud if =0, not added if =1
%(object_cutoffmass).1f # object_cutoffmass: (Msun) masses lower than this will be ignored
%(object_sfr_file)s %(object_sfr_mult_factorA).1f %(object_sfr_mult_factorB).1f # File with (t, SFR, Z), factors A, B # la vera eta' e' t_OK=A*(t+B)

%(output_file_type)i # output file: 1=data points"""
    return fmt


def cmd_input_fmt():
    fmt = \
        """%(kind_tracks)i %(file_isotrack)s %(file_lowzams)s # kind_tracks, file_isotrack, file_lowzams
%(kind_tpagb)i %(file_tpagb)s # kind_tpagb, file_tpagb
%(kind_postagb)i %(file_postagb)s # kind_postagb, file_postagb DA VERIFICARE file_postagb
%(ifmr_kind)i %(file_ifmr)s # ifmr_kind, file with ifmr
%(kind_rgbmloss)i %(rgbmloss_law)s %(rgbmloss_efficiency).2f # RGB mass loss: kind_rgbmloss, law, and its efficiency
################################explanation######################
kind_tracks: 1= normal file
file_isotrack: tracks for low+int mass
file_lowzams: tracks for low-ZAMS
kind_tpagb: 0= none
        1= Girardi et al., synthetic on the flight, no dredge up
        2= Marigo & Girardi 2001, from file, includes mcore and C/O
        3= Marigo & Girardi 2007, from file, includes period, mode and mloss
        4= Marigo et al. 2011, from file, includes slope
file_tpagb: tracks for TP-AGB

kind_postagb: 0= none
        1= from file
file_postagb: PN+WD tracks

kind_ifmr: 0= default
           1= from file

kind_rgbmloss: 0=off
               1=on (with law=Reimers for the moment)"""
    return fmt


def cmd_input_dict():
    return {'kind_tracks': 2,
            'file_isotrack': 'isotrack/parsec/CAF09_S12D_NS.dat',
            'file_lowzams': 'isotrack/bassazams_fasulla.dat',
            'kind_tpagb': 4,
            'file_tpagb': 'isotrack/isotrack_agb/tracce_CAF09_S_JAN13.dat',
            'kind_postagb': 1,
            'file_postagb': 'isotrack/final/pne_wd_test.dat',
            'ifmr_kind': 0,
            'file_ifmr': 'tab_ifmr/weidemann.dat',
            'kind_rgbmloss': 1,
            'rgbmloss_law': 'Reimers',
            'rgbmloss_efficiency': 0.2}


class trilegal_sfh(object):
    def __init__(self, filename, galaxy_input=True):
        '''
        file can be galaxy input file for trilegal or trilegal age, sfr, z
        file.
        '''
        if galaxy_input is True:
            self.galaxy_input = filename
            self.current_galaxy_input = filename
        else:
            self.sfh_file = filename
            self.current_sfh_file = filename
        self.load_sfh()

    def load_sfh(self):
        if not hasattr(self, 'sfh_file'):
            with open(self.galaxy_input, 'r') as f:
                lines = f.readlines()
            self.sfh_file = lines[-3].split()[0]
            self.current_sfh_file = self.sfh_file[:]
            self.galaxy_input_sfh_line = ' '.join(lines[-3].split()[1:])

        self.age, self.sfr, self.z = np.loadtxt(self.sfh_file, unpack=True)
        # should I do this with dtype?
        self.z = np.round(self.z, 4)

    def __format_cut_age(self, cut1_age):
        '''
        takes the > or < out of the string, and makes it in yrs.
        '''
        flag = cut1_age[0]
        yrfmt = 1.
        possible_yrmfts = {'Gyr': 1e9, 'Myr': 1e6, 'yr': 1.}
        for py, yrfmt in sorted(possible_yrmfts.items(),
                                key=lambda (k, v): (v, k), reverse=True):
            if py in str(cut1_age):
                import matplotlib
                if matplotlib.cbook.is_numlike(flag):
                    cut1_age = float(cut1_age.replace(py, ''))
                    flag = ''
                else:
                    cut1_age = float(cut1_age.replace(py, '').replace(flag, ''))
                cut1_age *= yrfmt
        return cut1_age, flag

    def increase_sfr(self, factor, cut_age, over_write_galaxy_input=True):
        '''
        cut_age is in Myr.
        '''
        new_fmt = '%s_inc%i.dat'
        new_file = new_fmt % (self.sfh_file.replace('.dat', ''), factor)
        if over_write_galaxy_input is True:
            galaxy_input = self.galaxy_input
        else:
            galaxy_input = new_fmt % (self.galaxy_input.replace('.dat', ''),
                                      factor)

        # copy arrays to not overwrite attributes
        sfr = self.sfr[:]
        age = self.age[:]
        z = self.z[:]

        # convert cut_age to yr
        cut_age *= 1e6

        inds, = np.nonzero(age <= (cut_age))
        sfr[inds] *= factor
        np.savetxt(new_file, np.array([age, sfr, z]).T)
        # update galaxy_input file
        print 'this is broken!!!!!'
        #lines[-3] = '%s %s \n' % (new_file, self.galaxy_input_sfh_line)
        #logger.debug('new line: %s' % lines[-3])
        #with open(galaxy_input, 'w') as out:
        #    [out.write(l) for l in lines]

        self.current_galaxy_input = galaxy_input
        self.current_sfh_file = new_file
        return self.current_galaxy_input

    def edit_sfh_file(self, z, cut_age='<400Myr',
                      over_write_sfh_file=False,
                      new_sfh_file=None):
        '''
        give a constant [median value] sfr for all ages > or < than some cut_age
        (Gyr or Myr).
        saves new sfr to new_sfr_file or sfr_file_csfr[cut_age].dat (without ><)
        '''
        # the new sfr file will be the same as the old but with a different
        # extension
        print 'this is broken!!'
        if not over_write_sfh_file or new_sfh_file is None:
            if cut_age.startswith('>') or cut_age.startswith('<'):
                cutname = cut_age[1:]
            else:
                cutname = cut_age
            new_sfh_file = fileIO.replace_ext(self.sfh_file,
                                              '_csfr%s.dat' % cutname)
        else:
            new_sfh_file = self.sfh_file[:]

        if cut_age:
            # takes the > or < out of the string, and makes it in yrs.
            pass
            #cut_age, flag = format_cut_age(cut_age)
        else:
            logger.error('nothing to do.')
            return -1

        if flag == '<' or not flag:
            overwrites, = np.nonzero(sfh[:, 0] < cut_age)
            saves, = np.nonzero(sfh[:, 0] >= cut_age)
        else:
            overwrites, = np.nonzero(sfh[:, 0] > cut_age)
            saves, = np.nonzero(sfh[:, 0] <= cut_age)
        # this line should break the program... go ahead and fix it if you 
        # still use it.
        sfhtmp = sfh[overwrites]
    
        # sometimes sfr = 1e-16 so I'm also rounding.
        is_sf, = np.nonzero(np.round(sfhtmp[:, 1], 6))
    
        # assign constant value to all sfr in this range
        median_recent_sfr = np.median(sfhtmp[:, 1][is_sf])
        if not new_sfr:
            new_sfr = median_recent_sfr
        if type(new_sfr) is int or type(new_sfr) is float:
            new_sfr = median_recent_sfr * float(new_sfr)
        sfhtmp[:, 1][is_sf] = new_sfr
    
        # assign new metallicity
        sfhtmp[:, 2] = z 
    
        if saves[0] < overwrites[0]:
            new_sfh = np.vstack((sfh[saves], sfhtmp))
        else:
            new_sfh = np.vstack((sfhtmp, sfh[saves]))
    
        np.savetxt(new_sfr_file, new_sfh, fmt=['%.5g', '%.6f', '%.4f'])
    
        return new_sfr_file

    def change_galaxy_input_file(self, over_write=True, **kwargs):
        '''
        
        This should go away! Use fileIO.input_parameters class
        and change galaxy_input
        
        this doesn't work for everything. only tested on object_mass.
        what's needed is a parser for the galaxy_input file, and then
        replace by dict key...
        
        also do we keep track of every new gal input?
        '''
        print 'stop using change_galaxy_input_file!'
        
        with open(self.current_galaxy_input, 'r') as f:
            lines = f.readlines()
        for k, v in kwargs.items():
            try:
                (i, line), = [(i, l) for i, l in enumerate(lines) if k in l]
            except ValueError:
                logger.error('%s not found' % k)
                continue
            vals, info = line.strip().split('#')
            val_ind = info.strip().index(k)
            old_vals = map(float, vals.split())
            if over_write is False:
                logger.debug('current galaxy input value: %s=%g:' % (k, old_vals[val_ind]))
                return old_vals[val_ind]
            
            new_vals = old_vals.copy()
            new_vals[val_ind] = v
            new_line = '%s # %s\n' % (' '.join(['%g' % x for x in new_vals]), info)
            logger.debug('new line: %s' % new_line.strip())
            lines[i] = new_line

        with open(self.current_galaxy_input, 'w') as o:
            [o.write(l) for l in lines]

        return self.current_galaxy_input


def find_photsys_number(photsys, filter):
    mag_file = os.path.join(os.environ['BCDIR'],
                            'tab_mag_odfnew/tab_mag_%s.dat' % photsys)
    mf = open(mag_file, 'r')
    nmags = mf.readline()
    magline = mf.readline().strip().split()
    return magline.index(filter), mag_file


def write_pytrilegal_params(sfh, parfile, photsys, filter, object_mass=1e7):
    mag_num = find_photsys_number(photsys, filter)[0]
    lines = (
            "photosys           %s" % photsys,
            "mag_num            %s" % mag_num,
            "mag_lim            2               ",
            "mag_res            0.1             ",
            "dust               0               ",
            "dustM              dpmod60alox40   ",
            "dustC              AMCSIC15        ",
            "binary_kind        0               ",
            "binary_frac        0.              ",
            "extinction_kind    0               ",
            "thindisk_kind      0               ",
            "thickdisk_kind     0               ",
            "halo_kind          0               ",
            "bulge_kind         0               ",
            "object_kind        1               ",
            "object_mass        %g" % object_mass,
            "object_dist        10.0            ",
            "object_avkind      1               ",
            "object_av          0.000           ",
            "object_cutoffmass  0.8             ",
            "object_sfr         %s" % os.path.abspath(sfh),
            "object_sfr_A       1.              ",
            "object_sfr_B       0.0             ")

    with open(parfile, 'w') as oo:
        [oo.write(line + '\n') for line in lines]

    logger.info('wrote  %s' % parfile)
    return


def run_trilegal(cmd_input, galaxy_input, output):
    '''
    runs trilegal with os.system. might be better with subprocess? Also
    changes directory to trilegal root, if that's not in a .cshrc need to
    somehow tell it where to go.

    to do:
    add -a or any other flag options
    possibly add the stream output to the end of the output file.
    '''
    here = os.getcwd()
    os.chdir(os.environ['TRILEGAL_ROOT'])

    logger.info('running trilegal...')
    cmd = 'code/main -f %s -a -l %s %s > trilegal.msg\n' % (cmd_input,
                                                            galaxy_input, output)
    logger.debug(cmd)
    t = os.system(cmd)
    logger.info('done.')

    if t != 0:
        logger.debug('\n'.join([l.strip()
                                for l in open('trilegal.msg').readlines()]))
    else:
        os.remove('trilegal.msg')
    os.chdir(here)
    return


def run_pytrilegal(cmd_input, parfile, inp, out, agb=True, tagstages=True):
    '''
    This is to run Marco's python trilegal implementation.
    '''
    from subprocess import Popen, PIPE
    cmd = "/Users/phil/research/PyTRILEGAL/run_trilegal.py  -e code/main"
    cmd += "%s" % parfile
    if agb is True:
        cmd += " -a"
    if tagstages is True:
        cmd += " -l"
    cmd += " -i  %s" % inp
    cmd += " -o  %s" % out
    cmd += " -f  %s" % cmd_input

    logger.debug(cmd)
    p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE, close_fds=True)
    stdout, stderr = (p.stdout, p.stderr)
    lines = stdout.readlines()
    p.wait()
    # append the standard output to the end of the output file
    with open(out, 'a') as ff:
        for l in lines:
            ff.write("# %s" % l)
    
    logger.debug([l.strip() for l in lines])
    return


def get_args(filename, ext='.dat'):
    filename = os.path.split(filename)[1]
    a = filename.split(ext)[0]
    a = a.replace('_', ' ').replace('R1', '').replace('isoch', '')
    s = ''.join(c for c in a if not c.isdigit())
    s = s.replace('.', ' ').split()
    d = {}
    x = a[:]
    s.append(' ')
    for i in range(len(s) - 1):
        if s[i] in a:
            x = x.replace(s[i], '')
            y = x.split(s[i + 1])[0]
            x = x.split(s[i + 1])[-1]
            d[s[i]] = float(y)
    try:
        del d['F']
    except KeyError:
        pass
    return d
    

def get_stage_label(region):
    # see parametri.h
    regions = ['PMS', 'MS', 'SUBGIANT', 'RGB', 'HEB', 'RHEB', 'BHEB', 'EAGB',
               'TPAGB', 'POSTAGB', 'WD']
    stage_lab = regions.index(region.upper())
    return stage_lab


def get_label_stage(stage_lab):
    # see parametri.h
    regions = ['PMS', 'MS', 'SUBGIANT', 'RGB', 'HEB', 'RHEB', 'BHEB', 'EAGB',
               'TPAGB', 'POSTAGB', 'WD']
    return regions[stage_lab]


def get_loop_data(cmd_input_file, metallicity):
    filename = read_cmd_input_file(cmd_input_file)['file_isotrack']
    file_isotrack = os.path.join(os.environ['ISOTRACK'],
                                 filename.replace('isotrack/', ''))
    z, y, mh, files = read_isotrack_file(file_isotrack)
    if not metallicity in z:
        logger.warning('%s: metallicity not found.' % get_loop_data.__name__)
        data = 0
    ptcri = [f[0] for f in files if str(metallicity) in f[0] and
             f[0].endswith('INT2')][0].replace('isotrack/', '')
    ptcri_file = os.path.join(os.environ['ISOTRACK'], ptcri)
    d = read_loop_from_ptrci(ptcri_file)
    return d


def string_in_lines(lines):
    for line in lines:
        strs = [math_utils.is_numeric(l) for l in line.strip().split()
                if type(math_utils.is_numeric(l)) == str]
        if len(strs) != 0:
            if ' M=' in line:
                print line, strs


def read_ptcri(ptcri_file):
    d = {}
    lines = open(ptcri_file, 'r').readlines()
    for line in lines:
        if not 'kind' in line:
            continue
        try:
            age, logL, logTe, mod, model, cript, M, mass, npt, kind = \
                line.strip().split()
            try:
                float(cript)
            except ValueError:
                mass = float(mass)
                if not mass in d.keys():
                    d[mass] = {}
                if not cript in d[mass].keys():
                    d[mass][cript] = {}
                d[mass][cript] = {'age': float(age),
                                  'logTe': float(logTe),
                                  'logL': float(logL),
                                  'model': np.int(model),
                                  'mod': np.int(mod)}
        except ValueError:
            pass
    return d


def read_ptcri2(ptcri_file):
    '''
    after .revisegrid
    '''
    lines = open(ptcri_file, 'r').readlines()
    mline = []
    for i, line in enumerate(lines):
        if 'M=' in line:
            mline.append(i)
            continue
    d = {}
    for j in range(len(mline) - 1):
        mass = float(lines[mline[j]].split('M=')[-1].split('PMS')[0])
        for k in range(mline[j] + 1, mline[j + 1] - 1):
            try:
                a, l, t, m = map(float, lines[k].strip().split())
            except:
                data = lines[k].strip().split()
                age, logL, logTe = map(float, data[0:3])
                cript = data[-2]
                model = int(data[-1])
                if not mass in d.keys():
                    d[mass] = {}
                if not cript in d[mass].keys():
                    d[mass][cript] = {}
                d[mass][cript] = {'age': age, 'logTe': logTe, 'logL': logL,
                                  'model': model}
        if mass == 12:
            break
    return d


def read_loop_from_ptrci(ptcri_file):
    lines = open(ptcri_file, 'r').readlines()
    # Get starting points for each track
    line_nums = map(int, lines[4].strip().split())
    # don't forget the first
    line_nums.insert(0, 0)
    # they actually start at line 7, that is 6 if you start counting with 0.
    line_nums = np.array(line_nums) + 6
    mass_lines = np.cumsum(line_nums)
    #ax = plt.axes()
    #col = {'Loop_A': 'red', 'Loop_B': 'blue', 'Loop_C': 'green'}
    d = {}
    for i, line in enumerate(lines):
        if not 'Loop' in line and not 'M=' in line:
            continue
        if 'M=' in line:
            mass = float(line.split('M=')[-1].split('P')[0])
            mline = i
        if 'Loop' in line:
            data = line.strip().split()
            age, logL, logTe = map(float, data[0:3])
            loop = data[3]
            model = int(data[4])
            #ax.plot(logTe, logL, 'o', color = col[loop])
            if not mass in d.keys():
                d[mass] = {}
            if not loop in d[mass].keys():
                d[mass][loop] = {}
            d[mass][loop] = {'age': age, 'logTe': logTe, 'logL': logL,
                             'model': model}
    return d


def read_isotrack_file(filename):
    if os.path.split(filename)[1].startswith('cmd_input'):
        file_isotrack = read_cmd_input_file(filename)['file_isotrack']
        filename = os.path.join(os.environ['ISOTRACK'],
                                file_isotrack.replace('isotrack / ', ''))
    try:
        isotf = open(filename, 'r').readlines()
    except:
        fname = os.path.join(os.environ['ISOTRACK'], 'parsec', filename)
        isotf = open(fname, 'r').readlines()
    nmets = int(isotf[0])
    z = np.zeros(nmets)
    y = np.zeros(nmets)
    mh = np.zeros(nmets)
    files = []
    met_count = 0
    for j in range(len(isotf)):
        if  j == 0:
            continue
        line = isotf[j].strip().split(' ')
        if len(line) == 3:
            z[met_count], y[met_count], mh[met_count] = map(float, line)
            met_count += 1
        else:
            files.append(line)
    return z, y, mh, files


def read_cmd_input_file(filename):
    lines = open(filename, 'r').readlines()
    d = {}
    for line in lines:
        if line.startswith('##'):
            break
        data_line, comment_line = line.strip().split('#')
        data = data_line.split()
        comment = comment_line.strip().split('DA ')[0].split(', ')
        for c, dat in zip(comment, data):
            d[c.strip().replace(' ', '_')] = math_utils.is_numeric(dat)
    return d


def write_cmd_input_file(**kwargs):
    '''
    make a TRILEGAL cmd_input file based on default.

    Send each parameter that is different than default by:
    kwargs = { 'kind_tpagb': 4, 'file_tpagb': 'isotrack/tracce_CAF09_S0.dat'}
    cmd_input_file = write_cmd_input_file(**kwargs)

    To make the default file:
    cmd_input_file = write_cmd_input_file()

    if you don't specify cmd_input_file, output goes to cmd_input_TEMP.dat
    '''
    kind_tracks = kwargs.get('kind_tracks', 2)
    file_isotrack = kwargs.get('file_isotrack', 'isotrack_parsec/CAF09.dat')
    file_lowzams = kwargs.get('file_lowzams', 'isotrack/bassazams_fasulla.dat')
    kind_tpagb = kwargs.get('kind_tpagb', 4)
    # should be isotrack_agb...
    file_tpagb = kwargs.get('file_tpagb',
                            'isotrack/tracce_CAF09_AFEP02_I1_S1.dat')
    kind_postagb = kwargs.get('kind_postagb', 0)
    file_postagb = kwargs.get('file_postagb', 'isotrack/final/pne_wd_test.dat')

    # these are for using cmd2.2:
    kind_mag = kwargs.get('kind_mag', None)
    photsys = kwargs.get('photsys', 'wfpc2')
    file_mag = 'tab_mag_odfnew/tab_mag_' + photsys + '.dat'
    kind_imf = kwargs.get('kind_imf', None)
    file_imf = kwargs.get('file_imf', 'tab_imf/imf_chabrier_lognormal.dat')

    # if not using cmd2.2:
    if kind_imf is None:
        kind_imfr = kwargs.get('kind_imfr', 0)
        file_imfr = kwargs.get('file_imfr', 'tab_ifmr/weidemann.dat')

    track_comments = '# kind_tracks, file_isotrack, file_lowzams'
    tpagb_comments = '# kind_tpagb, file_tpagb'
    pagb_comments = '# kind_postagb, file_postagb DA VERIFICARE file_postagb'
    mag_comments = '# kind_mag, file_mag'
    imf_comments = '# kind_imf, file_imf'
    imfr_comments = '# ifmr_kind, file with ifmr'
    footer = (
        "################################explanation######################\n"
        "kind_tracks: 1 = normal file\n"
        "file_isotrack: tracks for low + int mass\n"
        "file_lowzams: tracks for low - ZAMS\n"
        "kind_tpagb:\n"
        "    0 = none\n"
        "    1 = Girardi et al., synthetic on the flight, no dredge up\n"
        "    2 = Marigo & Girardi 2001, from file, includes mcore and C/O\n"
        "    3 = Marigo & Girardi 2007, from file, includes period,\n"
        "        mode and mloss\n"
        "    4 = Marigo et al. 2011, from file, includes slope\n"
        "file_tpagb: tracks for TP - AGB\n"
        "kind_postagb:\n"
        "    0 = none\n"
        "    1 = from file\n"
        "file_postagb: PN + WD tracks\n"
        "kind_ifmr: 0 = default\n"
        "           1 = from file\n")
    cmd_input_file = kwargs.get('cmd_input_file', 'cmd_input_TEMP.dat')
    fh = open(cmd_input_file, 'w')
    formatter = '%i %s %s \n'
    fh.write('%i %s %s %s \n' % (kind_tracks, file_isotrack, file_lowzams,
                                 track_comments))
    fh.write(formatter % (kind_tpagb, file_tpagb, tpagb_comments))
    fh.write(formatter % (kind_postagb, file_postagb, pagb_comments))
    if kind_mag is not None:
        fh.write(formatter % (kind_mag, file_mag, mag_comments))
    if kind_imf is None:
        fh.write(formatter % (kind_imfr, file_imfr, imfr_comments))
    else:
        fh.write(formatter % (kind_imf, file_imf, imf_comments))
    fh.write(footer)
    fh.close()
    logger.info('%s wrote %s' % (write_cmd_input_file.__name__,
                                 cmd_input_file))
    return cmd_input_file


def colorplot_by_bin(x, y, marker, z, bins=None, labels=[], **kwargs):
    if bins is None:
        bins = np.linspace(min(z), max(z), 8)
    for bin in bins:
        labels.append('%.3f' % bin)
    colors = kwargs.get('colors', GraphicsUtils.discrete_colors(len(bins)))
    ax = kwargs.get('ax', plt.axes())
    for i in range(len(bins) - 1):
        # bins go 0 to bin[0], bin[i-1] to bin[i], bin[-1] and up
        if i == 0:
            ind = np.nonzero(z < bins[i])[0]
        elif i == len(bins) - 1:
            ind = np.nonzero(z > bins[i])[0]
        else:
            ind = np.nonzero((z < bins[i]) & (z > bins[i - 1]))[0]
        if len(ind) == 0:
            continue
        if labels is None:
            ax.plot(x[ind], y[ind], marker, color=colors[i])
        else:
            ax.plot(x[ind], y[ind], marker, label=labels[i], color=colors[i])
    return ax


def estimate_sfh(gal, sgal, sfr_file, metallicity, **kwargs):
    '''
    tagged_file: string location of tagged photometry
    sim_cmd_name: string
    sim_cmd_name should be a large simulation of one metallicity at constant
    sfr.

    This takes a tagged fits file (one that has PARSEC labels) and uses the
    bright MS to estimate the relative SFR for a trilegal simulated cmd. It
    returns a formatted sfr file to run in trilegal for a more realistic
    simulation.

    kwargs
    z -- metallicity for the sfr file.
    save_cmd [False] save a color, mag ascii file of normalized big sim.
    mag_bins [np.linspace(22, 25, 5)] bins to divide up the MS.
    target [reads from tagged_file]
    '''

    # save cmd file
    save_cmd = kwargs.get('save_cmd', False)

    # single out main sequence
    gal.all_stages('ms')

    # mag bins to divide up MS
    mag_bins = kwargs.get('mag_bins')
    if mag_bins is None:
        mag_bins = np.arange(np.round(np.min(gal.mag2[gal.ims])),
                             np.round(np.max(gal.mag2[gal.ims])))
    # make photometry cut
    mag2limit = gal.comp50mag2
    mag1limit = gal.comp50mag1
    gst50 = gal.cut_mag_inds(mag2limit, mag1cut=mag1limit)
    gst50_ms = list(set(gst50) & set(gal.ims))

    # bin up data MS
    dhist, mag_bins = np.histogram(gal.mag2[gst50_ms], bins=5)
    dhist = np.array(map(float, dhist))
    print dhist
    print mag_bins

    # convert AbsMag to mag
    sgal.convert_mag(target=gal.target)

    # bin up sim MS
    sgal.all_stages('ms')
    shist, lixo = np.histogram(sgal.mag2[sgal.ims], bins=mag_bins)
    shist = np.array(map(float, shist))

    # calculate relative SFR
    sfr = shist / dhist
    if np.sum(sfr) == 0:
        logger.error('no star formation!')

    # grab indices of sim MS stars in each mag bin
    inds = np.digitize(sgal.mag2[sgal.ims], mag_bins)

    # grab upper limit age for each mag bin
    age = sgal.data.get_col('logAge')
    lages = [age[inds == i] for i in range(len(mag_bins) - 1)]

    # write out sfr file for TRILEGAL.
    fp_sfr = open(sfr_file, 'w')
    fmt = ' % .3e %.3f %.4f\n'
    for la, sf in zip(lages, sfr):
        if sf == 0:
            continue
        if len(la) == 0:
            la = np.min(age)
        else:
            la = np.max(la)
        fp_sfr.write(fmt % (10 ** np.array(la - 0.01), 0, metallicity))
        fp_sfr.write(fmt % (10 ** np.array(la), sf, metallicity))
        fp_sfr.write(fmt % (10 ** np.array(la + 0.01), 0, metallicity))
    fp_sfr.close()
    logger.info(' % s write %s' % (estimate_sfh.__name__, sfr_file))

    # save file
    if save_cmd:
        np.savetxt(sim_cmd_name, (sgal.color, sgal.mag2), delimiter=' ')
        logger.info(' % s write %s' % estimate_sfh.__name__, sim_cmd_name)

    return sfr_file


def color_color_diagnostic(trilegal_output_file, filter1, filter2, filter3,
                           filter4, ax=None, **pltkwargs):
    import time
    import fileIO
    logger.info('reading %s' % trilegal_output_file)
    tstart = time.time()
    t = fileIO.read_table(trilegal_output_file)
    tend = time.time()
    logger.info('reading took %.2f seconds' % (tend - tstart))
    #mag1 = np.array(t[filter1])
    #mag2 = np.array(t[filter2])
    #mag3 = np.array(t[filter3])
    #mag4 = np.array(t[filter4])

    mag1 = t.get_col(filter1)
    mag2 = t.get_col(filter2)
    mag3 = t.get_col(filter3)
    mag4 = t.get_col(filter4)

    if ax is None:
        ax = plt.axes()

    ax.plot(np.array(mag1) - np.array(mag2), np.array(mag3) - np.array(mag4),
            ',', label=trilegal_output_file, **pltkwargs)
    ax.set_ylabel('$%s - %s$' % (filter1, filter2))
    ax.set_xlabel('$%s - %s$' % (filter3, filter4))
    return ax


def mc_tests(ID, dir_name, outdir="MC_TESTS", bestfit_loc="MODELS/"):
    '''
    ID is the Galaxy name
    dir_name is PropID_GalaxyName
    Will put mc tests in MC_TESTS/*/dirname
    '''
    from subprocess import Popen, PIPE
    import fileIO
    # Make new folders
    folders = ['PARS/', 'INPUT/', 'OUTPUT/']
    outdir = fileIO.ensure_dir(outdir)
    dummy = [fileIO.ensure_dir(os.path.join(outdir, folder))
             for folder in folders]
    this_outdir = fileIO.ensure_dir(os.path.join(outdir, folder, dir_name))

    # A place to write the commands if this needs to be run again
    cmds_out = fileIO.ensure_dir(os.path.join(outdir, 'MC_COMMANDS/'))
    cmds_file = open('%s/%s_MC_commands.dat' % (cmds_out, ID), 'r')

    # Place for the output table
    table_out = fileIO.ensure_dir(os.path.join(outdir, 'TABLES/'))
    out = open('%s/%s_MC.dat' % (table_out, ID), 'w')
    out.write("# mc_ID p_value NRGB_data NAGB_data NRGB_model "
              "NAGB_model mass_model N_wind Flux1_wind Flux2_wind\n")

    # best fits:
    # bfs[0] = pars for run_trilegal,
    # bfs[1] = input for trilegal
    # bfs[2] = output for trilegal
    bfs = [get_afile(bestfit_loc + folder, '*' + ID + '*')[0]
           for folder in folders]

    # find model from file!!! This is not general!!
    model = bfs[-1].split('model_')[-1].replace('.dat.dat', '.dat')

    # Switch to the MC test directories
    new_place = [bf.replace(bestfit_loc, 'MC_TESTS/').replace(folder, folder +
                 dir_name + '/') for bf in bfs]

    # Best fit pars is the template, we'll swap out sfh file
    in_pars = open(bfs[0]).readlines()

    # Load SFHs
    sfhs = get_afile(mc_dir + '/', 'SFR*mc*dat')

    for i, sfh in enumerate(sfhs):
        mcid = sfh.split('/')[-1].split('.')[1]  # need to change jason / me
        new_names = [np.replace(ext, '.' + mcid + ext)
                     for np, ext in zip(new_place,
                                        ['.pars', '.dat', '.dat.dat'])]

        # New pars file for run_trilgal.py
        pfile = open(new_names[0], 'w')
        # sfh line is at the bottom.
        [pfile.write(inpar) for inpar in in_pars[:-1]]
        pfile.write("%-18s %s\n" % ('object_sfr  ', sfh))
        pfile.close()

        cmd = "/Users/Phil/research/Italy/WXTRILEGAL/run_trilegal.py "
        # cmd = "/home/philrose/WXTRILEGAL/run_trilegal.py "
        cmd += "-e code_2.0/main "
        cmd += "-a "
        cmd += "-i %s " % new_names[1]
        cmd += "-o %s " % new_names[2]
        cmd += "-f %s " % model
        cmd += new_names[0]

        cmds_file.write('%s \n' % cmd)
        print 'running TRILEGAL: ', model, ID
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        p = subprocess.Popen(cmd, shell=True)
        sts = os.waitpid(p.pid, 0)[1]

        fak_file = get_fakFILE(ID, jason=jason)
        tmp = new_names[2].split('/')[-1]
        ast_file = new_names[2].split('/')[0] + '/ast_' + tmp
        cmd = "AST/code/spread_angst <<EOF \n"
        cmd += fak_file + "\n"
        cmd += new_names[2] + "\n"
        cmd += ast_file + "\n"
        cmd += "EOF \n"
        print "  ... completeness using %s" % fak_file
        print "  %s -> %s" % (new_names[2], ast_file)
        print 'Running spread_angst...'
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        sts = os.waitpid(p.pid, 0)[1]
        os.system("wc  -l %s %s|head -2" % (new_names[2], ast_file))

        cmds_file.write('%s \n' % cmd)

        synthcmd = fileIO.read_table(ast_file)
        tmp2 = synthcmd.get_col('diff_mag2'.strip())
        tmp1 = synthcmd.get_col('diff_mag1'.strip())
        s_mag2 = synthcmd.get_col('mag2') + tmp2
        s_mag2 = s_mag2[np.nonzero(abs(tmp2) < 90.)[0]]
        s_mag1 = synthcmd.get_col('mag1') + tmp1
        s_mag1 = s_mag1[np.nonzero(abs(tmp1) < 90.)[0]]
        s_color = s_mag1 - s_mag2
        Norm = trgb + 1.5
        # this no longer works...
        #ind, nB_AGB, nNorm, ps_nNorm, ps_nB_AGBm, hist, bins, s_hist_normed,
        # p_value, normalization = calc_LF(mag2, s_mag2, Norm, trgb)
        #Nstars, flux_rates = flux_from_mass_loss(synthcmd, rates,
        #[filt1, filt2], ast_inds=ind, rel_flux=True)
        #out.write('# mc_ID p_value NRGB_data NAGB_data NRGB_model NAGB_model
        # mass_model N_wind Flux1_wind Flux2_wind\n')
        #out.write('%s  %.3f  %i  %i  %i  %i  %e  %i  %e  %e \n' %
        #           (mcid, p_value, nNorm, nB_AGB, ps_nNorm, ps_nB_AGBm,
        #            object_mass, Nstars[0], flux_rates[0][1],
        #            flux_rates[1][0]))

        os.remove(ast_file)
        print 'deleted', ast_file
        os.remove(new_names[2])
        print 'deleted', new_names[2]

    out.close()
    cmds_file.close()
