'''
padova track files
'''
from __future__ import print_function
import os
import numpy as np
import fileIO
from eep.critical_point import critical_point
from eep.critical_point import eep
import matplotlib.pylab as plt
from graphics.GraphicsUtils import discrete_colors


class Track(object):
    '''
    Padova stellar evolutoin track object.
    '''
    def __init__(self, filename, ptcri=None, min_lage=0.1, cut_long=False):
        (self.base, self.name) = os.path.split(filename)
        self.load_track(filename, min_lage=min_lage, cut_long=cut_long)
        self.filename_info()
        self.mass = self.data.MASS[0]
        if self.mass >= 12:
            # for high mass tracks, the mass starts much larger than it is
            # for (age<0.2). The mass only correct at the beginning of the MS.
            # Rather than forcing a ptcri load, we read the mass from the title.
            self.mass = float(self.name.split('_M')[1].split('.PMS')[0])
        self.ptcri = ptcri
        test = np.diff(self.data.AGE) >= 0
        if False in test:
            print('Track has age decreasing!!', self.mass)
            bads, = np.nonzero(np.diff(self.data.AGE) < 0)
            print('offensive MODEs:', self.data.MODE[bads])

    def calc_Mbol(self, z_sun=4.77):
        '''
        Uses Z_sun = 4.77 adds self.Mbol and returns Mbol
        '''
        self.Mbol = z_sun - 2.5 * self.data.LOG_L
        return self.Mbol

    def calc_logg(self):
        '''
        cgs constant is -10.616 adds self.logg and returns logg
        '''
        self.logg = -10.616 + np.log10(self.mass) + 4.0 * self.data.LOG_TE - \
            self.data.LOG_L
        return self.logg

    def calc_core_mu(self):
        '''
        Uses X, Y, C, and O.
        '''
        xi = np.array(['XCEN', 'YCEN', 'XC_cen', 'XO_cen'])
        ai = np.array([1., 4., 12., 16.])
        # fully ionized
        qi = ai/2.
        self.MUc = 1. / (np.sum((self.data[xi[i]] / ai[i]) * (1 + qi[i])
                                 for i in range(len(xi))))
        return self.Muc

    def filename_info(self):
        '''
        I wish I knew regex...
        '''
        (pref, __, smass) = self.name.split('.PMS')[0].split('_')
        #self.__setattr__[]
        #get that into attrs: 'Z0.0002Y0.4OUTA1.74M2.30'
        Z, Ymore = self.name.split('Z')[1].split('Y')
        for i, y in enumerate(Ymore):
            if y == '.':
                continue
            try:
                float(y)
            except:
                break
        self.Z = float(Z)
        self.Y = float(Ymore[:i])
        return

    def load_track(self, filename, min_lage=0.1, cut_long=False):
        '''
        reads PMS file into a record array. Stores header as string self.header
        '''
        with open(filename, 'r') as f:
            lines = f.readlines()

        begin_track, = [i for i, l in enumerate(lines) if 'BEGIN TRACK' in l]
        self.header = lines[:begin_track]
        col_keys = lines[begin_track + 1].replace('#', '').strip().split()
        begin_track_skip = 2
        skip_footer = len([i for i, l in enumerate(lines)
                           if 'Comp Time' in l or 'AGE EXCEEDS AGELIMIT' in l])
        # Hack to read tracks that have been "colored"
        if 'information' in lines[begin_track + 2]:
            col_keys = self.add_to_col_keys(col_keys, lines[begin_track + 2])
            begin_track_skip += 1

        data = np.genfromtxt(filename, skiprows=begin_track + begin_track_skip,
                             names=col_keys, skip_footer=skip_footer,
                             invalid_raise=False)

        self.data = data.view(np.recarray)
        self.col_keys = col_keys

        return

    def add_to_col_keys(self, col_keys, additional_col_line):
        '''
        If fromHR2mags was run, A new line "Additional information Added:..."
        is added, this adds these column keys to the list.
        '''
        new_cols = additional_col_line.split(':')[1].strip().split()
        col_keys = list(np.concatenate((col_keys, new_cols)))
        return col_keys

    def format_track_file(self):
        '''
        add # comment to header save new file with .dat extension.
        Useful to prepare for fromHR2mags.
        '''
        self.header = ['# %s' % h for h in self.header]

        name_dat = '%s.dat' % self.name
        filename = os.path.join(self.base, name_dat)
        with open(filename, 'w') as f:
            f.writelines(self.header)
            f.write('# %s \n' % ' '.join(self.col_keys))
            np.savetxt(f, self.data, dtype=self.data.dtype)
        return filename


class TrackSet(object):
    '''

    '''
    def __init__(self, inputs):

        if inputs.ptcrifile_loc is not None or inputs.ptcri_file is not None:
            self.load_ptcri_eep(inputs)
        else:
            self.ptcri = None

        self.tracks_base = os.path.join(inputs.tracks_dir, inputs.prefix)
        if inputs.hb_only is False:
            self.load_tracks(track_search_term=inputs.track_search_term,
                             masses=inputs.masses)
        if inputs.hb is True:
            self.load_tracks(track_search_term=inputs.hbtrack_search_term,
                             hb=inputs.hb,
                             masses=inputs.masses)

    def load_ptcri_eep(self, inputs):
        '''
        load the ptcri and eeps, simple call to the objects.
        way isn't this in eep?
        '''
        self.ptcri = None
        self.eep = None
        if hasattr(inputs, 'ptcri_file'):
            self.ptcri_file = inputs.ptcri_file
        else:
            self.prefix = inputs.prefix
            if inputs.from_p2m is True:
                # this is the equivalent of Sandro's ptcri files, but mine.
                search_term = 'p2m*%s*dat' % self.prefix
                self.ptcri_file, = fileIO.get_files(inputs.ptcrifile_loc,
                                                    search_term)
                print('reading ptcri from saved p2m file.')
            else:
                search_term = 'pt*%s*dat' % self.prefix
                self.ptcri_file, = fileIO.get_files(inputs.ptcrifile_loc,
                                                    search_term)

        if inputs.eep_list is not None:
            eep_kw = {'eep_lengths': inputs.eep_lengths,
                      'eep_list_hb': inputs.eep_list_hb,
                      'eep_lengths_hb': inputs.eep_lengths_hb}
            self.eep = eep(inputs.eep_list, **eep_kw)

        self.ptcri = critical_point(self.ptcri_file, eep_obj=self.eep)

    def load_tracks(self, track_search_term='*F7_*PMS', hb=False, masses=None):
        '''
        loads tracks or hb tracks, can load subset if masses (list, float, or
        string) is set. If masses is string, it must be have format like:
        '%f < 40' and it will use masses that are less 40.
        '''
        track_names = np.array(fileIO.get_files(self.tracks_base,
                               track_search_term))
        assert len(track_names) != 0, \
            'No tracks found: %s/%s' % (self.tracks_base, track_search_term)
        mass = np.array([os.path.split(t)[1].split('_M')[1].split('.P')[0]
                         for t in track_names], dtype=float)
        track_names = track_names[np.argsort(mass)]
        mass = mass[np.argsort(mass)]

        # only do a subset of masses
        if masses is not None:
            if type(masses) == float:
                masses = [masses]
            elif type(masses) == str:
                track_masses = [i for i in range(len(mass))
                                if eval(masses % mass[i])]
            if type(masses) == list:
                track_masses = []
                for set_mass in masses:
                    try:
                        track_masses.append(list(mass).index(set_mass))
                    except ValueError:
                        pass
                track_masses = np.array(track_masses)
        else:
            track_masses = np.argsort(mass)


        track_str = 'track'
        mass_str = 'masses'
        if hb is True:
            track_str = 'hb%s' % track_str
            mass_str = 'hb%s' % mass_str
        self.__setattr__('%s_names' % track_str, track_names[track_masses])

        self.__setattr__('%ss' % track_str, [Track(track, ptcri=self.ptcri,
                                                   min_lage=0., cut_long=0)
                                             for track in track_names[track_masses]])

        self.__setattr__('%s' % mass_str,
                         np.round([t.mass for t in
                                   self.__getattribute__('%ss' % track_str)], 3))

    def save_ptcri(self, filename=None, hb=False):
        #assert hasattr(self, ptcri), 'need to have ptcri objects loaded'
        if hb is True:
            tracks = self.hbtracks
        else:
            tracks = self.tracks

        if filename is None:
            base, name = os.path.split(self.ptcri_file)
            filename = os.path.join(base, 'p2m_%s' % name)
            if hb is True:
                filename = filename.replace('p2m', 'p2m_hb')

        sorted_keys, inds = zip(*sorted(self.ptcri.key_dict.items(),
                                        key=lambda (k, v): (v, k)))

        header = '# critical points in F7 files defined by sandro, basti, and phil \n'
        header += '# i mass lixo %s fname \n' % (' '.join(sorted_keys))
        with open(filename, 'w') as f:
            f.write(header)
            linefmt = '%2i %.3f 0.0 %s %s \n'
            for i, track in enumerate(tracks):
                self.ptcri.please_define = []
                # this line should just slow everything down, why is it here?
                self.load_critical_points(track, eep_obj=self.eep, hb=hb,
                                          ptcri=self.ptcri, diag_plot=False)
                ptcri_str = ' '.join(['%5d' % p for p in track.iptcri])
                f.write(linefmt % (i+1, track.mass, ptcri_str,
                                   os.path.join(track.base, track.name)))

        print('wrote %s' % filename)

    def plot_all_tracks(self, tracks, xcol, ycol, annotate=True, ax=None,
                        reverse_x=False, sandro=True, cmd=False,
                        convert_mag_kw={}, hb=False, plot_dir=None,
                        zoomin=True, one_plot=False):
        '''
        It would be much easier to discern breaks in the sequences if you did
        three separate plots: PMS_BEG to MS_BEG,
        MS_BEG to RG_TIP, and RG_TIP to TPAGB.
        As it stands, one is trying to tell RGB features from AGB features.
        Likewise, there is such a small color difference between some of the
        different points that I'm not entire sure what I'm seeing.

        I see a sharp break in the RGB bump and RGB tip sequences.
        Are those visible in the isochrones?
        '''
        line_pltkw = {'color': 'black', 'alpha': 0.1}

        if one_plot is True:
            for t in tracks:
                import TrackDiag
                td = TrackDiag()
                all_inds, = np.nonzero(t.data.AGE > 0.2)
                ax = td.plot_track(t, xcol, ycol, ax=ax, inds=all_inds,
                                     plt_kw=line_pltkw, cmd=cmd,
                                     convert_mag_kw=convert_mag_kw, hb=hb)
            return ax

        ptcri_kw = {'sandro': sandro, 'hb': hb}

        if hb is False:
            if sandro is True:
                plots = [['PMS_BEG', 'PMS_MIN', 'PMS_END', 'NEAR_ZAM', 'MS_BEG'],
                         ['MS_BEG', 'POINT_B', 'POINT_C'],
                         ['POINT_C', 'RG_BASE', 'RG_BMP1', 'RG_BMP2', 'RG_TIP'],
                         ['Loop_A', 'Loop_B', 'Loop_C', 'TPAGB']]
                fig_extra = ['pms', 'ms', 'rg', 'loop']
            else:
                plots = [['PMS_BEG', 'PMS_MIN', 'PMS_END', 'MS_BEG'],
                         ['MS_BEG', 'MS_TMIN', 'MS_TO', 'SG_MAXL', 'RG_MINL',
                          'RG_BMP1', 'RG_BMP2', 'RG_TIP'],
                         ['RG_TIP', 'HE_BEG', 'YCEN_0.550', 'YCEN_0.500'],
                         ['YCEN_0.550', 'YCEN_0.500', 'YCEN_0.400', 'YCEN_0.200',
                          'YCEN_0.100', 'YCEN_0.000', 'TPAGB']]
                fig_extra = ['pms', 'ms', 'rg', 'ycen']
        else:
            plots = [['HB_BEG', 'YCEN_0.500', 'YCEN_0.400', 'YCEN_0.200',
                     'YCEN_0.100', 'YCEN_0.005', 'AGB_LY1', 'AGB_LY2']]
            # overwriting kwargs!!
            ptcri_kw['sandro'] = False
            fig_extra = ['hb']

        assert len(fig_extra) == len(plots), \
            'need correct plot name extensions.'

        xlims = np.array([])
        ylims = np.array([])
        for j in range(len(plots)):
            fig, ax = plt.subplots()
            if annotate is True:
                point_pltkw = {'marker': '.', 'ls': '', 'alpha': 0.5}
                cols = discrete_colors(len(plots[j]), colormap='spectral')
                labs = ['$%s$' % p.replace('_', '\_') for p in plots[j]]

            didit = 0
            xlimi = np.array([])
            ylimi = np.array([])
            for t in tracks:
                if sandro is False:
                    ptcri = t.iptcri
                else:
                    ptcri = t.sptcri
                ainds = [t.ptcri.get_ptcri_name(cp, **ptcri_kw)
                         for cp in plots[j]]
                ainds = [i for i in ainds if i < len(ptcri)]

                inds = ptcri[ainds]


                if np.sum(inds) == 0:
                    continue

                some_inds = np.arange(inds[0], inds[-1])

                ax = self.plot_track(t, xcol, ycol, ax=ax, inds=some_inds,
                                     plt_kw=line_pltkw, cmd=cmd, clean=False,
                                     convert_mag_kw=convert_mag_kw)

                #line_pltkw['alpha'] = 1.
                #ax = self.plot_track(t, xcol, ycol, ax=ax, inds=some_inds,
                #                     plt_kw=line_pltkw, cmd=cmd,
                #                     convert_mag_kw=convert_mag_kw)

                xlims = np.append(xlims, np.array(ax.get_xlim()))
                ylims = np.append(ylims, np.array(ax.get_ylim()))

                if annotate is True:
                    xdata = t.data[xcol]
                    ydata = t.data[ycol]

                    if cmd is True:
                        xdata = t.data[xcol] - t.data[ycol]
                    pls = []
                    for i in range(len(inds)):
                        x = xdata[inds[i]]
                        y = ydata[inds[i]]
                        xlimi = np.append(xlimi, (np.min(x), np.max(x)))
                        ylimi = np.append(ylimi, (np.min(y), np.max(y)))
                        pl, = ax.plot(x, y, color=cols[i], **point_pltkw)
                        pls.append(pl)
                    ax.text(xdata[inds[0]], ydata[inds[0]], '%.3f' % t.mass,
                            fontsize=8, ha='right')
                    # only save the legend if all the points are made
                    if len(inds) == len(plots[j]):
                        didit += 1
                        if didit == 1:
                            plines = pls

            if zoomin is True:
                ax.set_xlim(np.min(xlimi), np.max(xlimi))
                ax.set_ylim(np.min(ylimi), np.max(ylimi))
            else:
                ax.set_xlim(np.min(xlims), np.max(xlims))

            if reverse_x is True:
                ax.set_xlim(ax.get_xlim()[::-1])

            if annotate is True:
                ax.legend(plines, labs, numpoints=1, loc=0, frameon=False)

            ylab = ycol.replace('_', '\ ')
            xlab = xcol.replace('_', '\ ')
            figname = '%s_%s_%s_%s.png' % (self.prefix, xcol, ycol,
                                           fig_extra[j])

            if cmd is True:
                xlab = '%s-%s' % (xlab, ylab)

            ax.set_xlabel('$%s$' % xlab)
            ax.set_ylabel('$%s$' % ylab)

            if plot_dir is not None:
                figname = os.path.join(plot_dir, figname)
            plt.savefig(figname)
            print('wrote %s' % figname)
            #plt.close()
        return

    def squish(self, *attrs, **kwargs):
        '''
        bad coder: I took this from Galaxies.galaxy. Some day I can make
        it all great, but first I need to have the same data fmts for all these
        things...

        concatenates an attribute or many attributes and adds them to galaxies
        instance -- with an 's' at the end to pluralize them... that might
        be stupid.
        ex
        for gal in gals:
            gal.ra = gal.data['ra']
            gal.dec = gal.data['dec']
        gs =  Galaxies.galaxies(gals)
        gs.squish('color', 'mag2', 'ra', 'dec')
        gs.ras ...

        kwargs: inds choose which tracks to include (all by default)
        new_attrs: if you don't like the attributes set.
        '''
        inds = kwargs.get('inds', np.arange(len(self.tracks)))
        new_attrs = kwargs.get('new_attrs', None)

        if new_attrs is not None:
            assert len(new_attrs) == len(attrs), \
                'new attribute titles must be list same length as given attributes.'

        for i, attr in enumerate(attrs):
            # do we have a name for the new attribute?
            if new_attrs is not None:
                new_attr = new_attrs[i]
            else:
                new_attr = '%ss' % attr

            new_list = [self.tracks[j].data[attr] for j in inds]
            # is attr an array of arrays, or is it now an array?
            try:
                new_val = np.concatenate(new_list)
            except ValueError:
                new_val = np.array(new_list)

            self.__setattr__(new_attr, new_val)

    def all_inds_of_eep(self, eep_name):
        '''
        get all the ind for all tracks of some eep name, for example
        want ms_to of the track set? set eep_name = point_c if sandro==True.
        '''
        inds = []
        for track in self.tracks:
            check = track.ptcri.load_sandro_eeps(track)
            if check == -1:
                inds.append(-1)
                continue
            eep_ind = track.ptcri.get_ptcri_name(eep_name)
            if len(track.sptcri) <= eep_ind:
                inds.append(-1)
                continue
            data_ind = track.sptcri[eep_ind]
            inds.append(data_ind)
        return inds
