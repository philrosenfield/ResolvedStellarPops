import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep, splprep
import fileIO

class Track(object):
    def __init__(self, filename, ptcri=None, ptcri_file=None):
        (self.base, self.name) = os.path.split(filename)
        self.load_track(filename)
        self.filename_info()
        self.mass = self.data.MASS[0]            
        if ptcri is not None:
            self.load_critical_points(ptcri=ptcri)
        
    def filename_info(self):
        (pref, __, smass) = self.name.replace('.PMS','').split('_')
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

    def cut_long_track(self):
        '''
        cuts tracks at the first thermal pulse or c burning. Taken from 
        Bressan's save_isoc_set
        '''
        icburn = []
        ycen = self.data.YCEN
        if ycen[-1] == 0:
            lc = self.data.LC
            lneutr = self.data.LNEUTR
            icburn, = np.nonzero((ycen == 0) & ((lc > 2) | (lc > lneutr)))
        if len(icburn) > 0:
            # actually C-burning
            itpagb = min(icburn) 
        else:
            # beginning thin shell
            ishell, = np.nonzero((ycen == 0) & (self.data.QCAROX > self.data.QHEL*3./4.))
            if len(ishell) > 0:
                itpagb = np.min(ishell)
            else:
                itpagb = len(self.data) - 1
                ishell, = np.nonzero((self.data.LY > 1) & (self.data.QCAROX > 0.1))
                if len(ishell) > 0:
                    itpagb = np.min(ishell)
        self.data = self.data[np.arange(itpagb)]
        
    def load_track(self, filename, min_lage=0.2, cut_long=True):
        '''
        reads PMS file into a record array. Stores header as self.header
        '''
        with open(filename, 'r') as f:
            lines = f.readlines()

        begin_track, = [i for i,l in enumerate(lines) if 'BEGIN TRACK' in l]
        self.header = lines[:begin_track]
        col_keys = lines[begin_track + 1].strip().split()
        dtype = [(c, '<f8') for c in col_keys]
        try:
            data = np.genfromtxt(filename,
                                 skiprows=begin_track + 2,
                                 names=col_keys)
        except ValueError:
            data = np.genfromtxt(filename,
                                 skiprows=begin_track + 2,
                                 skip_footer = 2,
                                 names=col_keys)
        # cut non-physical part of the model
        ainds, = np.nonzero(data['AGE'] > min_lage)
        data = data[ainds]
                
        self.data = data.view(np.recarray)
        self.col_keys = col_keys
        if cut_long:
            self.cut_long_track()
        return

    def load_critical_points(self, filename=None, ptcri=None):
        '''
        iptcri is the critical point index rel to self.data
        mptcri is the model number of the critical point
        '''
        assert filename is not None or ptcri is not None, \
            'Must supply either a ptcri file or ptcri object'

        if ptcri is None:
            ptcri = critical_point(filename)
        self.mptcri = ptcri.data_dict['M%.3f' % self.mass]
        self.iptcri = np.concatenate([np.nonzero(self.data.MODE == m)[0] for m in self.mptcri])
        self.ptcri_keys = ptcri.key_dict
        assert ptcri.Z == self.Z, 'Zs do not match between track and ptcri file'
        assert ptcri.Y == self.Y, 'Ys do not match between track and ptcri file'

    def write_culled_track(self, columns_list, **kwargs):
        '''
        write out only some columns of the PMS file.
        '''
        filename = kwargs.get('filename')
        if filename is None:
            name_dat = '%s.dat' % self.name
            filename = os.path.join(self.base, name_dat)
        to_write = [np.column_stack(self.data[c]) for c in columns_list]
        to_write = np.squeeze(np.array(to_write).T)
        with open(filename, 'w') as f:
            f.write('# %s \n' % ' '.join(columns_list))
            np.savetxt(f, to_write, fmt='%.6f')
        return filename

    
    def tracks_for_match(self, outfile='default', ptcrifile=None, nticklist=None):        
        assert ptcrifile is not None or hasattr(self, 'iptcri'), \
            'Need either a critical point file or critical_point object'
        
        if outfile == 'default':
            outfile = os.path.join('%s' % self.base, 'match_%s.dat' % self.name.replace('.PMS',''))
            print 'writing %s' % outfile

        if ptcrifile is not None:
            self.load_critical_points(filename=ptcrifile)

        # add the final point of the track to the crit point array.
        self.iptcri = np.append(self.iptcri, len(self.data)-1)

        nknots = -1
        spline_level = 3
        smooth = 0.

        if nticklist is None:
            nticks = np.repeat(200, len(self.iptcri)-1)
        else:
            nticks = nticklist
        logTe = np.array([])
        logL = np.array([])
        Age = np.array([])
        for i in range(len(self.iptcri)-1):
            inds = np.arange(self.iptcri[i], self.iptcri[i+1])
            
            if self.mptcri[i+1] - self.mptcri[i] < 35:
                print 'skipping! m %s' % self.mass
                continue
            if len(inds) < 20:
                print 'skipping! i %s' % self.mass
                continue
            
            tckp, u = splprep([self.data.LOG_TE[inds],
                               self.data.LOG_L[inds],
                               self.data.AGE[inds]],
                               s=smooth, k=spline_level,
                               nest=nknots)

            tenew, lnew, agenew = splev(np.linspace(0, 1, nticks[i]), tckp)
            #plt.plot(tenew,lnew, '.', color='red')
            logTe = np.append(logTe, tenew)
            logL = np.append(logL, lnew)
            Age = np.append(Age, agenew)
        
        Mbol = 4.77 - 2.5 * logL
        logg = -10.616 + np.log10(self.mass) + 4.0 * logTe - logL
        logAge = np.log10(Age)
        # place holder!
        CO = np.zeros(len(logL))
        mass_arr = np.repeat(self.mass, len(logL))
        
        header = '# logAge Mass logTe Mbol logg C/O \n'
        to_write = np.column_stack((logAge, mass_arr, logTe, Mbol, logg, CO))
        with open(outfile, 'w') as f:
            f.write(header)
            np.savetxt(f, to_write, fmt='%.6f')

        self.match_data = to_write
        
    def plot_track(self, xcol, ycol, reverse_x=False, reverse_y=False, ax=None, inds=None, plt_kw={}):
        if ax is None:
            fig = plt.figure()
            ax = plt.axes()
        
        if inds is not None:
            ax.plot(self.data[xcol][inds], self.data[ycol][inds], **plt_kw)
        else:
            ax.plot(self.data[xcol], self.data[ycol], **plt_kw)

        if reverse_x:
            ax.set_xlim(ax.get_xlim()[::-1])
        
        if reverse_y:
            ax.set_ylim(ax.get_ylim()[::-1])

        return ax
    
    def check_ptcris(self):
        ax = self.plot_track('LOG_TE', 'LOG_L', reverse_x=True, plt_kw={'color': 'black'})
        ax = self.plot_track('LOG_TE', 'LOG_L', ax=ax, inds=self.iptcri, plt_kw={'marker': 'o', 'ls': ''})
        if hasattr(self, 'match_data'):
            logl = (4.77 - self.match_data.T[3]) / 2.5
            ax.plot(self.match_data.T[2], logl, lw=2)
        ax.set_title('M = %.3f' % self.mass)
        plt.savefig('ptcri_Z%g_Y%g_M%.3f.png' % (self.Z, self.Y, self.mass))

class TrackDiag(object):
    def __init__(self, track):
        pass
    
    #finished phases
    #very low mass or not        
        
class critical_point(object):
    def __init__(self, filename):
        self.load_ptcri(filename)
        self.base, self.name = os.path.split(filename)
        self.get_args_from_name(filename)
        
    def get_args_from_name(self, filename):
        zstr = filename.split('_Z')[-1]
        self.Z = float(zstr[:5].split('_')[0])
        ystr = filename.split('_Y')[-1][:5].split('_')[0]
        if ystr.endswith('.'):
            ystr = ystr[:-1]
        self.Y = float(ystr)

    def load_ptcri(self, filename):
        '''
        reads the ptcri*dat file.
        '''
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        begin, = [i for i in range(len(lines)) if lines[i].startswith('#') and 'F7' in lines[i]]
        # the final column is a filename.
        col_keys = lines[begin + 1].replace('#', '').strip().split()[3:-1]
        
        data = np.genfromtxt(filename, usecols=range(0, 18),
                             skip_header=begin + 2)
        self.data = data
        self.masses = data[:, 1]
        data_dict = {}
        for i in range(len(data)):
            str_mass = 'M%.3f' % self.masses[i]
            data_dict[str_mass] = data[i][3:].astype('int')

        self.data_dict = data_dict
        self.key_dict = dict(zip(col_keys, range(len(col_keys))))
        
        
'''
trackname = '/Users/phil/Desktop/S11_set/F7/CAF09/S11_Z0.02_Y0.284/Z0.02Y0.284OUTA1.74_F7_M5.00.PMS'
ptcrifile = '/Users/phil/Desktop/S11_set/data/ptcri_CAF09_S11_Z0.02_Y0.284.dat'
t = ResolvedStellarPops.PadovaTracks.Track(trackname) 
t.load_critical_points(ptcrifile)
#ptcri = ResolvedStellarPops.PadovaTracks.critical_point(ptcrifile)
'''

tracks_dir = '/Users/phil/Desktop/S11_set/F7/CAF09'
ptcrifile_loc = '/Users/phil/Desktop/S11_set/data'
prefix = 'S11_Z0.02_Y0.284'

def track_set_for_match(prefix, **kwargs):
    tracks_dir = '/Users/phil/Desktop/S11_set/F7/CAF09'
    ptcrifile_loc = '/Users/phil/Desktop/S11_set/data'
    
    track_names = fileIO.get_files(os.path.join(tracks_dir, prefix), '*PMS')
    ptcri_file, = fileIO.get_files(os.path.join(ptcrifile_loc), '*%s*dat' % prefix)

    ptcri = critical_point(ptcri_file)

    for track in track_names:
        if 'M.6' in track or 'M.7' in track:
            continue
        t = Track(track, ptcri=ptcri)
        t.check_ptcris()
        t.tracks_for_match()

if __name__ == '__main__':
    import pdb
    pdb.set_trace()
    track_set_for_match(prefix)