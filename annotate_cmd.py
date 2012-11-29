from matplotlib.pyplot import ginput
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.nxutils as nxutils
import numpy as np
import os
import pdb
import scipy.interpolate

import Galaxies
from TrilegalUtils import get_stage_label
from scatter_contour import scatter_contour

import logging
logger = logging.getLogger()
if logger.name == 'root':
    rsp.fileIO.setup_logging()

def poly_fmt(polygon_str):
    polygon_str = polygon_str.replace(')', '').replace('(', '').strip()
    polygon_list = map(float, polygon_str.split(','))
    return np.column_stack((polygon_list[0::2], polygon_list[1::2]))


class CMDregion(object):
    '''
    class for messing with the region files that accompany the
    tagged data. These files are made in define_color_mag_region
    '''
    def __init__(self, region_file):
        self.__initialised = True
        self.base, self.name = os.path.split(region_file)
        CMDregion._load_data(self, region_file)
        self.region_names = self.regions.keys()

    def _load_data(self, region_file):
        self.regions = {}
        self.region_names = []
        with open(region_file) as f:
            for i, line in enumerate(f):
                if i == 0:
                    col_keys = line[i].strip().replace('#', '').split()
                else:
                    pid, reg, filt1, filt2 = line.split('polygon')[0].split()
                    reg = reg.lower()
                    polygon = line.split('polygon')[1].strip()
                    if not reg in self.regions.keys():
                        self.regions[reg] = {}
                    self.regions[reg] = poly_fmt(polygon)
            
            self.filter1 = filt1
            self.filter2 = filt2
            lixo, self.survey, __, camera, pidtarget = pid.split('_')
            self.target = '-'.join(pidtarget.split('-')[1:])
            self.photsys = camera.replace('-', '_')

    def shift_regions(self, shift_verts, regs, dcol=0., dmag=0.):
        '''
        shift part of a region (defined by N,2 array of vertices: shift_verts)
        adds attribute regions['%s_dc%.2f_dm%.2f'%(reg,dcol,dmag)] N,2 array
        '''
        for reg in regs:
            reg = reg.lower()
            polygon = self.regions[reg]
            new_verts = polygon.copy()
            # region to shift
            mask = nxutils.points_inside_poly(polygon, shift_verts)
            inds_to_shift, = np.nonzero(mask)
            # shift
            color = polygon[:, 0][inds_to_shift] + dcol
            mag = polygon[:, 1][inds_to_shift] + dmag
            shifted = np.column_stack((color, mag))
            # new region
            new_verts[inds_to_shift] = shifted
            self.regions['%s_dc%.2f_dm%.2f' % (reg, dcol, dmag)] = new_verts

    def join_regions(self, *regs):
        '''
        join two polygons into one, call by self.regions[reg]
        '''
        comb_region = np.concatenate([self.regions[r.lower()] for r in regs])
        new_verts = get_cmd_shape(comb_region[:, 0], comb_region[:, 1])
        self.regions['_'.join(regs).lower()] = new_verts

    def average_color(self, *regs):
        '''
        ave color of a region, is this stupid?
        I fill a box of npts randomly and cut out those not in the region.
        Then I take the average in bins of mag.min,mag.max dm=binsize...
        probably.
        new attribute(s):
            self.regions['%s_mean'%reg]: (N,2) array of color, mag.
        '''
        for reg in regs:
            reg = reg.lower()
            polygon = self.regions[reg]
            color = polygon[:, 0]
            mag = polygon[:, 1]

            # get a ton of random points inside the polygon
            x = random_array(color)
            y = random_array(mag)
            xy = np.column_stack((x, y))
            inds = nxutils.points_inside_poly(xy, polygon).nonzero()
            rand_mag = y[inds]
            rand_col = x[inds]

            # take the average of the points in each magbin
            magbins = np.linspace(mag.min(), mag.max(), 5)
            dinds = np.digitize(rand_mag, magbins)
            mean_colors = [np.mean(rand_col[dinds == i])
                           for i in range(magbins.size)]

            # get rid of nans
            shit, = np.isnan(mean_colors).nonzero()
            mean_colors = np.delete(np.asarray(mean_colors), shit)
            magbins = np.delete(magbins, shit)

            self.regions['%s_mean' % reg] = np.column_stack((mean_colors,
                                                             magbins))
        return

    def split_regions(self, *regs, **kwargs):
        '''
        splits a joined region on the average color - or by a spliced arr.
        adds attribues self.regions[name1] and self.regions[name2]: N,2 arrays.
        '''
        name1 = kwargs.get('name1', ' rheb_bym')
        name2 = kwargs.get('name2', 'bheb_bym')
        on_mean = kwargs.get('on_mean', True)
        on_mag = kwargs.get('on_mag', True)

        for reg in regs:
            reg = reg.lower()
            if on_mean:
                # make sure we have average colors.
                if not '%s_mean' % reg in self.regions.keys():
                    CMDregion.average_color(self, reg)
                    mean_split = self.regions['%s_mean' % reg]
            else:
                mean_split = kwargs.get('splice_arr')

            polygon = self.regions[reg]
            polygon = uniquify_reg(polygon)
            # insert extreme points and sort them
            i = 0
            if on_mag:
                i = 1

            maxpoint = mean_split[np.argmax(mean_split[:, i])]
            minpoint = mean_split[np.argmin(mean_split[:, i])]
            polygoni = polar_sort(insert_points(polygon, maxpoint, minpoint))

            # split the array by the inserted points
            imaxs = np.where(polygoni == maxpoint)[0]
            imax, = not_unique(imaxs)

            imins = np.where(polygoni == minpoint)[0]
            imin, = not_unique(imins)

            if on_mag:
                if imin < imax:
                    sideA = polygoni[imin:imax]
                    sideB = np.vstack((polygoni[imax:], polygoni[:imin]))

                if imin > imax:
                    sideA = polygoni[:imax]
                    sideB = polygoni[imax:imin]
            else:
                sideA = np.vstack((polygoni[imin:], polygoni[:imax]))
                sideB = polygoni[imax:imin]

            # attach the mean values
            stitchedA = np.vstack((sideA, mean_split[::-1]))
            stitchedB = np.vstack((sideB, mean_split))

            self.regions[name1] = stitchedA
            self.regions[name2] = stitchedB
        return

    def add_points(self, *regs, **kwargs):
        '''
        add points to each line segment, creates new attribute reg_HD.
        '''
        npts = kwargs.get('npts', 10)
        for reg in regs:
            polygon = self.regions[reg]
            color = np.array([])
            mag = np.array([])
            polygon = uniquify_reg(polygon)
            for (c1, m1), (c2, m2) in zip(polygon,
                                          np.roll(polygon, -1, axis=0)):
                # 1d fit
                z = np.polyfit((c1, c2), (m1, m2), 1)
                p = np.poly1d(z)
                # new array of npts
                x = np.linspace(c1, c2, npts)
                y = p(x)
                # plt.plot(x,y,'o')
                color = np.append(color, x)
                mag = np.append(mag, y)
            self.regions['%s_hd' % reg] = np.column_stack((color, mag))
    
    def __setattr__(self, item, value):
        if not self.__dict__.has_key('__initialised'):  # this test allows attributes to be set in the __init__ method
            return dict.__setattr__(self, item, value)
        elif self.__dict__.has_key(item):       # any normal attributes are handled normally
            dict.__setattr__(self, item, value)
            self.region_names = self.regions.keys()
        else:
            self.__setitem__(item, value)
            self.region_names = self.regions.keys()
        
def uniquify_reg(reg):
    lixo, inds = np.unique(reg[:, 0], return_index=True)
    # unique will sort the array...
    inds = np.sort(inds)
    return reg[inds]


def random_array(arr, offset=0.1, npts=1e4):
    '''
    returns a random array of npts within the extremes of arr +/- offset
    '''
    return np.random.uniform(arr.min() - offset, arr.max() + offset, npts)


def insert_points(arr, *points):
    '''
    adds rows to array.
    '''
    arrcopy = np.vstack((arr, points))
    return arrcopy


def closest_point(point, verts):
    '''
    fortran days... smallest radius from point.
    '''
    return np.argmin(np.sqrt((verts[:, 0] - point[0]) ** 2 +
                    (verts[:, 1] - point[1]) ** 2))


def not_unique(arr):
    '''
    returns array of values that are repeated.
    '''
    arrcopy = arr.copy()
    uarr = np.unique(arrcopy)
    for u in uarr:
        arrcopy = np.delete(arrcopy, list(arrcopy).index(u))
    return arrcopy


def get_bounds(color, mag):
    '''
    find cmd bounding box, don't need it...
    '''
    min_color = np.min(color)
    max_color = np.max(color)
    bright_mag = np.min(mag)
    faint_mag = np.max(mag)
    bl = (min_color, faint_mag)
    br = (max_color, faint_mag)
    ul = (min_color, bright_mag)
    ur = (max_color, bright_mag)
    verts = np.vstack((ul, ur, br, bl, ul))
    return verts


def polar_sort(l):
    '''
    polar sorting, you know, in a circle like. I got it from some website.
    '''
    import math
    mean_col = np.mean([l[i][0] for i in range(len(l))])
    mean_mag = np.mean([l[i][1] for i in range(len(l))])
    slist = sorted(l, key=lambda c: math.atan2(c[0] - mean_col,
                   c[1] - mean_mag), reverse=True)
    return np.array(slist).squeeze()


def get_cmd_shape(color, mag):
    '''
    gets the outline of a cmd. Guesses at a large polygon, and then add points
    that are outside of the polygon, ignores points within.

    then polar sorts the result.
    returns: N,2 array.
    '''
    # make a guess at the polygon.
    left = (np.min(color), mag[np.argmin(color)])
    right = (np.max(color), mag[np.argmax(color)])
    up = (color[np.argmin(mag)], np.min(mag))
    down = (color[np.argmax(mag)], np.max(mag))
    verts = np.vstack((left, right, up, down))

    points = np.column_stack((color, mag))
    for point in points:
        if nxutils.pnpoly(point[0], point[1], verts) == 0.:
            # add point to verts
            col = verts[:, 0]
            m = verts[:, 1]
            col = np.append(col, point[0])
            m = np.append(m, point[1])
            # order the new points in a circle
            verts = polar_sort(zip(col, m))

    verts = np.append(verts, [verts[0]], axis=0)
    # plt.plot(verts[:, 0], verts[:, 1], lw = 2)
    return verts


def define_color_mag_region(fitsfiles, region_names, **kwargs):
    '''
    Define a polygon on a cmd of a fitsfile or a list of fitsfiles.
    the tagged fitsfile can then be used in match_light, or all other b/r
    modules that make use of MS vs BHeB vs RHeB vs RGB (which isn't picked
    out well)

    writes to two text files:
    1) for each fits file,
        fitsfile.dat: with ra dec mag1 mag2 mag1err mag2err stage
       stage is set in the code to match trilegal's parametri.h, will be -1
       if not assigned
    2) kwarg 'region file' [cmd_regions.dat] with
       propid_target region filter1 filter2 polygon

    also saves annotated cmd
    locations:
        all in base BRFOLDER/data/cmd_regions
        tagged fits file: base/tagged_photometery/[fitsfile].dat
        plot: base/plots/[fitsfile].png
        region file: base/regions/cmd_regions_[fitsfile].dat

    Draw points for each region with mouse click,
        ctrl click to delete,
        command+click to exit (edit: I think it's actually alt)
    '''
    cmd_regions_loc = kwargs.get('cmd_regions_loc')
    if not cmd_regions_loc:
        from BRparams import *
        cmd_regions_loc = os.path.join(BRFOLDER, 'data', 'cmd_regions')
    tagged_data_loc = os.path.join(cmd_regions_loc, 'tagged_photometry')
    plot_dir = os.path.join(cmd_regions_loc, 'plots')

    xlim = kwargs.get('xlim', (-1, 3))
    ylim = kwargs.get('ylim')

    if type(fitsfiles) == str:
        fitsfiles = [fitsfiles]
    for j, fitsfile in enumerate(fitsfiles):
        fileIO.ensure_file(fitsfile)
        datafile = os.path.join(tagged_data_loc,
                                fileIO.replace_ext(fitsfile, '.dat'))
        if os.path.isfile(datafile):
            logger.warning('%s exists, skipping' % datafile)
            continue

        logger.debug('%s of %s fitsfiles' % (j + 1, len(fitsfiles)))
        logger.debug('now working on %s' % fitsfile)

        # begin outfiles
        outname = os.path.split(fitsfile)[1].replace('.fits', '.dat')
        outfile = os.path.join(cmd_regions_loc, 'regions',
                               'cmd_regions_%s' % outname)
        out = open(outfile, 'a')
        header = '# propid_target region filter1 filter2 polygon\n'
        out.write(header)

        dout = open(datafile, 'w')
        dheader = '# ra dec mag1 mag2 mag1err mag2err stage\n'
        dout.write(dheader)

        fits = Galaxies.galaxy(fitsfile, fitstable=1)
        photsys = fits.photsys
        extrastr = '_%s' % photsys.upper()[0]

        ra = fits.data['ra']
        dec = fits.data['dec']
        mag1err = fits.data['mag1_err']
        mag2err = fits.data['mag2_err']
        mag1 = fits.mag1
        mag2 = fits.mag2
        filter1 = fits.filter1
        filter2 = fits.filter2

        errlimit, = np.nonzero(np.sqrt(mag1err ** 2 + mag2err ** 2) < 0.5)
        stage = np.zeros(shape=(len(mag1), )) - 1

        color = mag1 - mag2
        datapts = np.column_stack((color, mag2))

        #z,dx,dy = GenUtils.bin_up(color,mag2,dx=0.1,dy=0.1)

        fig = plt.figure(figsize=(20, 12))
        ax = plt.axes()
        ax.set_xlim(-1, 4)
        ax.set_ylim(mag2[errlimit].max(), mag2[errlimit].min())

        ncolbin = int(np.diff((np.min(color), np.max(color))) / 0.05)
        nmagbin = int(np.diff((np.min(mag2), np.max(mag2))) / 0.05)

        ax.set_xlabel('$%s-%s%s$' % (filter1, filter2, extrastr), fontsize=20)
        ax.set_ylabel('$%s%s$' % (filter2, extrastr), fontsize=20)
        ax.annotate('%s' % fits.target, xy=(0.97, 0.1),
                    xycoords = 'axes fraction', fontsize = 20, ha = 'right')
        contour_args = {'cmap': cm.gray_r, 'zorder': 100}
        scatter_args = {'marker': '.', 'color': 'black', 'alpha': 0.2,
                        'edgecolors': None, 'zorder': 1}
        plt_pts, cs = scatter_contour(color[errlimit], mag2[errlimit],
                                      threshold=10, levels=20,
                                      hist_bins=[ncolbin, nmagbin],
                                      contour_args=contour_args,
                                      scatter_args=scatter_args,
                                      ax=ax)

        ax.plot((np.min(color), np.max(color)), (fits.trgb, fits.trgb),
                color='red', lw=2)

        for region in region_names:
            ax.set_title(region)
            plt.draw()
            stage_lab = get_stage_label(region)
            pts = ginput(0, timeout=-1)
            xs, ys = np.transpose(pts)
            inds, = np.nonzero(nxutils.points_inside_poly(datapts, pts))
            stage[inds] = stage_lab
            ax.plot(color[inds], mag2[inds], '.', alpha=0.5, zorder=stage_lab)
            out.write('%s %s %s %s polygon(' %
                      (fits.target, region, filter1, filter2))
            outdata = ['%.4f, %.4f' % (x, y) for x, y in zip(xs, ys)]
            out.write(', '.join(outdata) + ')\n')
        ax.set_title('')
        plt.draw()
        figname = os.path.split(datafile)[1].replace('.dat', '.png')
        plt.savefig(os.path.join(plot_dir, figname))
        plt.close()

        logger.info('%s wrote %s' % (define_color_mag_region.__name__,
                                     datafile.replace('dat', 'png')))
        fmt = '%.8f %.8f %.4f %.4f %.4f %.4f %.0f\n'
        [dout.write(fmt % (ra[i], dec[i], mag1[i], mag2[i], mag1err[i],
         mag2err[i], stage[i])) for i in range(len(ra))]
        dout.close()
        logger.info('%s wrote %s' % (define_color_mag_region.__name__,
                                     datafile))
        plt.cla()
        out.close()
    logger.info('%s wrote %s' % (define_color_mag_region.__name__, outfile))
    return outfile


def test(regions):
    Regions = [CMDregion(r) for r in regions]
    for R in Regions:
        print R.target, R.filter1, R.filter2
        try:
            R.join_regions('BHeB', 'RHeB')
        except:
            print 'nope!'
            continue
        R.split_regions_on_mean('BHeB_RHeB')
        plt.figure()
        plt.title('%s' % R.target)
        for key in ('bheb_bym', 'rheb_bym'):
            plt.plot(R.regions[key][:, 0], R.regions[key][:, 1])


if __name__ == '__main__':
    pdb.set_trace()
    test()
