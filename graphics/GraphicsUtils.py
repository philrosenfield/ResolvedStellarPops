import numpy as np

import matplotlib.nxutils as nxutils
import matplotlib.pyplot as plt
from matplotlib import cm,rc,rcParams
from matplotlib.patches import FancyArrow
from matplotlib.ticker import NullFormatter, MultipleLocator

nullfmt   = NullFormatter() # no labels
rcParams['text.usetex']=True
rcParams['text.latex.unicode']=False
rcParams['axes.linewidth'] = 2
rcParams['ytick.labelsize'] = 'large'
rcParams['xtick.labelsize'] = 'large'
rcParams['axes.edgecolor'] = 'grey'
rc('text',usetex=True)

import math_utils

def discrete_colors(Ncolors,colormap='gist_rainbow'):
    colors = []
    cmap = cm.get_cmap(colormap)
    for i in range(Ncolors):
        colors.append( cmap(1.*i/Ncolors) ) # color will now be an RGBA tuple
    return colors

def load_scatter_kwargs(color_array,cmap=cm.jet):
    kw = {'zorder': 100,
          'alpha': 1,
          'edgecolor': 'k',
          'c': color_array,
          'cmap': cmap,
          'vmin': np.min(color_array),
          'vmax': np.max(color_array)
          }
    
    return kw

def scatter_colorbar(x,y,color_array,markersize = 20, ax = None):
    '''
    makes a scatter plot with a float array as colors.
    '''
    # c_array is an array of floats -- so this would be your fit value
    scatter_kwargs = load_scatter_kwargs(color_array)
    if ax == None: ax = plt.axes()
    sc = ax.scatter(x, y, markersize, **scatter_kwargs)
    
    # lets you get the colors if you want them for something else
    # ie sc.get_facecolors()
    # you might not need to do this     
    #sc.update_scalarmappable()

    # then to get the colorbar
    plt.colorbar(sc)
    return ax,sc


def reverse_yaxis(ax):
    ax.set_ylim(ax.get_ylim()[::-1])

def reverse_xaxis(ax):
    ax.set_xlim(ax.get_xlim()[::-1])

def load_ann_kwargs():
    from matplotlib.patheffects import withStroke
    myeffect = withStroke(foreground="w", linewidth=3)
    ann_kwargs = dict(path_effects=[myeffect])
    return ann_kwargs

ann_kwargs = load_ann_kwargs()

def plot_cmd(fitsfile,yaxis='I',upper_contour=False,**kwargs):
    '''
    upper_contour contour only above TRGB
    kwargs: outfile, save_plot
    absmag
    '''
    new_plot = kwargs.get('new_plot',True)
    ax = kwargs.get('ax','')
    save_plot = kwargs.get('save_plot',False)
    if fitsfile==None:
        mag1 = kwargs.get('mag1',None)
        mag2 = kwargs.get('mag2',None)
        filter1 = kwargs.get('filter1','')
        filter2 = kwargs.get('filter2','')
        title = kwargs.get('title','')
    else:
        fits = ReadFileUtils.read_binary_table(fitsfile)
        filter1,filter2 = misc.get_filters(fitsfile)
        title = misc.get_title(fitsfile)
        trgb= Astronomy.get_tab5_trgb_Av_dmod(title)[0]
        try:
            mag1 = fits.get_col('MAG1_WFPC2')
            mag2 = fits.get_col('MAG2_WFPC2')
        except:
            mag1 = fits.get_col('MAG1_ACS')
            mag2 = fits.get_col('MAG2_ACS')
    
    abs_mag = kwargs.get('abs_mag',True)
    if abs_mag == True:
        mag1 = Astronomy.HSTmag2Mag(mag1,title,filter1)
        mag2 = Astronomy.HSTmag2Mag(mag2,title,filter2)
    
    color = mag1-mag2
        
    if yaxis == 'I':
        mag = mag2
        Filter = filter2
    else:
        mag = mag1
        Filter = filter1

    dx = 0.05
    dy = 0.05
    if upper_contour == False:
        Z,xrange,yrange = math-utils.bin_up(color,mag,dx=dx,dy=dy)
    else:
        Trgb = trgb-dmod-Av
        inds = np.nonzero(mag<(Trgb))[0]
        Z,xrange,yrange = math-utils.bin_up(color[inds],mag[inds],dx=dx,dy=dy)

    if new_plot == True:
        fig = plt.figure()
        ax = plt.axes()
    
    ax.plot(color,mag,',',color='black')
    cs = plt.contour(Z,10,extent=[xrange[0],xrange[-1],yrange[0],yrange[-1]],zorder=10)
    #ax.clabel(cs, cs.levels, inline=True, fmt='%r %%', fontsize=10)
    cbar = plt.colorbar(cs)
    cbar.ax.set_ylabel(r'$\rho_{*}$')
    ax.set_xlabel(r'$%s-%s$'%(filter1,filter2))
    ax.set_ylabel(r'$%s$'%Filter)
    if title != '':
        ax.set_title(r'$%s$'%title)
    reverse_yaxis(ax)
    xlim = kwargs.get('xlim','None')
    if xlim != 'None': ax.set_xlim(xlim)
    if save_plot==True:
        outfile = kwargs.get('outfile',fitsfile+'.cmd.png')
        plt.savefig(outfile)
        print 'wrote',outfile
        plt.close()
    return ax

def basic_plot(xdata,ydata,**kwargs):
    """ Basic 2-d plot. 
    takes arrays xdata, ydata and possible kwargs:
    
    kwarg , default param, notes
    ax, None, axis instance to plot to, if none, will make fig with:
    xsize, 8, figure xdim inches
    ysize, 8, figure ydim inches
    
    
    xlabel, '', xaxis label
    ylabel, '', yaxis label
    label, '', data legend label
    lw, 2, line width
    marker, '.', dot
    color, black, data color
    linestyle = '-'
    returns ax instance
    """ 
    ax = kwargs.get('ax',None)
    if ax == None:
        xsize = kwargs.get('xsize',8)
        ysize = kwargs.get('ysize',8)
        fig = plt.figure(figsize=(xsize,ysize))
        ax = plt.axes()
    
    title = kwargs.get('title',None)
    xlim = kwargs.get('xlim',None)
    ylim = kwargs.get('ylim',None)
    xlabel = kwargs.get(r'xlabel','')
    ylabel = kwargs.get(r'ylabel','')
    label = kwargs.get('label','')
    marker = kwargs.get('marker','.')
    color = kwargs.get('color','black')
    lw = kwargs.get('lw',2)
    linestyle = kwargs.get('linestyle','-')
    ax.plot(xdata,ydata,marker,
            color=color,lw=lw,label=label,linestyle=linestyle)
    if xlim !=None: ax.set_xlim(xlim)
    if ylim !=None: ax.set_ylim(ylim)
    if title !=None: ax.set_title(title)
    if xlabel !=None: ax.set_xlabel(xlabel)
    if ylabel !=None: ax.set_ylabel(ylabel)
    return ax

def basic_line_plot(xdata,ydata,**kwargs):
    """ Basic 2-d line plot. 
    takes arrays xdata, ydata and possible kwargs:
    
    kwarg , default param, notes
    ax, None, axis instance to plot to, if none, will make fig with:
    xsize, 8, figure xdim inches
    ysize, 8, figure ydim inches
    
    
    xlabel, '', xaxis label
    ylabel, '', yaxis label
    label, '', data legend label
    lw, 2, line width
    color, black, data color
    linestyle = '-'
    returns ax instance
    """ 
    ax = kwargs.get('ax',None)
    if ax == None:
        xsize = kwargs.get('xsize',8)
        ysize = kwargs.get('ysize',8)
        fig = plt.figure(figsize=(xsize,ysize))
        ax = plt.axes()
    
    title = kwargs.get('title',None)
    xlim = kwargs.get('xlim',None)
    ylim = kwargs.get('ylim',None)
    xlabel = kwargs.get(r'xlabel','')
    ylabel = kwargs.get(r'ylabel','')
    label = kwargs.get(r'label','')
    color = kwargs.get('color','black')
    lw = kwargs.get('lw',2)
    linestyle = kwargs.get('linestyle','-')
    ax.plot(xdata,ydata,
            color=color,lw=lw,label=label,linestyle=linestyle)
    if xlim !=None: ax.set_xlim(xlim)
    if ylim !=None: ax.set_ylim(ylim)
    if title !=None: ax.set_title(title)
    if xlabel !=None: ax.set_xlabel(xlabel)
    if ylabel !=None: ax.set_ylabel(ylabel)
    return ax



def set_up_three_panel_plot():
    fig = plt.figure(figsize=(8,8))
    
    left,width= 0.1,0.4
    bottom, height = 0.1, 0.4
    d=0.01
    lefter = left+width+d
    mid = bottom+height+2*d
    lefts = [left,lefter,left]
    bottoms = [mid,mid,bottom]
    widths = [width,width,2*width+d]
    heights = [height,height,height-0.1]
    
    axs = [plt.axes([l,b,w,h]) for l,b,w,h in zip(lefts,bottoms,widths,heights)]
    return axs

def two_panel_plot(sizex,sizey,xlab1,xlab2,ylab,ylab2=None,ftsize=20,mag2_cut=0,mag2_max=1):
    fig = plt.figure(2, figsize=(sizex,sizey))
    left, width = 0.1, 0.4
    bottom, height = 0.12, 0.8  
    
    left2 = left+width + 0.065
    if ylab2 != None: left2 = left+width + 0.08
    axis1 = [left, bottom, width, height]
    axis2 = [left2, bottom, width, height]
    
    ax1 = plt.axes(axis1)
    ax1.set_xlim( (mag2_cut,mag2_max) ) # set all axes limits here
    #ax1.set_ylim( (0.0001, 10.) )
    ax1.set_xlabel(r'%s'%xlab1,fontsize=ftsize)
    ax1.set_ylabel(r'%s'%ylab,fontsize=ftsize)
    
    ax2 = plt.axes(axis2)
    ax2.set_xlim( ax1.get_xlim() )
    #ax2.set_ylim( ax1.get_ylim() )
    ax2.set_xlabel(r'%s'%xlab2,fontsize=ftsize)
    if ylab2 !=None: ax2.set_ylabel(r'%s'%ylab2,fontsize=ftsize)
    return ax1,ax2

def two_panel_plot_vert(oney=True,ftsize=20,mag2_cut=0,mag2_max=1):
    fig = plt.figure(2, figsize=(8,8))
    left, width = 0.13, 0.83
    bottom, height = 0.1, 0.41
    dh = 0.03
    
    axis1 = [left, bottom, width, height]
    axis2 = [left, (bottom+height+dh), width, height]
    
    ax1 = plt.axes(axis1)
    ax1.set_xlim( (mag2_cut,mag2_max) ) # set all axes limits here
    #ax1.set_ylim( (0.0001, 10.) )
    if oney==True: ax1.annotate(r'$\#/ 3.6 \mu \rm{m\ Region\ Integrated\ Flux\ (Jy}^{-1}\rm{)}$',fontsize=ftsize,xy=(0.025,.5),xycoords='figure fraction',va='center',rotation='vertical')
    ax1.set_xlabel(r'$\rm{mag}$',fontsize=ftsize)    
    ax2 = plt.axes(axis2)
    ax2.set_xlim( ax1.get_xlim() )
    #ax2.set_ylim( ax1.get_ylim() )
    ax2.xaxis.set_major_formatter(nullfmt)
    
    return ax1,ax2

def setup_four_panel(ftsize=20):
    fig = plt.figure(figsize=(8,8))
    left, width = 0.1, 0.4
    bottom, height = 0.1, 0.4
    lefter = left+width+0.01
    higher = bottom+height+0.01
    
    # plot and fig sizes
    fig = plt.figure(1, figsize=(8,8))
    
    ll_axis = [left, bottom, width, height]
    lr_axis = [lefter, bottom, width, height] 
    ul_axis = [left, higher, width, height]
    ur_axis = [lefter, higher, width, height]
    
    ax_ll = plt.axes(ll_axis)
    ax_ll.set_xlim( (-0.75,1)) # model and data x limits here
    ax_ll.set_ylim( (24.8, 18)) # set all y limits here
    
    ax_lr = plt.axes(lr_axis)
    ax_lr.set_xlim( ax_ll.get_xlim() )
    ax_lr.set_ylim( ax_ll.get_ylim() )    
    
    ax_ul = plt.axes(ul_axis)
    ax_ul.set_xlim( ax_ll.get_xlim() )
    ax_ul.set_ylim( ax_ll.get_ylim() )    
    
    ax_ur = plt.axes(ur_axis)
    ax_ur.set_xlim( ax_ll.get_xlim() )
    ax_ur.set_ylim( ax_ll.get_ylim() )    
    
    ax_lr.yaxis.set_major_formatter(nullfmt)
    ax_ur.yaxis.set_major_formatter(nullfmt)
    ax_ur.xaxis.set_major_formatter(nullfmt)
    ax_ul.xaxis.set_major_formatter(nullfmt)
    
    # titles
    #x = fig.text(0.5,0.96,r'$\rm{%s}$' % ('Disk Field'),horizontalalignment='center',verticalalignment='top',size=20)
    ax_ur.set_title(r'$\rm{Disk\ Field}$',color='black',fontsize=ftsize)
    ax_ul.set_title(r'$\rm{Bulge\ Field}$',color='black',fontsize=ftsize)
    ax_ll.set_ylabel(r'$F336W$',fontsize=ftsize)
    ax_ll.set_xlabel(r'$F275W-F336W$',fontsize=ftsize)
    ax_ul.set_ylabel(ax_ll.get_ylabel(),fontsize=ftsize)    
    ax_lr.set_xlabel(ax_ll.get_xlabel(),fontsize=ftsize)
    
    return ax_ll,ax_lr,ax_ul,ax_ur

def setup_five_panel_plot(ftsize=20):
    nullfmt   = NullFormatter() # no labels
    left, width = 0.06, 0.175
    bottom, height = 0.14, 0.82
    dl = 0.01
    lefts = [left+(width+dl)*float(i) for i in range(5)]

    # plot and fig sizes
    fig = plt.figure(1, figsize=(15,4))

    axiss = [[l,bottom,width,height] for l in lefts]

    axs = []
    for i in range(len(axiss)):
        axs.append( plt.axes(axiss[i]))

    for ax in axs:
        ax.set_xlim( (-0.75,2)) # model and data x limits here
        ax.set_ylim( (24.8, 19)) # set all y limits here
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        if axs.index(ax) > 0: ax.yaxis.set_major_formatter(nullfmt)
        ax.set_xlabel(r'$F275W-F336W$',fontsize=ftsize)
        if axs.index(ax) ==0: ax.set_ylabel(r'$F336W$',fontsize=ftsize)
        
    return axs


def histOutline(dataIn, *args, **kwargs):
    '''
    Then you do something like..

    Any of the np keyword arguments can be passed:

    bins, hist = histOutline(x)

    plt.plot(bins, hist)

    or plt.fill_between(bins, hist)
    '''
    (histIn, binsIn) = np.histogram(dataIn, *args, **kwargs)

    stepSize = binsIn[1] - binsIn[0]

    bins = np.zeros(len(binsIn)*2 + 2, dtype=np.float)
    data = np.zeros(len(binsIn)*2 + 2, dtype=np.float)
    for bb in range(len(binsIn)):
        bins[2*bb + 1] = binsIn[bb]
        bins[2*bb + 2] = binsIn[bb] + stepSize
        if bb < len(histIn):
            data[2*bb + 1] = histIn[bb]
            data[2*bb + 2] = histIn[bb]

    bins[0] = bins[1]
    bins[-1] = bins[-2]
    data[0] = 0
    data[-1] = 0

    return (bins, data)

'''
Stuff from Adrienne:
I started this email because I wanted to know if you had the answer, and I found it along the way. Now I think it's useful knowledge. Basically, it's an easy way to put a legend anywhere on a figure in figure coordinates, not axis coordinates or data coordinates. Helpful if you have one legend for multiple subplots.

Basically, you can do a transform in legend to tell matplotlib that you're specifying the coordinates in data units or in axis units, ie,

bbox_transform = ax.transAxes (0 - 1 means left to right or bottom to top of current axis)

bbox_transform = ax.transData (specify location in data coordinates for that particular axis)

bbox_transform = fig.transFigure (specify location in figure coordinates, so 0 - 1 means bottom to top or left to right of the current *figure*, not the current axis)

Now I am trying to figure out how to get it to use points from different subplots in one legend.. fun.

proxy artists:
    p_corr = matplotlib.lines.Line2D([0], [0], marker = 'o', color = 'k',
                                     linestyle = 'None')
    p_uncorr = matplotlib.lines.Line2D([0], [0], marker = 'o', color = '0.7',
                                       linestyle = 'None',
                                       markeredgecolor = '0.7')

    l = plt.legend([p_corr, p_uncorr], ["Correlated", "Uncorrelated"],
                   bbox_transform = fig.transFigure, loc = 'upper right',
                   bbox_to_anchor = (0.9, 0.9),
                   numpoints = 1,
                   title = "Log scaling",
                   borderpad = 1,
                   handletextpad = 0.3,
                   labelspacing = 0.5)#,
                   #prop = matplotlib.font_manager.FontProperties(size=20))

I've been looking for this for a while. I don't know how you do colors in plotting but I think it's different than me. This is good for scatterplots with color.

Easy to way go from array of float values of any range -> floats between 0 and 1 with some scaling so you can pass it to a colormap -> colors:


import matplotlib

norm_instance = matplotlib.colors.Normalize(vmin = np.min(float_array),
    vmax = np.max(float_array) )

normed_floats = norm_instance( float_array )

colors = matplotlib.cm.jet( normed_floats ) # or any cmap really.


and voila, colors is your usual array of [R G B alpha] values for the color or each point. I've been trying to find something like this for a long time and finally stumbled across the right google search terms.

There's also a matplotlib.colors.LogNorm if you want to normalize the colors on a log scale.

'''
    
