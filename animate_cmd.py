import os
import ResolvedStellarPops as rsp
import matplotlib.pyplot as plt
import brewer2mpl

galfile = '/Users/phil/research/BRratio/models/TRILEGAL_RUNS/big_galaxy/output_big_galaxy_wfpc2_z0.001.dat'
sgal = rsp.Galaxies.simgalaxy(galfile, filter1='F555W', filter2='F814W')

bmap = brewer2mpl.get_map('Paired', map_type='Qualitative', number=7)

ages = np.array([50e6, 100e6, 500e6, 1e9, 2e9, 5e9, 10e9])
lages = np.log10(ages)

lage = sgal.data.get_col('logAge')

inds = np.digitize(lage, lages)

xlim = (-1, 4.5)
ylim = (.1, -7.5)
plot_args = {'mew': 0}
fig = plt.figure(figsize=(8, 8))
ax = plt.axes()
for j, i in enumerate(np.unique(inds)):
    if i == len(ages):
        continue
    plot_args['color'] = bmap.mpl_colors[j]
    slice_inds, = np.nonzero(inds == i)
    sgal.plot_cmd(sgal.Color, sgal.Mag2, slice_inds=slice_inds, fig=fig, ax=ax,
                  scatter_off=True, xlim=xlim, ylim=ylim, plot_args=plot_args)
    plt.savefig('cmd_%i.png' % (j+1))