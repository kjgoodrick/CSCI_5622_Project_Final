import numpy as np
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy

def plt_ant_feat(z, x=None, y=None, fig=None, ax=None, cmap='gist_rainbow', vmin=None, vmax=None, label='', title='', kwargs_mesh={}, kwargs_cbar={}):
    if x is None or y is None:
        x = np.linspace(-179.5, 179.5, 360)
        y = np.linspace(-60.5, -89.5, 30)
        x, y = np.meshgrid(x, y)

    if len(z.shape) == 1:
        z = z.reshape(x.shape)
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
    ax.set_extent([-180, 180, -90, -60], ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.COASTLINE, lw=1)
    mesh = ax.pcolormesh(x, y, z, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(), **kwargs_mesh)
    cb = plt.colorbar(mesh, ax=ax, **kwargs_cbar)
    cb.set_label(label)
    ax.set_title(title)
    
    return