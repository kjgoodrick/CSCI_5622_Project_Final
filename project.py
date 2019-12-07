import numpy as np
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy
# from mpl_toolkits.basemap import maskoceans
from shapely.ops import cascaded_union

artic_poly = cartopy.feature.NaturalEarthFeature('physical', 'antarctic_ice_shelves_polys', '10m')
# coast_poly = cartopy.feature.COASTLINE
ocean_poly = cartopy.feature.OCEAN
artic_polys = list(artic_poly.geometries())
# artic_polys.extend(coast_poly.geometries())
artic_polys = cascaded_union(artic_polys)
ocean_polys = cascaded_union(list(ocean_poly.geometries()))

def plt_ant_feat(z, x=None, y=None, fig=None, ax=None, cmap='jet', vmin=20, vmax=135, hide_water=True, ups=False,  label='', title='', kwargs_mesh={'shading':'gouraud', 'aa':True}, kwargs_cbar={}):
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
    ax.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'antarctic_ice_shelves_polys', '10m'),facecolor='none', edgecolor='black')
    if hide_water:
        ax.add_geometries(ocean_polys.symmetric_difference(artic_polys), ccrs.PlateCarree(), facecolor='w', edgecolor=None)
        ax.add_feature(cartopy.feature.NaturalEarthFeature('physical', 'antarctic_ice_shelves_polys', '10m'), facecolor='none', edgecolor='black', lw=1)
    if ups:
        mesh = ax.scatter(x, y, c=z, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        mesh = ax.pcolormesh(x, y, z, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(), **kwargs_mesh)
    cb = plt.colorbar(mesh, ax=ax, **kwargs_cbar)
    cb.set_label(label)
    ax.set_title(title)
    
    return