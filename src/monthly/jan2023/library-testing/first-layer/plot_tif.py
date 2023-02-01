import geopandas as gpd 
import matplotlib.pyplot as plt 
from osgeo import gdal
import numpy as np

import georaster
from mpl_toolkits.basemap import Basemap
import rasterio
import rasterio.plot 
from shapely.geometry import box

fp = '/Users/yaroslav/Documents/Work/NASA/layers/first-working-input/cutout_1.tif'
annotations_gdf = gpd.read_file('/Users/yaroslav/Documents/Work/NASA/layers/first-working-input/annotations_1.gpkg')
annotations_gdf.set_crs('epsg:32628', allow_override=True)

fig, ax = plt.subplots()

'''
dataset = gdal.Open(fp)
b1 = dataset.GetRasterBand(1).ReadAsArray() # PAN channel
b2 = dataset.GetRasterBand(2).ReadAsArray() # NDVI channel
ax.imshow(b1)

print(b1[2])
'''

raster = rasterio.open(fp)

rasterio.plot.show(raster, ax=ax, cmap='Greys_r') # you can do cmap, vmin and vmax! 

annotations_gdf.plot(ax=ax, color='orange')

bounds = annotations_gdf.bounds
print(bounds)
dx = np.abs(np.abs(bounds['maxx'])-np.abs(bounds['minx']))
dy = np.abs(np.abs(bounds['maxy'])-np.abs(bounds['miny']))
dxy = np.max(np.array([dx,dy]).T, axis=1)
print(dxy)

square_bounds = np.array([[minx, miny, minx+diff, miny+diff] for minx, miny, diff in zip(bounds['minx'], bounds['miny'], dxy)])


d = {'geometry':[box(*i) for i in square_bounds]}
bounds_gdf = gpd.GeoDataFrame(d, crs='epsg:32628')

bounds_gdf.plot(ax=ax, color="None", edgecolor='red')

plt.savefig('/Users/yaroslav/Documents/GitHub/cnn-tree-heights/src/monthly/jan2023/library-testing/first-layer/plot_tif.png', dpi=350)