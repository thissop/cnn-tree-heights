import geopandas as gpd 
import matplotlib.pyplot as plt 
from osgeo import gdal
import numpy as np

import georaster
from mpl_toolkits.basemap import Basemap
import rasterio
import rasterio.plot 

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

plt.savefig('/Users/yaroslav/Documents/GitHub/cnn-tree-heights/src/monthly/jan2023/library-testing/first-layer/plot_tif.png', dpi=350)