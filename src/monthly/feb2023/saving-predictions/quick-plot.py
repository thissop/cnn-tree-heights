import matplotlib.pyplot as plt 
import geopandas as gpd 

import rasterio 
import rasterio.plot 

predictions_file = '/Users/yaroslav/Documents/GitHub/cnn-tree-heights/src/monthly/feb2023/saving-predictions/predicted_polygons.shp'

fig, ax = plt.subplots(figsize=(5,5))

gdf = gpd.read_file(predictions_file)

raster = rasterio.open('/Users/yaroslav/Documents/Work/NASA/data/old/july2022-testing-input/ndvi_thaddaeus_training_area_1.tif')

gdf['geometry'].plot(ax=ax)

rasterio.plot.show(raster, ax=ax, cmap='Greys_r')

#ax.set_aspect('equal', 'box')

plt.savefig('/Users/yaroslav/Documents/GitHub/cnn-tree-heights/src/monthly/feb2023/saving-predictions/quick-plot.png', dpi=200)

from cnnheights.utilities import shadows_from_annotations

cutlines_shp = '/Users/yaroslav/Documents/Work/NASA/data/jesse/thaddaeus_cutline/SSAr2_32628_GE01-QB02-WV02-WV03-WV04_PAN_NDVI_010_003_mosaic_cutlines.shp'
background_tif = '/Users/yaroslav/Documents/Work/NASA/layers/first-working-input/cutout_1.tif'
big_tif = '/Users/yaroslav/Documents/Work/NASA/data/jesse/big mosaic/big mosaic.tif'

save_path = '/Users/yaroslav/Documents/GitHub/cnn-tree-heights/src/monthly/feb2023/saving-predictions/shadows_gdf.feather'

shadows_gdf = shadows_from_annotations(annotations_gpkg=predictions_file, 
                                       cutlines_shp=cutlines_shp, 
                                       north=1707319.01, east=450252.85, epsg='32628', save_path=save_path) # 450252.85,1707319.01

from cnnheights.plotting import plot_shadow_lengths
plot_path = 'src/monthly/feb2023/saving-predictions/predicted heights from predictions.png'
plot_shadow_lengths(shadows_gdf.iloc[0:10], background_tif=big_tif, save_path=plot_path)

