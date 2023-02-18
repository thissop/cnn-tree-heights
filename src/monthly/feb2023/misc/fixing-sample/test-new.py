from cnnheights.utilities import sample_background
import os 
import rasterio

input_tif = '/Users/yaroslav/Documents/Work/NASA/data/jesse/big mosaic/big mosaic.tif'
key = 'src/monthly/feb2023/misc/fixing-sample/sample_backgrounds_key.gpkg'

output_path = f'src/monthly/feb2023/misc/fixing-sample/'
plot_path = f'src/monthly/feb2023/misc/fixing-sample/background_test.pdf'
sample_background(input_tif=input_tif, output_path=output_path, crs='EPSG:32628', key=key, plot_path=plot_path)

fs = ['/Users/yaroslav/Documents/GitHub/cnn-tree-heights/src/monthly/feb2023/misc/fixing-sample/cutout_0.tif',
      'src/monthly/feb2023/misc/fixing-sample/cutout_0/raw_ndvi_0.tif',
      'src/monthly/feb2023/misc/fixing-sample/cutout_0/raw_pan_0.tif']

for f in fs: 
    raster = rasterio.open(f)
    print(raster.bounds)
    raster.close()

for i in [0,1,3,4]:

    fs = [f'/Users/yaroslav/Documents/Work/NASA/data/first-annotations-push/individual_cutouts/cutout_{i}/raw_ndvi_{i}.tif',
          f'/Users/yaroslav/Documents/Work/NASA/data/first-annotations-push/individual_cutouts/cutout_{i}/raw_pan_{i}.tif',
          f'/Users/yaroslav/Documents/Work/NASA/data/first-annotations-push/individual_cutouts/cutout_{i}/cutout_{i}.tif']

    for f in fs: 
        raster = rasterio.open(f)
        print(raster.bounds)
        raster.close()