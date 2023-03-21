import rasterio 
import os 
import numpy as np

def fix_tiffs(): 

    output_dir = ''
    for counter in []: 

        output_tif = ''

        output_ndvi = os.path.join(output_dir, f'raw_ndvi_{counter}.tif')
        output_pan = os.path.join(output_dir, f'raw_pan_{counter}.tif')

        cutout_raster = rasterio.open(output_tif)
        width = cutout_raster.width
        height = cutout_raster.height

        panImg = np.array([cutout_raster.read(1)])
        ndviImg = np.array([cutout_raster.read(2)])

        with rasterio.open(output_ndvi, 'w',
            driver='GTiff', width=width, height=height, count=1,
            dtype=ndviImg.dtype, crs=cutout_raster.crs, transform=window_transform) as ndvi_dataset:
            ndvi_dataset.write(ndviImg)#, indexes=2)

        with rasterio.open(output_pan, 'w',
            driver='GTiff', width=width, height=height, count=1,
            dtype=panImg.dtype, crs=cutout_raster.crs, transform=window_transform) as pan_dataset:
            pan_dataset.write(panImg)#, indexes=2)

        cutout_raster.close()
