import os 
import rasterio
import matplotlib.pyplot as plt
from rasterio.windows import transform
import numpy as np
import geopandas as gpd 

for counter in [0,1,3,4]:
    output_path = f'/Users/yaroslav/Documents/Work/NASA/data/first-annotations-push/individual_cutouts/'
    input_tif = f'/Users/yaroslav/Documents/Work/NASA/data/first-annotations-push/individual_cutouts/cutout_{counter}/cutout_{counter}.tif'

    dataset = rasterio.open(input_tif) 

    dims = (dataset.width, dataset.height) 

    print(dims)

    x1, y1 = (0,0)

    window = rasterio.windows.Window(x1, y1, 1056, 1056) 
    img = dataset.read(window=window) 
    # read both layers with dataset.read(window=window)
    # read first layer with dataset.read(1, window=window)
    # read second layer with dataset.read(2, window=window)

    window_transform = transform(window, dataset.transform) 

    output_ndvi = output_path+f'cutout_{counter}/raw_ndvi_{counter}.tif'
    output_pan = output_path+f'cutout_{counter}/raw_pan_{counter}.tif'

    cutout_raster = rasterio.open(input_tif)
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
        pan_dataset.write(ndviImg)#, indexes=2)