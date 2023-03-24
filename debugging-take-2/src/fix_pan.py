import matplotlib.pyplot as plt
from rasterio import plot
from rasterio.windows import transform
import rasterio
from shapely.geometry import Polygon
import os 
import numpy as np
import geopandas as gpd 
import rasterio 

def re_make_pan_tiff(): 

    for i in [0,1,3,4]: 
        cutout_path  = f'/Users/yaroslav/Documents/Work/NASA/old/data/first-annotations-push/individual_cutouts/cutout_{i}/cutout_{i}.tif'  
        pan_path = f'/Users/yaroslav/Documents/Work/NASA/old/data/first-annotations-push/individual_cutouts/cutout_{i}/raw_pan_{i}_[fixed].tif'

        cutout_raster = rasterio.open(cutout_path)

        panImg = np.array([cutout_raster.read(1)])

        width = cutout_raster.width
        height = cutout_raster.height

        window = rasterio.windows.Window(0, 0, width, height) 
        with rasterio.open(pan_path, 'w',
            driver='GTiff', width=width, height=height, count=1,
            dtype=panImg.dtype, crs=cutout_raster.crs, transform=cutout_raster.window_transform(window)) as pan_dataset:
            pan_dataset.write(panImg)#, indexes=2)

        cutout_raster.close()

        # randomly choose left corner 
        # eventually, make sure no overlap with other regions 

#re_make_pan_tiff()

def reprocess(): 
    from cnnheights.preprocessing import preprocess
    import shutil 

    pre_preprocess_dir = '/Users/yaroslav/Documents/Work/NASA/old/data/first-annotations-push/fixed-input-data/not-preprocessed'

    for i in [0,1,3,4]: 
        cutout_path  = f'/Users/yaroslav/Documents/Work/NASA/old/data/first-annotations-push/individual_cutouts/cutout_{i}/cutout_{i}.tif'  
        pan_path = f'/Users/yaroslav/Documents/Work/NASA/old/data/first-annotations-push/individual_cutouts/cutout_{i}/raw_pan_{i}_[fixed].tif'
        ndvi_path = f'/Users/yaroslav/Documents/Work/NASA/old/data/first-annotations-push/individual_cutouts/cutout_{i}/raw_ndvi_{i}.tif'
        vector_rectangle_path = f'/Users/yaroslav/Documents/Work/NASA/old/data/first-annotations-push/individual_cutouts/cutout_{i}/vector_rectangle_{i}.gpkg'
        annotation_path = f'/Users/yaroslav/Documents/Work/NASA/old/data/first-annotations-push/individual_cutouts/cutout_{i}/annotations_{i}.gpkg'
        for f in [pan_path, ndvi_path, vector_rectangle_path, annotation_path]: 
            shutil.copyfile(f, os.path.join(pre_preprocess_dir, f.split('/')[-1]))

    output_dir = '/Users/yaroslav/Documents/Work/NASA/old/data/first-annotations-push/fixed-input-data/preprocessed'

    preprocess(input_data_dir=pre_preprocess_dir, output_data_dir=output_dir)

reprocess()