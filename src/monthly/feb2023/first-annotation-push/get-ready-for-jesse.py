def vector_rectangles(): 
    import os 
    import geopandas as gpd

    counters = [0,1,3,4]
    data_base = '/Users/yaroslav/Documents/Work/NASA/data/first-annotations-push/individual_cutouts/'
    area_files = [data_base+f'cutout_{i}/vector_rectangle_{i}.gpkg' for i in counters]

    key_gdf = gpd.read_file('src/monthly/feb2023/first-annotation-push/sample_backgrounds_key.gpkg')
    print(key_gdf.crs)
    print(key_gdf['geometry'])
    for i in area_files: 
        gdf = gpd.read_file(i)
        print(gdf['geometry'])

#vector_rectangles()

def fix_vector_rectangles():
    import os 
    import geopandas as gpd
    import rasterio 
    from shapely.geometry import Polygon

    counters = [0,1,3,4]
    data_base = '/Users/yaroslav/Documents/Work/NASA/data/first-annotations-push/individual_cutouts/'
    area_files = [data_base+f'cutout_{i}/vector_rectangle_{i}.gpkg' for i in counters]
    cutout_files = [data_base+f'cutout_{i}/cutout_{i}.tif' for i in counters]

    for i, cutout in enumerate(cutout_files):
        raster = rasterio.open(cutout) 

        # BoundingBox(left=464489.310663, bottom=1691468.70945, right=465017.310663, top=1691996.70945)
        b = raster.bounds
        extracted_polygon = Polygon(((b[0],b[1]), (b[2],b[1]), (b[2], b[3]), (b[0],b[3]))) 
        vector_gdf = gpd.GeoDataFrame({'geometry':[extracted_polygon]}, crs=raster.crs)
        vector_gdf.to_file(area_files[i])

        raster.close()

#fix_vector_rectangles()

def preprocess_them_all(): 

    import numpy as np
    from cnnheights.main import preprocess
    import os 

    counters = [0,1,3,4]

    data_base = '/Users/yaroslav/Documents/Work/NASA/data/first-annotations-push/individual_cutouts/'
    area_files = [data_base+f'cutout_{i}/vector_rectangle_{i}.gpkg' for i in counters]
    annotation_files = [data_base+f'cutout_{i}/annotations_{i}.gpkg' for i in counters]
    raw_ndvi_images = [data_base+f'cutout_{i}/raw_ndvi_{i}.tif' for i in counters]
    raw_pan_images = [data_base+f'cutout_{i}/raw_pan_{i}.tif' for i in counters]

    temp = []
    for i in [area_files, annotation_files, raw_ndvi_images, raw_pan_images]:
        for j in i: 
            temp.append(os.path.exists(j))

    print(np.sum(temp), len(temp))

    output_path = '/Users/yaroslav/Documents/Work/NASA/data/first-annotations-push/first-shadows-dataset'
    preprocess(area_files=area_files, annotation_files=annotation_files, raw_ndvi_images=raw_ndvi_images, raw_pan_images=raw_pan_images, output_path=output_path)

preprocess_them_all()

def check_dtypes():
    import rasterio
    import cv2 

    input_tif = '/Users/yaroslav/Documents/Work/NASA/data/first-annotations-push/individual_cutouts/cutout_4/cutout_4.tif'
    dataset = rasterio.open(input_tif)  
    print(dataset.dtypes)

    ndvi_img_raw = '/Users/yaroslav/Documents/Work/NASA/data/first-annotations-push/individual_cutouts/cutout_4/raw_ndvi_4.png'
    print(cv2.imread(ndvi_img_raw, cv2.IMREAD_GRAYSCALE).dtype)

    qgis_ndvi_img_raw = '/Users/yaroslav/Documents/Work/NASA/data/first-annotations-push/individual_cutouts/cutout_4/qgis-raw-ndvi-4.png'
    print(cv2.imread(qgis_ndvi_img_raw, cv2.IMREAD_GRAYSCALE).dtype)

def make_ndvi_pan(): 
    import numpy as np
    from rasterio.windows import transform
    import rasterio

    counters = [0,1,3,4]

    data_base = '/Users/yaroslav/Documents/Work/NASA/data/first-annotations-push/individual_cutouts/'

    for i in counters: 
        cutout = data_base+f'cutout_{i}/cutout_{i}.tif'
        
        output_ndvi = data_base+f'cutout_{i}/raw_ndvi_{i}.tif'
        output_pan = data_base+f'cutout_{i}/raw_pan_{i}.tif'

        cutout_raster = rasterio.open(cutout)
        bounds = cutout_raster.bounds 
        print(bounds)
        width = cutout_raster.width
        height = cutout_raster.height

        panImg = np.array([cutout_raster.read(1)])
        ndviImg = np.array([cutout_raster.read(2)])

        with rasterio.open(output_ndvi, 'w',
            driver='GTiff', width=width, height=height, count=1, 
            dtype=ndviImg.dtype, crs=cutout_raster.crs) as ndvi_dataset:
            ndvi_dataset.write(ndviImg)#, indexes=2)

        with rasterio.open(output_pan, 'w', 
            driver='GTiff', width=width, height=height, count=1,
            dtype=panImg.dtype, crs=cutout_raster.crs) as pan_dataset:
            pan_dataset.write(ndviImg)#, indexes=2)

        cutout_raster.close()

        with rasterio.open(output_ndvi, 'r') as raster: 
            print(raster.bounds)

#make_ndvi_pan()

def check_old_bounds():
    import rasterio 

    cutout_raster = rasterio.open('/Users/yaroslav/Documents/Work/NASA/data/old/july2022-testing-input/pan_thaddaeus_training_area_5.tif')
    bounds = cutout_raster.bounds 
    print(bounds)

#check_old_bounds() # fixed!

#check_dtypes()