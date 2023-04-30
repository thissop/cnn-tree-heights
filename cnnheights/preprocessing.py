# CURRENT HIGH LEVEL PREPROCESS FUNCTIONS # 

def sample_background(input_tif:str, output_dir:str, crs:str, key:str=None, counter:int=0, plot_path:str=None, sample_dim:tuple=(1024,1024), dpi=350):
    r'''

    TO DO 
    -----

    Notes
    -----
    - note for jesse: opening from this window saves crazy memory. 
    - FIX THE vector background!!

    '''
    import rasterio 

    # randomly choose left corner 
    # eventually, make sure no overlap with other regions 

    import matplotlib.pyplot as plt
    from rasterio import plot
    from rasterio.windows import transform
    import rasterio
    from shapely.geometry import Polygon
    import os 
    import numpy as np
    import geopandas as gpd 
    from cnnheights.plotting import save_fig
    from shapely.geometry import box

    dataset = rasterio.open(input_tif) 

    dims = (dataset.width, dataset.height) 

    extant_key = False 
    if key is not None and os.path.exists(key):
        extant_key = True 

    output_dir = os.path.join(output_dir, f'cutout_{counter}')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    overlapping = True 
    while overlapping: 

        x1, y1 = (np.random.randint(0,dims[0]-sample_dim[0]), np.random.randint(0,dims[1]-sample_dim[1])) ## FIXX!! 
        x2, y2 = (x1+sample_dim[0], y1+sample_dim[1])

        window = rasterio.windows.Window(x1, y1, sample_dim[0], sample_dim[1]) 
        img = dataset.read(window=window) 
        # read both layers with dataset.read(window=window)
        # read first layer with dataset.read(1, window=window)
        # read second layer with dataset.read(2, window=window)
     
        window_transform = transform(window, dataset.transform) 
        extent = plot.plotting_extent(img, window_transform) 
        
        x1 = extent[0]
        #x2 = extent[1], #y2 = extent[3]
        y1 = extent[2]

        output_tif = os.path.join(output_dir, f'cutout_{counter}.tif')

        with rasterio.open(output_tif, 'w',
                driver='GTiff', width=sample_dim[0], height=sample_dim[1], count=2,
                dtype=img.dtype, crs=dataset.crs, transform=window_transform) as new_data_set:
            new_data_set.write(img)#, indexes=2)

        cutout_bounds = rasterio.open(output_tif).bounds
        extracted_polygon = box(*cutout_bounds)
        vector_rect_path = os.path.join(output_dir, f'vector_rectangle_{counter}.gpkg')
        vector_rectangle = gpd.GeoDataFrame({"geometry":[extracted_polygon]})
        vector_rectangle.to_file(vector_rect_path, driver='GPKG')#, layer='name')

        # PROBLEM COULD BE HERE? 

        if extant_key: 
            key_gdf = gpd.read_file(key)
            extracted_geometries = key_gdf['geometry']

            overlapping = extracted_geometries.overlaps(extracted_polygon)[0] # this probably needs to get fixed

        else: 
            overlapping = False      
            
    centroid = [np.median(extent[0:2]), np.median(extent[3:5])]

    if extant_key: 
        key_gdf = gpd.read_file(key)
        d = {'geometry':[extracted_polygon]+key_gdf['geometry'].to_list(),
             'centroidsx':[centroid[0]]+key_gdf['centroidsx'].to_list(), 
             'centroidsy':[centroid[1]]+key_gdf['centroidsy'].to_list()}

        key_gdf = gpd.GeoDataFrame(d, crs=crs)
        key_gdf.to_file(key)

    else: 
        d = {'geometry':[extracted_polygon], 'centroidsx':[centroid[0]], 'centroidsy':[centroid[1]]}
        key_gdf = gpd.GeoDataFrame(d, crs=crs) #lines_gdf = gpd.GeoDataFrame({'geometry':lines}, geometry='geometry', crs=annotations_gdf.crs)
        key_gdf.to_file(key)

    if plot_path is not None: 
        # load img here using rasterio
        fig, ax = plt.subplots(figsize=(5,5))

        plot.show(img[0], origin='upper', transform=window_transform, 
                  extent=extent, interpolation=None, ax=ax, vmin=250, vmax=750, cmap='Greys_r')
        plot.show(img[1], origin='upper', transform=window_transform, 
                  extent=extent, interpolation=None, cmap='Reds', ax=ax, alpha=0.5)
        
        #blended = lighten_only(np.array([img[0].astype(float)]), np.array([img[1].astype(float)]), opacity=0.5)
        #ax.plot(blended)
        
        ax.set_aspect('equal', 'box')
        
        save_fig(plot_path, dpi)

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

    return key_gdf

def preprocess(input_data_dir:str, output_data_dir:str): 

    r'''
    _Prepare all the standardized training data for the CNN. Extracts annotations, boundaries, nvdi images, and pan images._  

    Arguments
    ----------      

   
    output_path : `str`
        Output path for all the extracted files to be saved to. Should be Linux/Mac style, and last character should be forward slash `/` 

    Notes
    -----

    - The corresponding files in the area_files, annotation_files, raw_ndvi_images, and raw_pan_images lists need to all be in the same order index wise.  

    '''

    import geopandas as gps
    from cnnheights.preprocessing import extract_overlapping, divide_training_polygons
    import warnings 
    import numpy as np
    import os 

    #warnings.filterwarnings("ignore")

    #for f in os.listdir(output_data_dir): 
    #    if 'gpkg-shm' or 'gpkg-wal' in f: 
    #        os.remove(os.path.join(output_data_dir, f))

    input_files = [os.path.join(input_data_dir, i) for i in os.listdir(input_data_dir)]
    
    area_files = np.sort([i for i in input_files if 'vector_rectangle' in i]) 
    annotation_files = np.sort([i for i in input_files if 'annotation' in i]) 
    raw_ndvi_images = np.sort([i for i in input_files if 'raw_ndvi' in i])
    raw_pan_images = np.sort([i for i in input_files if 'raw_pan' in i])

    allAreasWithPolygons = [] 

    print([len(i) for i in (area_files, annotation_files, raw_ndvi_images, raw_pan_images)])
    write_counters = []
    for i in range(len(area_files)): 
        trainingArea = gps.read_file(area_files[i])
        trainingPolygon = gps.read_file(annotation_files[i])
        write_counters.append(int(area_files[i].split('_')[-1].split('.')[0]))

        #print(f'Read a total of {trainingPolygon.shape[0]} object polygons and {trainingArea.shape[0]} training areas.')
        #print(f'Polygons will be assigned to training areas in the next steps.') 

        #Check if the training areas and the training polygons have the same crs
        if trainingArea.crs  != trainingPolygon.crs:
            print('Training area CRS does not match training_polygon CRS')
            targetCRS = trainingPolygon.crs #Areas are less in number so conversion should be faster
            trainingArea = trainingArea.to_crs(targetCRS)
        
            print(trainingPolygon.crs)
            print(trainingArea.crs)
        
        assert trainingPolygon.crs == trainingArea.crs

        trainingArea['id'] = range(trainingArea.shape[0])
        #print(trainingArea)
        
        # areasWithPolygons contains the object polygons and weighted boundaries for each area!
        areasWithPolygons = divide_training_polygons(trainingPolygon, trainingArea, show_boundaries_during_processing=False)
        #print(f'Assigned training polygons in {len(areasWithPolygons)} training areas and created weighted boundaries for polygons')

        allAreasWithPolygons.append(areasWithPolygons)

    #Parallel(n_jobs=n_jobs)(preprocess_single(index) for index in range(total_jobs))

    inputImages = list(zip(raw_ndvi_images, raw_pan_images))
    #print(len(inputImages))
    #print(f'Found a total of {len(input_images)} pair of raw image(s) to process!')

    # For each raw satellite image, determine if it overlaps with a training area. 
    # If a overlap if found, then extract + write the overlapping part of the raw image, create + write an image from training polygons and create + write an image from boundary weights in the that overlapping region.
    # Run the main function for extracting part of ndvi and pan images that overlap with training areas
    extract_overlapping(inputImages, allAreasWithPolygons=allAreasWithPolygons, writePath=output_data_dir, ndviFilename='extracted_ndvi',
                                                panFilename='extracted_pan', annotationFilename='extracted_annotation',
                                                boundaryFilename='extracted_boundary', bands=[0], write_counters=write_counters)

    #for f in os.listdir(output_data_dir): 
    #    if 'gpkg-shm' or 'gpkg-wal' in f: 
    #        os.remove(os.path.join(output_data_dir, f))

# DEVELOPMENT HIGH LEVEL PREPROCESS FUNCTIONS #

def combined_preprocess(input_data_dir:str, output_data_dir:str): 
    r'''
    
    Arguments
    -----   

    input_data_dir : str
        - directory string with f"vector_rectangle_{v_id}.gpkg", f"annotations_{vid}.gpkg", f"raw_ndvi_{v_id}.tif", and f"raw_pan_{v_id}.tif" files

    output_data_dir : str
        - directory into which f"extracted_ndvi_{v_id}.png", f"extracted_pan_{v_id}.png", f"extracted_boundary_{v_id}.png", f"extracted_annotation_{v_id}.png" will be saved. 

    '''

    import geopandas as gpd
    from functools import partial 
    from osgeo import gdal, ogr
    from multiprocessing import Pool
    import os 
    from cnnheights.preprocessing import extract_overlapping, divide_training_polygons
    import warnings 
    import numpy as np
    import geopandas as gpd 
    #warnings.filterwarnings("ignore")

    gdal.UseExceptions()
    ogr.UseExceptions()

    input_files = [os.path.join(input_data_dir, i) for i in os.listdir(input_data_dir)]
    
    area_files = np.sort([i for i in input_files if 'vector_rectangle' in i]) 
    annotation_files = np.sort([i for i in input_files if 'annotation' in i]) 
    raw_ndvi_images = np.sort([i for i in input_files if 'raw_ndvi' in i])
    raw_pan_images = np.sort([i for i in input_files if 'raw_pan' in i])

    total_jobs = len(area_files)
    n_jobs = np.min([os.cpu_count(), total_jobs])

    def preprocess_single(area_file:str): 

        # STEP 0: GET FILES

        local_dir = '/'.join(area_file.split('/')[:-1])+'/'
        v_id = area_file.split('_')[-1].split('.')[0]

        trainingArea = gpd.read_file(area_file)
        trainingPolygon = gpd.read_file(f'{local_dir}annotations_{v_id}.gpkg')

        trainingArea['id'] = range(trainingArea.shape[0])

        # STEP 1: EQUIVALENT OF ORIGINAL PREPROCESS SINGLE

        # areasWithPolygons contains the object polygons and weighted boundaries for each area
        # GET areasWithPolygons .... originally called like ```areasWithPolygons = divide_training_polygons(trainingPolygon, trainingArea, show_boundaries_during_processing=False)``` 

        cpTrainingPolygon = trainingPolygon.copy()
        areasWithPolygons = {}
        for i in trainingArea.index:
            spTemp = []
            allocated = []
            for j in cpTrainingPolygon.index:
                if trainingArea.loc[i]['geometry'].intersects(cpTrainingPolygon.loc[j]['geometry']):
                    spTemp.append(cpTrainingPolygon.loc[j])
                    allocated.append(j)

            # Order of bounds: minx miny maxx maxy
            
            # CALCULATE BOUNDARIES...function originally called under preprocess single via ```boundary = calculate_boundary_weight(spTemp, scale_polygon = 1.5, output_plot = False)```
            scale_polygon = 1.5

            # If there are polygons in a area, the boundary polygons return an empty geo dataframe
            if not spTemp:
                return gpd.GeoDataFrame({})

            tempPolygonDf = gpd.GeoDataFrame(spTemp)
            scaledPolygonDf = tempPolygonDf.scale(xfact=scale_polygon, yfact=scale_polygon, zfact=scale_polygon, origin='centroid')
            new_c = []

            length = len(scaledPolygonDf.index)
            counter = 0
            for i in range(length-1): 
                left = scaledPolygonDf.iloc[i]
                for j in range(i+1, length):
                    right = scaledPolygonDf.iloc[j]
                    left_intersection = left.intersection(right) # 
                    counter +=1 
                    if not left_intersection.is_empty: 
                        new_c.append(left_intersection)

            new_c = gpd.GeoSeries(new_c)
            new_cc = gpd.GeoDataFrame({'geometry': new_c})
            new_cc.columns = ['geometry']
            boundary = gpd.overlay(new_cc, tempPolygonDf, how='difference')

            #change multipolygon to polygon
            boundary = boundary.explode()
            boundary.reset_index(drop=True,inplace=True)

            areasWithPolygons[trainingArea.loc[i]['id']] = {'polygons':spTemp, 'boundaryWeight': boundary, 'bounds':list(trainingArea.bounds.loc[i]),}
            cpTrainingPolygon = cpTrainingPolygon.drop(allocated)
        
        # STEP TWO: EXTRACT ANNOTATIONS (originall second pooled step)

        #areasWithPolygons 

        r'''
        def extract_overlapping(inputImages, allAreasWithPolygons, writePath, bands, ndviFilename='extracted_ndvi', panFilename='extracted_pan', annotationFilename='extracted_annotation', boundaryFilename='extracted_boundary'):
            """
            Iterates over raw ndvi and pan images and using find_overlap() extract areas that overlap with training data. The overlapping areas in raw images are written in a separate file, and annotation and boundary file are created from polygons in the overlapping areas.
            Note that the intersection with the training areas is performed independently for raw ndvi and pan images. This is not an ideal solution and it can be combined in the future.
            
            old name used to be: extractAreasThatOverlapWithTrainingData
            
            """

            from cnnheights.utilities import find_overlap
            import os 
            import rasterio 

            if not os.path.exists(writePath):
                os.makedirs(writePath)
            
            for i in range(len(inputImages)): 
                input_images = inputImages[i]
                areasWithPolygons=allAreasWithPolygons[i]
                writeCounter=i 

                overlapppedAreas = set()                   
                ndviImg = rasterio.open(input_images[0])
                panImg = rasterio.open(input_images[1])

                ncndvi,imOverlapppedAreasNdvi = find_overlap(ndviImg, areasWithPolygons, writePath=writePath, imageFilename=[ndviFilename], annotationFilename=annotationFilename, boundaryFilename=boundaryFilename, bands=bands, writeCounter=writeCounter)
                ncpan, imOverlapppedAreasPan = find_overlap(panImg, areasWithPolygons, writePath=writePath, imageFilename=[panFilename], annotationFilename='', boundaryFilename='', bands=bands, writeCounter=writeCounter)
                if ncndvi != ncpan:
                    
                    print(ncndvi)
                    print(ncpan)  
                    raise Exception('Couldnt create mask!!!')

                if overlapppedAreas.intersection(imOverlapppedAreasNdvi):
                    print(f'Information: Training area(s) {overlapppedAreas.intersection(imOverlapppedAreasNdvi)} spans over multiple raw images. This is common and expected in many cases. A part was found to overlap with current input image.')
                overlapppedAreas.update(imOverlapppedAreasNdvi)
                
                allAreas = set(areasWithPolygons.keys())

                print(overlapppedAreas)
                print(allAreas)

                if allAreas.difference(overlapppedAreas):
                    print(f'Warning: Could not find a raw image corresponding to {allAreas.difference(overlapppedAreas)} areas. Make sure that you have provided the correct paths!')
        '''
        
def better_preprocess(input_data_dir:str, output_data_dir:str):
    
    r'''
    
    Arguments
    -----   

    input_data_dir : str
        - directory string with f"vector_rectangle_{v_id}.gpkg", f"annotations_{vid}.gpkg", f"raw_ndvi_{v_id}.tif", and f"raw_pan_{v_id}.tif" files

    output_data_dir : str
        - directory into which f"extracted_ndvi_{v_id}.png", f"extracted_pan_{v_id}.png", f"extracted_boundary_{v_id}.png", f"extracted_annotation_{v_id}.png" will be saved. 

    '''

    import geopandas as gpd
    from functools import partial 
    from osgeo import gdal, ogr
    from multiprocessing import Pool
    import os 
    from cnnheights.preprocessing import extract_overlapping, divide_training_polygons
    import warnings 
    import numpy as np
    #warnings.filterwarnings("ignore")
    import concurrent.futures
    import multiprocessing
    import sys
    import uuid

    def globalize(func):
        def result(*args, **kwargs):
            return func(*args, **kwargs)
        result.__name__ = result.__qualname__ = uuid.uuid4().hex
        setattr(sys.modules[result.__module__], result.__name__, result)
        return result

    gdal.UseExceptions()
    ogr.UseExceptions()

    input_files = [os.path.join(input_data_dir, i) for i in os.listdir(input_data_dir)]
    
    area_files = np.sort([i for i in input_files if 'vector_rectangle' in i]) 
    annotation_files = np.sort([i for i in input_files if 'annotation' in i]) 
    raw_ndvi_images = np.sort([i for i in input_files if 'raw_ndvi' in i])
    raw_pan_images = np.sort([i for i in input_files if 'raw_pan' in i])

    total_jobs = len(area_files)
    n_jobs = np.min([os.cpu_count(), total_jobs])
    
    @globalize
    def preprocess_single(area_file:str): 
        
        local_dir = '/'.join(area_file.split('/')[:-1])+'/'
        v_id = area_file.split('_')[-1].split('.')[0]

        trainingArea = gpd.read_file(area_file)
        trainingPolygon = gpd.read_file(f'{local_dir}annotations_{v_id}.gpkg')

        trainingArea['id'] = range(trainingArea.shape[0])
        
        # areasWithPolygons contains the object polygons and weighted boundaries for each area!
        areasWithPolygons = divide_training_polygons(trainingPolygon, trainingArea, show_boundaries_during_processing=False)
        
        return areasWithPolygons


    #print('checking if name is main')
    print(__name__)
    
    if __name__ != 'cnnheights.main':
        pool = Pool(processes=n_jobs)
        allAreasWithPolygons = pool.map(preprocess_single, area_files)

        print(allAreasWithPolygons)

        inputImages = list(zip(raw_ndvi_images,raw_pan_images))

        pool = Pool(processes=n_jobs)
        partial_func = partial(extract_overlapping, inputImages=inputImages, allAreasWithPolygons=allAreasWithPolygons, writePath=output_data_dir, bands=[0])
        pool.map(partial_func, range(total_jobs))

# CURRENT LOW LEVEL FUNCTIONS USED BY HIGH LEVEL PREPROCESS # 

def get_cutline_data(epsg:str=None, north=None, east=None, predictions=None, cutlines_shp:str='/Users/yaroslav/Documents/Work/NASA/data/jesse/thaddaeus_cutline/SSAr2_32628_GE01-QB02-WV02-WV03-WV04_PAN_NDVI_010_003_mosaic_cutlines.shp'):
    r'''
    
    return cutline information for an observation based on lat/long

    TO DO
    -----

    - need to improve it so it can work when we use multiple cutouts 
    - needs to get fixed to use the better lat/long format! 
    - needs better docstring
    - make the cutline path better?

    '''
    import geopandas as gpd
    import pandas as pd

    # 
    if predictions is not None:
        if type(predictions) is str: 
            predictions = gpd.read_file(predictions)
        epsg = str(predictions.crs).split(':')[-1]
        reference_poly = predictions['geometry'][0]
        c = reference_poly.centroid 
        east, north = (c.x, c.y)
        
    elif north is None or east is None or epsg is None:
        raise Exception('') 

    cutlines_gdf = gpd.read_file(cutlines_shp)
    cutlines_gdf = cutlines_gdf.set_crs(f'EPSG:{epsg}', allow_override=True)
    df = pd.DataFrame({'North': [north], 'East': [east]}) # LOL. Lat is N/S, Long is W/E
    point_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.East, df.North)).set_crs(f'EPSG:{epsg}', allow_override=True)

    polygons_contains = gpd.sjoin(cutlines_gdf, point_gdf, op='contains')

    labels = ['ACQDATE', 'OFF_NADIR', 'SUN_ELEV', 'SUN_AZ', 'SAT_ELEV', 'SAT_AZ']
    values = [polygons_contains[i].values[0] for i in labels]
    
    return dict(list(zip(labels, values)))

def image_normalize(im, axis = (0,1), c = 1e-8):
    r'''Normalize to zero mean and unit standard deviation along the given axis'''
    return (im - im.mean(axis)) / (im.std(axis) + c)

def extract_overlapping(inputImages, allAreasWithPolygons, writePath, bands, ndviFilename='extracted_ndvi', panFilename='extracted_pan', annotationFilename='extracted_annotation', boundaryFilename='extracted_boundary', write_counters:list=None):
    """
    Iterates over raw ndvi and pan images and using find_overlap() extract areas that overlap with training data. The overlapping areas in raw images are written in a separate file, and annotation and boundary file are created from polygons in the overlapping areas.
    Note that the intersection with the training areas is performed independently for raw ndvi and pan images. This is not an ideal solution and it can be combined in the future.
    
    old name used to be: extractAreasThatOverlapWithTrainingData
    
    """

    from cnnheights.preprocessing import find_overlap
    import os 
    import rasterio 

    if not os.path.exists(writePath):
        os.makedirs(writePath)
    
    for i in range(len(inputImages)): 
        input_images = inputImages[i]
        areasWithPolygons=allAreasWithPolygons[i]
        writeCounter=i 

        overlapppedAreas = set()                   
        ndviImg = rasterio.open(input_images[0])
        panImg = rasterio.open(input_images[1])

        # at this point, issue is not with pan/ndvi

        if write_counters is not None: 
            writeCounter = write_counters[i]
            
        ncndvi,imOverlapppedAreasNdvi = find_overlap(ndviImg, areasWithPolygons, writePath=writePath, imageFilename=[ndviFilename], annotationFilename=annotationFilename, boundaryFilename=boundaryFilename, bands=bands, writeCounter=writeCounter)
        ncpan, imOverlapppedAreasPan = find_overlap(panImg, areasWithPolygons, writePath=writePath, imageFilename=[panFilename], annotationFilename='', boundaryFilename='', bands=bands, writeCounter=writeCounter)
        if ncndvi != ncpan:
            
            print(ncndvi)
            print(ncpan)  
            raise Exception('Couldnt create mask!!!')

        if overlapppedAreas.intersection(imOverlapppedAreasNdvi):
            print(f'Information: Training area(s) {overlapppedAreas.intersection(imOverlapppedAreasNdvi)} spans over multiple raw images. This is common and expected in many cases. A part was found to overlap with current input image.')
        overlapppedAreas.update(imOverlapppedAreasNdvi)
        
        allAreas = set(areasWithPolygons.keys())

        ##print(overlapppedAreas)
        #print(allAreas)

        if allAreas.difference(overlapppedAreas):
            print(f'Warning: Could not find a raw image corresponding to {allAreas.difference(overlapppedAreas)} areas. Make sure that you have provided the correct paths!')

def divide_training_polygons(trainingPolygon, trainingArea, show_boundaries_during_processing:bool):
    '''
    As input we received two shapefile, first one contains the training areas/rectangles and other contains the polygon of trees/objects in those training areas
    The first task is to determine the parent training area for each polygon and generate a weight map based upon the distance of a polygon boundary to other objects.
    Weight map will be used by the weighted loss during the U-Net training

    I.E. Assign annotated ploygons in to the training areas.
    Note: older name was divide_polygons_in_training_areas and I think the even older name was dividePolygonsInTrainingAreas
   ''' 

    # For efficiency, assigned polygons are removed from the list, we make a copy here. 
    
    from cnnheights.preprocessing import calculate_boundary_weight
    
    cpTrainingPolygon = trainingPolygon.copy()
    splitPolygons = {}
    for i in trainingArea.index:
        spTemp = []
        allocated = []
        for j in cpTrainingPolygon.index:
            if trainingArea.loc[i]['geometry'].intersects(cpTrainingPolygon.loc[j]['geometry']):
                spTemp.append(cpTrainingPolygon.loc[j])
                allocated.append(j)

        # Order of bounds: minx miny maxx maxy
        boundary = calculate_boundary_weight(spTemp, scale_polygon = 1.5, output_plot = show_boundaries_during_processing)
        splitPolygons[trainingArea.loc[i]['id']] = {'polygons':spTemp, 'boundaryWeight': boundary, 'bounds':list(trainingArea.bounds.loc[i]),}
        cpTrainingPolygon = cpTrainingPolygon.drop(allocated)
    
    return splitPolygons

def find_overlap(img, areasWithPolygons, writePath, imageFilename, annotationFilename, boundaryFilename, bands, writeCounter):
    """
    Finds overlap of image with a training area.
    Use write_extracted() to write the overlapping training area and corresponding polygons in separate image files.
    """

    from rasterio.mask import mask
    import rasterio 
    import geopandas as gpd
    from shapely.geometry import box
    from cnnheights.preprocessing import write_extracted
    
    overlapppedAreas = set() # small question, but why set? 
    #print(areasWithPolygons)
    #print('about to look into finding overlap')
    for areaID, areaInfo in areasWithPolygons.items():
        # seems like issue happens before this? 
        #Convert the polygons in the area in a dataframe and get the bounds of the area. 
        polygonsInAreaDf = gpd.GeoDataFrame(areaInfo['polygons'])
        boundariesInAreaDf = gpd.GeoDataFrame(areaInfo['boundaryWeight'])    
        bboxArea = box(*areaInfo['bounds'])
        bboxImg = box(*img.bounds)

        #print(bboxArea)
        #print(bboxImg)

        #Extract the window if area is in the image
        if(bboxArea.intersects(bboxImg)):
            #print('intersects')
            profile = img.profile  
            sm = mask(img, [bboxArea], all_touched=True, crop=True )
            profile['height'] = sm[0].shape[1]
            profile['width'] = sm[0].shape[2]
            # issue exists before this
            #print(profile['height'], profile['width'])
            profile['transform'] = sm[1] 
            # That's a problem with rasterio, if the height and the width are less then 256 it throws: ValueError: blockysize exceeds raster height 
            # So I set the blockxsize and blockysize to prevent this problem
            profile['blockxsize'] = 32
            profile['blockysize'] = 32
            profile['count'] = 1
            profile['dtype'] = rasterio.float32 # rasterio.float32 # THIS GOT CHANGED! CHANGE BACK? or unint32?? # maybe fix this
            # write_extracted writes the image, annotation and boundaries and returns the counter of the next file to write. 
            writeCounter = write_extracted(img, sm, profile, polygonsInAreaDf, boundariesInAreaDf, writePath, imageFilename, annotationFilename, boundaryFilename, bands, writeCounter)
            overlapppedAreas.add(areaID)

        else: 
            print('not in area??')

    return(writeCounter, overlapppedAreas)

def write_extracted(img, sm, profile, polygonsInAreaDf, boundariesInAreaDf, writePath, imagesFilename, annotationFilename, boundaryFilename, bands, writeCounter, normalize=False):
    """
    Write the part of raw image that overlaps with a training area into a separate image file. 
    Use rowColPolygons to create and write annotation and boundary image from polygons in the training area.
    Note: original name was: writeExtractedImageAndAnnotation

    To Do: remove img from args because it's not used? will need to make chagnes to other files that reference it. 

    set normalize to False for Jesse 

    """

    import os 
    import rasterio 
    from cnnheights.preprocessing import row_col_polygons
    import numpy as np
    #print('about to try to write extracted')
    try:
        for band, imFn in zip(bands, imagesFilename):
            # Rasterio reads file channel first, so the sm[0] has the shape [1 or ch_count, x,y]
            # If raster has multiple channels, then bands will be [0, 1, ...] otherwise simply [0] # can i remove this? 
            dt = sm[0][band].astype(profile['dtype'])

            if normalize: # Note: If the raster contains None values, then you should normalize it separately by calculating the mean and std without those values.
                dt = image_normalize(dt, axis=None) #  Normalize the image along the width and height, and since here we only have one channel we pass axis as None # FIX THIS!
            with rasterio.open(os.path.join(writePath, imFn+'_{}.png'.format(writeCounter)), 'w', **profile) as dst:
                    dst.write(dt, 1) 
        
        if annotationFilename:
            annotation_filepath = os.path.join(writePath,annotationFilename+'_{}.png'.format(writeCounter))
            # The object is given a value of 1, the outline or the border of the object is given a value of 0 and rest of the image/background is given a a value of 0
            row_col_polygons(polygonsInAreaDf,(sm[0].shape[1], sm[0].shape[2]), profile, annotation_filepath, outline=0, fill = 1)
        if boundaryFilename:
            boundary_filepath = os.path.join(writePath,boundaryFilename+'_{}.png'.format(writeCounter))
            # The boundaries are given a value of 1, the outline or the border of the boundaries is also given a value of 1 and rest is given a value of 0
            row_col_polygons(boundariesInAreaDf,(sm[0].shape[1], sm[0].shape[2]), profile, boundary_filepath, outline=1 , fill=1)
        return(writeCounter+1)
    except Exception as e:
        print(e)
        print("Something nasty happened, could not write the annotation or the mask file!")
        return writeCounter

def calculate_boundary_weight(polygonsInArea, scale_polygon = 1.5, output_plot = False): 
    '''
    For each polygon, create a weighted boundary where the weights of shared/close boundaries is higher than weights of solitary boundaries.
   
    I.E. Create boundary from polygon file

    Note: this is the improved version, that scales all the polygons *once* by scaling the gdf by centroid (as opposed to scaling from some cooridinate within the gdf)
    '''
    
    import geopandas as gpd 
    import matplotlib.pyplot as plt 

    # If there are polygons in a area, the boundary polygons return an empty geo dataframe
    if not polygonsInArea:
        return gpd.GeoDataFrame({})

    tempPolygonDf = gpd.GeoDataFrame(polygonsInArea)
    scaledPolygonDf = tempPolygonDf.scale(xfact=scale_polygon, yfact=scale_polygon, zfact=scale_polygon, origin='centroid')
    new_c = []

    length = len(scaledPolygonDf.index)
    counter = 0
    for i in range(length-1): 
        left = scaledPolygonDf.iloc[i]
        for j in range(i+1, length):
            right = scaledPolygonDf.iloc[j]
            left_intersection = left.intersection(right) # 
            counter +=1 
            if not left_intersection.is_empty: 
                new_c.append(left_intersection)

    new_c = gpd.GeoSeries(new_c)
    new_cc = gpd.GeoDataFrame({'geometry': new_c})
    new_cc.columns = ['geometry']
    bounda = gpd.overlay(new_cc, tempPolygonDf, how='difference')
    
    if output_plot:
        import random 
        fig, ax = plt.subplots()
        bounda.plot(ax=ax, color = 'red')
        ax.set(xlabel='Longitude', ylabel='Latitude')
        plt.savefig(f'temp_{random.randint(1,10000)}.png', dpi=150)

    #change multipolygon to polygon
    bounda = bounda.explode()
    bounda.reset_index(drop=True,inplace=True)
    #bounda.to_file('boundary_ready_to_use.shp')
    return bounda

def row_col_polygons(areaDf, areaShape, profile, filename, outline, fill):
    """
    Convert polygons coordinates to image pixel coordinates, create annotation image using drawPolygons() and write the results into an image file.
    """

    import rasterio 
    from cnnheights.preprocessing import draw_polygons

    transform = profile['transform']
    polygons = []
    for i in areaDf.index:
        gm = areaDf.loc[i]['geometry']
        a,b = zip(*list(gm.exterior.coords))
        row, col = rasterio.transform.rowcol(transform, a, b)
        zipped = list(zip(row,col)) #[list(rc) for rc in list(zip(row,col))]
        polygons.append(zipped)
    #with open(filename, 'w') as outfile:  
    #    json.dump({'Trees': polygons}, outfile)
    mask = draw_polygons(polygons, areaShape, outline=outline, fill=fill)    
    profile['dtype'] = rasterio.int16
    with rasterio.open(filename, 'w', **profile) as dst:
        dst.write(mask.astype(rasterio.int16), 1)

def draw_polygons(polygons, shape, outline, fill):
    """
    From the polygons, create a numpy mask with fill value in the foreground and 0 value in the background.
    Outline (i.e the edge of the polygon) can be assigned a separate value.
    """

    from PIL import Image, ImageDraw
    import numpy as np 

    mask = np.zeros(shape, dtype=np.uint8)
    mask = Image.fromarray(mask)
    draw = ImageDraw.Draw(mask)
    #Syntax: PIL.ImageDraw.Draw.polygon(xy, fill=None, outline=None)
    #Parameters:
    #xy – Sequence of either 2-tuples like [(x, y), (x, y), …] or numeric values like [x, y, x, y, …].
    #outline – Color to use for the outline.
    #fill – Color to use for the fill.
    #Returns: An Image object.
    for polygon in polygons:
        xy = [(point[1], point[0]) for point in polygon]
        draw.polygon(xy=xy, outline=outline, fill=fill)
    mask = np.array(mask)#, dtype=bool)   

    return (mask) # why is this a tuple? 