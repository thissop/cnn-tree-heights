# CURRENT HIGH LEVEL PREPROCESS FUNCTIONS # 

def sample_background(input_tif:str, output_dir:str, counter:int=0, plot_path:str=None, sample_dim:tuple=(1024,1024), dpi=350):
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

    output_dir = os.path.join(output_dir, f'cutout_{counter}')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    x1, y1 = (np.random.randint(0,dims[0]-sample_dim[0]), np.random.randint(0,dims[1]-sample_dim[1])) ## FIXX!! 

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

    output_tif_path = os.path.join(output_dir, f'cutout_{counter}.tif')

    with rasterio.open(output_tif_path, 'w',
            driver='GTiff', width=sample_dim[0], height=sample_dim[1], count=2,
            dtype=img.dtype, crs=dataset.crs, transform=window_transform) as new_data_set:
        new_data_set.write(img)#, indexes=2)

    output_tif = rasterio.open(output_tif_path)
    crs = output_tif.crs.to_string()
    bounds = output_tif.bounds

    gdf = gpd.GeoDataFrame({"geometry":[box(*bounds)]})
    vector_rect_path = os.path.join(output_dir, f'vector_boundary_{counter}.gpkg')
    gdf.to_file(vector_rect_path, crs=crs, driver="GPKG")

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

    output_ndvi = os.path.join(output_dir, f'ndvi_{counter}.tif')
    output_pan = os.path.join(output_dir, f'pan_{counter}.tif')

    cutout_raster = rasterio.open(output_tif_path)
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
    import numpy as np
    import os 

    #warnings.filterwarnings("ignore")

    #for f in os.listdir(output_data_dir): 
    #    if 'gpkg-shm' or 'gpkg-wal' in f: 
    #        os.remove(os.path.join(output_data_dir, f))

    for f in os.listdir(input_data_dir):
        if 'aux.xml' in f: 
            os.remove(os.path.join(input_data_dir, f))

    input_files = [os.path.join(input_data_dir, i) for i in os.listdir(input_data_dir)]

    _ = [os.remove(i) for i in input_files if '-shm' in i or '-wal' in i]
    input_files = [i for i in input_files if '-shm' not in i and '-wal' not in i]

    area_files = np.sort([i for i in input_files if 'vector_boundary' in i]) 
    annotation_files = np.sort([i for i in input_files if 'vector_annotation' in i]) 
    raw_ndvi_images = np.sort([i for i in input_files if 'ndvi' in i]) 
    raw_pan_images = np.sort([i for i in input_files if 'pan' in i]) 

    allAreasWithPolygons = [] 

    print([len(i) for i in (area_files, annotation_files, raw_ndvi_images, raw_pan_images)]) 
    write_counters = []
    for i in range(len(area_files)): 
        print(area_files[i])
        trainingArea = gps.read_file(area_files[i])
        trainingPolygon = gps.read_file(annotation_files[i])

        print(trainingPolygon)

        #trainingPolygon['is_valid'] = trainingPolygon['geometry'].is_valid
        #trainingPolygon = trainingPolygon[trainingPolygon['is_valid']]
        #trainingPolygon.to_file(annotation_files[i])

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
    extract_overlapping(inputImages, allAreasWithPolygons=allAreasWithPolygons, writePath=output_data_dir, 
                                                annotationFilename='raster_annotation',
                                                boundaryFilename='raster_boundary', bands=[0], write_counters=write_counters)

    #for f in os.listdir(output_data_dir): 
    #    if 'gpkg-shm' or 'gpkg-wal' in f: 
    #        os.remove(os.path.join(output_data_dir, f))

# CURRENT LOW LEVEL FUNCTIONS USED BY HIGH LEVEL PREPROCESS # 

def get_cutline_data(cutlines_shp:str, epsg:str=None, north=None, east=None, predictions=None):
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
        
        epsg = f'{predictions.crs.to_epsg()}'
            
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

def extract_overlapping(inputImages, allAreasWithPolygons, writePath, bands, annotationFilename='extracted_annotation', boundaryFilename='extracted_boundary', write_counters:list=None):
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
            
        ncndvi,imOverlapppedAreasNdvi = find_overlap(ndviImg, areasWithPolygons, writePath=writePath, imageFilename=['ndvi'], annotationFilename=annotationFilename, boundaryFilename=boundaryFilename, bands=bands, writeCounter=writeCounter)
        ncpan, imOverlapppedAreasPan = find_overlap(panImg, areasWithPolygons, writePath=writePath, imageFilename=['pan'], annotationFilename='', boundaryFilename='', bands=bands, writeCounter=writeCounter)
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

def write_extracted(img, sm, profile, polygonsInAreaDf, boundariesInAreaDf, writePath, imagesFilename, annotationFilename, boundaryFilename, bands, writeCounter, normalize=False, copy_images:bool=False):
    """
    Write the part of raw image that overlaps with a training area into a separate image file. 
    Use rowColPolygons to create and write annotation and boundary image from polygons in the training area.
    Note: original name was: writeExtractedImageAndAnnotation

    To Do: remove img from args because it's not used? will need to make chagnes to other files that reference it. 

    set normalize to False for Jesse 

    UPDATE: Not going to save png copies of images for now (will save for annotation and bounadry, because these are going from vector to png)

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
            
            
            if copy_images: 
                with rasterio.open(os.path.join(writePath, imFn+'_{}.png'.format(writeCounter)), 'w', **profile) as dst:
                        dst.write(dt, 1) 
        
        if annotationFilename:
            annotation_filepath = os.path.join(writePath,annotationFilename+'_{}.tiff'.format(writeCounter))
            # The object is given a value of 1, the outline or the border of the object is given a value of 0 and rest of the image/background is given a a value of 0
            row_col_polygons(polygonsInAreaDf,(sm[0].shape[1], sm[0].shape[2]), profile, annotation_filepath, outline=0, fill = 1)
        if boundaryFilename:
            boundary_filepath = os.path.join(writePath,boundaryFilename+'_{}.tiff'.format(writeCounter))
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