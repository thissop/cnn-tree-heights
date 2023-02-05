def height_from_shadow(shadow_length:float, zenith_angle:float):
    r'''
    _get height from shadow length and zenith angle_


    TO DO 
    -----

    - need to give user information about what way zenith angle should be oriented?
    
    '''
    import numpy as np
    
    height = shadow_length*np.tan(np.radians(zenith_angle)) # does this need to get corrected for time zone? 
     # H = L tan (x), where x is solar elevation angle from ground? 
    return height 

def zenith_from_location(time:str, lat:float, lon:float): 
    r'''
    _get zenith angle for calculations based on time and location_  

    Parameters
    ----------      

    time : `str`
        Time in `"yyyy-mm-dd hh:mm:ss"` format

    lat : `float`

    lon : `float`

    Returns
    -------

    zenith : `float`
        Angle of sun's zenith.


    TO DO 
    -----
    - make it get azimuth to!
     
    '''
    
    
    from pvlib.location import Location
    import numpy as np

    site = Location(latitude=lat, longitude=lon)#, 'Etc/GMT+1') # latitude, longitude, time_zone, altitude, name
    zenith = 180-float(site.get_solarposition(time)['zenith']) # correct? 

    return zenith

def shadows_from_annotations(annotations_gpkg, cutlines_shp:str, north:float, east:float, epsg:str, save_path:str=None, d:float=3):
    r'''
    _get shadow lengths and heights from annotations gpkg file, coordinate, and cutfile_

    Arguments 
    ---------

    annotations_gpkg : `str`
        path to annotations gpkg file 

    NOTES 
    -----

    - get lat/long from center ish of the 

    - uses lat/long to get container in cutlines file to reference to get azimuth angle from. 
    - work with feather files (my preference) for saving these (e.g. save_path='./data.feather')...I love flexibility of .to_file(...)

    TO DO 
    -----
    - fix lat long stuff?
    - d is in meters because it's in the UTM projection? something to mention in paper how we update it for different regions (because inaccurate outside itself)

    ''' 

    import geopandas as gpd 
    import numpy as np
    from shapely.geometry import LineString, box

    annotations_gdf = gpd.read_file(annotations_gpkg)
    annotations_gdf = annotations_gdf[annotations_gdf.geom_type != 'MultiPolygon']

    annotations_gdf = annotations_gdf.set_crs(f'EPSG:{epsg}', allow_override=True)
    print(annotations_gdf.index)
    annotations_gdf.set_index = list(range(len(annotations_gdf.index)))
    print(annotations_gdf)
    quit()
    centroids = annotations_gdf.centroid    

    cutline_info = get_cutline_data(north=north, east=east, epsg=epsg, cutlines_shp=cutlines_shp)

    dy = np.abs(d/np.tan(np.radians(cutline_info['SUN_AZ'])))
    lines = [LineString([(x-d, y+dy), (x+d, y-dy)]) for x, y in zip(centroids.x, centroids.y)]
    lines_gdf = gpd.GeoDataFrame({'geometry':lines}, geometry='geometry', crs=annotations_gdf.crs)

    shadow_lines = lines_gdf.intersection(annotations_gdf, align=False)
    shadow_lengths = shadow_lines.length

    heights = height_from_shadow(shadow_lengths, zenith_angle=cutline_info['SUN_ELEV'])
    print(cutline_info['SUN_ELEV'])

    bounds = annotations_gdf.bounds
    dx = np.abs(np.abs(bounds['maxx'])-np.abs(bounds['minx']))
    dy = np.abs(np.abs(bounds['maxy'])-np.abs(bounds['miny']))
    dxy = np.max(np.array([dx,dy]).T, axis=1)
    square_bounds = np.array([[minx, miny, minx+diff, miny+diff] for minx, miny, diff in zip(bounds['minx'], bounds['miny'], dxy)])    

    d = {'shadow_geometry':annotations_gdf['geometry'], 
         'centroids':centroids,
         'bounds_geometry':[box(*i) for i in square_bounds],
         'heights':heights, 
         'line_geometries':shadow_lines, 
         'lengths':shadow_lengths}
    
    shadows_gdf = gpd.GeoDataFrame(d, crs=f'EPSG:{epsg}')

    if save_path is not None: 
        shadows_gdf.to_file(save_path)

    return shadows_gdf

## PREPROCESS UTILITIES ## 

def sample_background(input_tif:str, output_path:str, crs:str, key:str=None, counter:int=0, plot_path:str=None, sample_dim:tuple=(1056,1056), dpi=350):
    r'''

    TO DO 
    -----

    Notes
    -----
    - output_path is used for output_tif and output vector rectangle.
    - note for jesse: opening from this window saves crazy memory. 

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
    from blend_modes import lighten_only

    dataset = rasterio.open(input_tif) 

    dims = (dataset.width, dataset.height) 

    extant_key = False 
    if key is not None and os.path.exists(key):
        extant_key = True 

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

        extracted_polygon = Polygon(((x1,y1), (x2,y1), (x2, y2), (x1,y2))) 

        if extant_key: 
            key_gdf = gpd.read_file(key)
            extracted_geometries = key_gdf['geometry']

            overlapping = extracted_geometries.overlaps(extracted_polygon)[0]

        else: 
            overlapping = False      
            
    centroid = [np.median(extent[0:2]), np.median(extent[3:5])]

    if extant_key: 
        key_gdf = gpd.read_file(key)
        d = {'geometry':[extracted_polygon]+key_gdf['geometry'].to_list(),
             'centroidsx':[centroid[0]]+key_gdf['centroidsx'].to_list(), 
             'centroidsy':[centroid[1]]+key_gdf['centroidsy'].to_list()}

        key_gdf = gpd.GeoDataFrame(d, crs=crs)

    else: 
        d = {'geometry':[extracted_polygon], 'centroidsx':[centroid[0]], 'centroidsy':[centroid[1]]}
        key_gdf = gpd.GeoDataFrame(d, crs=crs) #lines_gdf = gpd.GeoDataFrame({'geometry':lines}, geometry='geometry', crs=annotations_gdf.crs)
        print(key_gdf)

    if plot_path is not None: 
        # load img here using rasterio
        fig, ax = plt.subplots(figsize=(5,5))

        plot.show(img[0], origin='upper', transform=window_transform, extent=extent, interpolation=None, ax=ax, vmin=250, vmax=750, cmap='Greys_r')
        plot.show(img[1], origin='upper', transform=window_transform, extent=extent, interpolation=None, cmap='Reds', ax=ax, alpha=0.5)
        
        #blended = lighten_only(np.array([img[0].astype(float)]), np.array([img[1].astype(float)]), opacity=0.5)
        #ax.plot(blended)
        
        ax.set_aspect('equal', 'box')
        
        save_fig(plot_path, dpi)

    output_tif = os.path.join(output_path, f'cutout_{counter}.tiff')

    with rasterio.open(output_tif, 'w',
            driver='GTiff', width=sample_dim[0], height=sample_dim[1], count=2,
            dtype=img.dtype, crs=dataset.crs, transform=window_transform) as new_data_set:
        new_data_set.write(img)#, indexes=2)

    vector_rect_path = os.path.join(output_path, f'vector_rectangle_{counter}.gpkg')
    vector_rectangle = gpd.GeoDataFrame({'geometry':[extracted_polygon]}, crs=crs)
    vector_rectangle.to_file(vector_rect_path, driver='GPKG')#, layer='name')

    return key_gdf

## HEIGHTS UTILITIES ##
def get_cutline_data(north:float, east:float, epsg:str, cutlines_shp:str='/Users/yaroslav/Documents/Work/NASA/data/jesse/thaddaeus_cutline/SSAr2_32628_GE01-QB02-WV02-WV03-WV04_PAN_NDVI_010_003_mosaic_cutlines.shp'):
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

## PREPROCESSING UTILITIES ##

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
        if allAreas.difference(overlapppedAreas):
            print(f'Warning: Could not find a raw image corresponding to {allAreas.difference(overlapppedAreas)} areas. Make sure that you have provided the correct paths!')

def divide_training_polygons(trainingPolygon, trainingArea, show_boundaries_during_processing:bool):
    '''
    As input we received two shapefile, first one contains the training areas/rectangles and other contains the polygon of trees/objects in those training areas
    The first task is to determine the parent training area for each polygon and generate a weight map based upon the distance of a polygon boundary to other objects.
    Weight map will be used by the weighted loss during the U-Net training

    I.E. Assign annotated ploygons in to the training areas.
    Note: older name was divide_polygons_in_training_areas
   ''' 

    # For efficiency, assigned polygons are removed from the list, we make a copy here. 
    
    from cnnheights.utilities import calculate_boundary_weight
    
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
    import geopandas as gps
    from shapely.geometry import box
    from cnnheights.utilities import write_extracted
    
    overlapppedAreas = set() # small question, but why set? 
    #print(areasWithPolygons)
    
    for areaID, areaInfo in areasWithPolygons.items():
        #Convert the polygons in the area in a dataframe and get the bounds of the area. 
        polygonsInAreaDf = gps.GeoDataFrame(areaInfo['polygons'])
        boundariesInAreaDf = gps.GeoDataFrame(areaInfo['boundaryWeight'])    
        bboxArea = box(*areaInfo['bounds'])
        bboxImg = box(*img.bounds)
        #Extract the window if area is in the image
        if(bboxArea.intersects(bboxImg)):
            profile = img.profile  
            sm = mask(img, [bboxArea], all_touched=True, crop=True )
            profile['height'] = sm[0].shape[1]
            profile['width'] = sm[0].shape[2]
            profile['transform'] = sm[1] 
            # That's a problem with rasterio, if the height and the width are less then 256 it throws: ValueError: blockysize exceeds raster height 
            # So I set the blockxsize and blockysize to prevent this problem
            profile['blockxsize'] = 32
            profile['blockysize'] = 32
            profile['count'] = 1
            profile['dtype'] = rasterio.float32
            # write_extracted writes the image, annotation and boundaries and returns the counter of the next file to write. 
            writeCounter = write_extracted(img, sm, profile, polygonsInAreaDf, boundariesInAreaDf, writePath, imageFilename, annotationFilename, boundaryFilename, bands, writeCounter)
            overlapppedAreas.add(areaID)

    return(writeCounter, overlapppedAreas)

def write_extracted(img, sm, profile, polygonsInAreaDf, boundariesInAreaDf, writePath, imagesFilename, annotationFilename, boundaryFilename, bands, writeCounter, normalize=True):
    """
    Write the part of raw image that overlaps with a training area into a separate image file. 
    Use rowColPolygons to create and write annotation and boundary image from polygons in the training area.
    Note: original name was: writeExtractedImageAndAnnotation
    """

    import os 
    import rasterio 
    from cnnheights.utilities import row_col_polygons

    try:
        for band, imFn in zip(bands, imagesFilename):
            # Rasterio reads file channel first, so the sm[0] has the shape [1 or ch_count, x,y]
            # If raster has multiple channels, then bands will be [0, 1, ...] otherwise simply [0]
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
    
    import geopandas as gps 
    import matplotlib.pyplot as plt 

    # If there are polygons in a area, the boundary polygons return an empty geo dataframe
    if not polygonsInArea:
        return gps.GeoDataFrame({})

    tempPolygonDf = gps.GeoDataFrame(polygonsInArea)
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

    new_c = gps.GeoSeries(new_c)
    new_cc = gps.GeoDataFrame({'geometry': new_c})
    new_cc.columns = ['geometry']
    bounda = gps.overlay(new_cc, tempPolygonDf, how='difference')
    
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
    from cnnheights.utilities import draw_polygons

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

## TRAINING UTILITIES ## 

def load_train_test(ndvi_images:list,
                    pan_images:list, 
                    annotations:list,
                    boundaries:list,
                    logging_dir:str=None,
                    normalize:float = 0.4, BATCH_SIZE = 8, patch_size=(256,256,4), 
                    input_shape = (256,256,2), input_image_channel = [0,1], input_label_channel = [2], input_weight_channel = [3]):
    
    r'''
    
    Arguments 
    ---------

    area_files : list 
        List of the area files 

    annotations : list
        List of the full file paths to the extracted annotations that got outputed by the earlier preproccessing step. 

    ndvi_images : list 
        List of full file paths to the extracted ndvi images 

    pan_images : list 
        Same as ndvi_images except for pan 

    boundaries : list
        List of boundary files extracted by previous preprocessing step 

    logging_dir : str
        the directory all the logging stuff should be saved into. defaults to none, which will make all the directories in directory that the python file that executes this function is run in. 
    
    '''

    import os
    import rasterio 
    import numpy as np
    from PIL import Image
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    from cnnheights.original_core.frame_utilities import FrameInfo, split_dataset
    from cnnheights.original_core.dataset_generator import DataGenerator

    if logging_dir is not None: 
        patch_dir = os.path.join(logging_dir, f'patches{patch_size[0]}/')
    else: 
        patch_dir = './patches{}'.format(patch_size[0])
    
    frames_json = os.path.join(patch_dir,'frames_list.json')

    # Read all images/frames into memory
    frames = []

    for i in range(len(ndvi_images)):
        ndvi_img = rasterio.open(ndvi_images[i])
        pan_img = rasterio.open(pan_images[i])
        read_ndvi_img = ndvi_img.read()
        read_pan_img = pan_img.read()
        comb_img = np.concatenate((read_ndvi_img, read_pan_img), axis=0)
        comb_img = np.transpose(comb_img, axes=(1,2,0)) #Channel at the end
        annotation_im = Image.open(annotations[i])
        annotation = np.array(annotation_im)
        weight_im = Image.open(boundaries[i])
        weight = np.array(weight_im)
        f = FrameInfo(comb_img, annotation, weight)
        frames.append(f)
    
    training_frames, validation_frames, testing_frames  = split_dataset(frames, frames_json, patch_dir)

    annotation_channels = input_label_channel + input_weight_channel
    train_generator = DataGenerator(input_image_channel, patch_size, training_frames, frames, annotation_channels, augmenter = 'iaa').random_generator(BATCH_SIZE, normalize = normalize)
    val_generator = DataGenerator(input_image_channel, patch_size, validation_frames, frames, annotation_channels, augmenter= None).random_generator(BATCH_SIZE, normalize = normalize)
    test_generator = DataGenerator(input_image_channel, patch_size, testing_frames, frames, annotation_channels, augmenter= None).random_generator(BATCH_SIZE, normalize = normalize)

    return train_generator, val_generator, test_generator

def train_model(train_generator, val_generator, 
                BATCH_SIZE = 8, NB_EPOCHS = 21, VALID_IMG_COUNT = 1, MAX_TRAIN_STEPS = 500, # NB_EPOCHS=200, MAX_TRAIN_STEPS=1000
                input_shape = (256,256,2), input_image_channel = [0,1], input_label_channel = [2], input_weight_channel = [3], 
                logging_dir:str=None, 
                model_path = './src/monthly/jan2023/library-testing/cnn-training-output/saved_models/UNet/'): 
    from cnnheights.original_core.losses import tversky, accuracy, dice_coef, dice_loss, specificity, sensitivity 
    from cnnheights.original_core.optimizers import adaDelta 
    import time 
    from functools import reduce 
    from cnnheights.original_core.UNet import UNet 
    from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
    import os 

    OPTIMIZER = adaDelta
    LOSS = tversky 

    # Only for the name of the model in the very end
    OPTIMIZER_NAME = 'AdaDelta'
    LOSS_NAME = 'weightmap_tversky'

    # Declare the path to the final model
    # If you want to retrain an exising model then change the cell where model is declared. 
    # This path is for storing a model after training.

    timestr = time.strftime("%Y%m%d-%H%M")
    chf = input_image_channel + input_label_channel
    chs = reduce(lambda a,b: a+str(b), chf, '')

    if logging_dir is not None: 
        model_dir= os.path.join(logging_dir, 'saved_models/UNet/')
        tensorboard_log_dir = os.path.join(logging_dir, 'logs/')

    else: 
        model_dir = './saved_models/UNet/'
        tensorboard_log_dir = './logs'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(tensorboard_log_dir): 
        os.mkdir(tensorboard_log_dir)

    model_path = os.path.join(model_dir,'trees_{}_{}_{}_{}_{}.h5'.format(timestr,OPTIMIZER_NAME,LOSS_NAME,chs,input_shape[0]))

    # The weights without the model architecture can also be saved. Just saving the weights is more efficent.

    # weight_path="./saved_weights/UNet/{}/".format(timestr)
    # if not os.path.exists(weight_path):
    #     os.makedirs(weight_path)
    # weight_path=weight_path + "{}_weights.best.hdf5".format('UNet_model')
    # print(weight_path)

    # Define the model and compile it
    print('\n')
    print([BATCH_SIZE, *input_shape])
    print('\n')
    model = UNet([BATCH_SIZE, *input_shape], input_label_channel) # *config.input_shape had asterisk originally?
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=[dice_coef, dice_loss, specificity, sensitivity, accuracy])

    # Define callbacks for the early stopping of training, LearningRateScheduler and model checkpointing

    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, 
                                save_best_only=True, mode='min', save_weights_only = False)

    #reduceonplatea; It can be useful when using adam as optimizer
    #Reduce learning rate when a metric has stopped improving (after some patience,reduce by a factor of 0.33, new_lr = lr * factor).
    #cooldown: number of epochs to wait before resuming normal operation after lr has been reduced.
    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.33,
                                    patience=4, verbose=1, mode='min',
                                    min_delta=0.0001, cooldown=4, min_lr=1e-16)

    #early = EarlyStopping(monitor="val_loss", mode="min", verbose=2, patience=15)



    tensorboard_log_path = os.path.join(tensorboard_log_dir,'UNet_{}_{}_{}_{}_{}'.format(timestr,OPTIMIZER_NAME,LOSS_NAME, chs, input_shape[0]))
    tensorboard = TensorBoard(log_dir=tensorboard_log_path, histogram_freq=0, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

    callbacks_list = [checkpoint, tensorboard] #reduceLROnPlat is not required with adaDelta

    # do training  

    # last line: PROBLEMS START HERE!

    import time 

    s = time.time()

    loss_history = [model.fit(train_generator, 
                            steps_per_epoch=MAX_TRAIN_STEPS, 
                            epochs=NB_EPOCHS, 
                            validation_data=val_generator,
                            validation_steps=VALID_IMG_COUNT,
                            callbacks=callbacks_list, workers=1, use_multiprocessing=True)] # the generator is not very thread safe

    print('time elapsed', time.time()-s, '(s)')

    return model, loss_history[0].history