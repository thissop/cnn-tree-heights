def height_from_shadow(shadow:float, time:str, lat:float, lon:float): 
    r'''
    _Calculate height of tree from shadow length, time, and location_  

    Parameters
    ----------      

    shadow : `float`
        Shadow length in meters

    time : `str`
        Time in `"yyyy-mm-dd hh:mm:ss"` format

    lat : `float`

    lon : `float`

    Returns
    -------

    height : `float`
        The tree's height in meters. 
    '''
    
    
    from pvlib.location import Location
    import numpy as np

    site = Location(latitude=lat, longitude=lon)#, 'Etc/GMT+1') # latitude, longitude, time_zone, altitude, name
    zenith = 180-float(site.get_solarposition(time)['zenith']) # correct? 

    height = shadow/np.tan(np.radians(zenith)) # does this need to get corrected for time zone? 

    return height

def image_normalize(im, axis = (0,1), c = 1e-8):
    r'''Normalize to zero mean and unit standard deviation along the given axis'''
    return (im - im.mean(axis)) / (im.std(axis) + c)

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