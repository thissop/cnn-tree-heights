import os 
import rasterio 
import numpy as np
from rasterio.mask import mask
import rasterio 
import geopandas as gpd
from shapely.geometry import box
import rasterio 
import matplotlib.pyplot as plt
from functools import partial 
from osgeo import gdal, ogr
from multiprocessing import Pool
import os 
import warnings 
#warnings.filterwarnings("ignore")
import multiprocessing
import sys
import uuid
from PIL import Image, ImageDraw

# FUNCTION DEFS TO KEEP INTERNAL

def image_normalize(im, axis = (0,1), c = 1e-8):
    r'''Normalize to zero mean and unit standard deviation along the given axis'''
    return (im - im.mean(axis)) / (im.std(axis) + c)

def extract_overlapping(inputImages, allAreasWithPolygons, writePath, bands, ndviFilename='extracted_ndvi', panFilename='extracted_pan', annotationFilename='extracted_annotation', boundaryFilename='extracted_boundary'):
    """
    Iterates over raw ndvi and pan images and using find_overlap() extract areas that overlap with training data. The overlapping areas in raw images are written in a separate file, and annotation and boundary file are created from polygons in the overlapping areas.
    Note that the intersection with the training areas is performed independently for raw ndvi and pan images. This is not an ideal solution and it can be combined in the future.
    
    old name used to be: extractAreasThatOverlapWithTrainingData
    
    """

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

def divide_training_polygons(trainingPolygon, trainingArea, show_boundaries_during_processing:bool):
    '''
    As input we received two shapefile, first one contains the training areas/rectangles and other contains the polygon of trees/objects in those training areas
    The first task is to determine the parent training area for each polygon and generate a weight map based upon the distance of a polygon boundary to other objects.
    Weight map will be used by the weighted loss during the U-Net training

    I.E. Assign annotated ploygons in to the training areas.
    Note: older name was divide_polygons_in_training_areas and I think the even older name was dividePolygonsInTrainingAreas
   ''' 

    # For efficiency, assigned polygons are removed from the list, we make a copy here. 
        
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
    
    overlapppedAreas = set() # small question, but why set? 
    #print(areasWithPolygons)
    print('about to look into finding overlap')
    for areaID, areaInfo in areasWithPolygons.items():
        #Convert the polygons in the area in a dataframe and get the bounds of the area. 
        polygonsInAreaDf = gpd.GeoDataFrame(areaInfo['polygons'])
        boundariesInAreaDf = gpd.GeoDataFrame(areaInfo['boundaryWeight'])    
        bboxArea = box(*areaInfo['bounds'])
        bboxImg = box(*img.bounds)

        print(bboxArea)
        print(bboxImg)

        #Extract the window if area is in the image
        if(bboxArea.intersects(bboxImg)):
            print('intersects')
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
            profile['dtype'] = rasterio.float32 # rasterio.float32 # THIS GOT CHANGED! CHANGE BACK? or unint32?? # maybe fix this
            # write_extracted writes the image, annotation and boundaries and returns the counter of the next file to write. 
            writeCounter = write_extracted(img, sm, profile, polygonsInAreaDf, boundariesInAreaDf, writePath, imageFilename, annotationFilename, boundaryFilename, bands, writeCounter)
            overlapppedAreas.add(areaID)

        else: 
            print('not in area??')

    return(writeCounter, overlapppedAreas)

def write_extracted(img, sm, profile, polygonsInAreaDf, boundariesInAreaDf, writePath, imagesFilename, annotationFilename, boundaryFilename, bands, writeCounter, normalize=True):
    """
    Write the part of raw image that overlaps with a training area into a separate image file. 
    Use rowColPolygons to create and write annotation and boundary image from polygons in the training area.
    Note: original name was: writeExtractedImageAndAnnotation

    To Do: remove img from args because it's not used? will need to make chagnes to other files that reference it. 

    """
    print('about to try to write extracted')
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

    #change multipolygon to polygon
    bounda = bounda.explode()
    bounda.reset_index(drop=True,inplace=True)
    #bounda.to_file('boundary_ready_to_use.shp')
    return bounda

def row_col_polygons(areaDf, areaShape, profile, filename, outline, fill):
    """
    Convert polygons coordinates to image pixel coordinates, create annotation image using drawPolygons() and write the results into an image file.
    """

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

# decomposed better_preprocess


input_data_dir = '/Users/yaroslav/Documents/Work/GitHub/cnn-tree-heights/src/monthly/feb2023/misc/test-preprocess-versions/input'
output_data_dir = '/Users/yaroslav/Documents/Work/GitHub/cnn-tree-heights/src/monthly/feb2023/misc/test-preprocess-versions/output'

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
#print(__name__)
#quit()
if __name__ == '__main__':
    pool = Pool(processes=n_jobs)
    allAreasWithPolygons = pool.map(preprocess_single, area_files)

    print(allAreasWithPolygons)

    inputImages = list(zip(raw_ndvi_images,raw_pan_images))

    pool = Pool(processes=n_jobs)
    partial_func = partial(extract_overlapping, inputImages=inputImages, allAreasWithPolygons=allAreasWithPolygons, writePath=output_data_dir, bands=[0])
    pool.map(partial_func, range(total_jobs))

    quit()
