def preprocess(area_files:list, 
               annotation_files:list, 
               raw_ndvi_images:list,
               raw_pan_images:list,
               output_path:str,
               bands=[0],
               show_boundaries_during_preprocessing:bool=False): 

    r'''
    _Short Description_  

    Parameters
    ----------      

    variable_name : `type`
        Description

    Returns
    -------

    variable_name : `type`
        Description 

    TO DO 
    -----

    - I want the the area and annotation files (as well as their images) to be forced to listed in same orders in list! I hate the annoying index thing! 

    '''

    import geopandas as gps
    from cnnheights.utilities import extract_overlapping, divide_training_polygons
    import warnings 

    warnings.filterwarnings("ignore")

    def preprocess_single(index:int, area_files=area_files, training_annotations=annotation_files): 
        
        trainingArea = gps.read_file(area_files[index])
        trainingPolygon = gps.read_file(training_annotations[index])

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
        areasWithPolygons = divide_training_polygons(trainingPolygon, trainingArea, show_boundaries_during_processing=show_boundaries_during_preprocessing, verbose=verbose)
        #print(f'Assigned training polygons in {len(areasWithPolygons)} training areas and created weighted boundaries for polygons')

        return areasWithPolygons

    total_jobs = len(area_files)

    if total_jobs<n_jobs: 
        n_jobs = total_jobs

    pool = Pool(processes=n_jobs)
    allAreasWithPolygons = pool.map(preprocess_single, range(total_jobs))

    #Parallel(n_jobs=n_jobs)(preprocess_single(index) for index in range(total_jobs))

    inputImages = list(zip(raw_ndvi_images,raw_pan_images))
    if verbose:
        print(f'Found a total of {len(inputImages)} pair of raw image(s) to process!')

    # For each raw satellite image, determine if it overlaps with a training area. 
    # If a overlap if found, then extract + write the overlapping part of the raw image, create + write an image from training polygons and create + write an image from boundary weights in the that overlapping region.
        
    # Run the main function for extracting part of ndvi and pan images that overlap with training areas
    pool = Pool(processes=n_jobs)
    partial_func = partial(extract_overlapping, inputImages=inputImages, allAreasWithPolygons=allAreasWithPolygons, writePath=output_path, bands=bands)
    pool.map(partial_func, range(total_jobs))

def train(): 
    pass 