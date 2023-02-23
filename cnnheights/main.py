def better_preprocess(area_files:list, 
                      annotation_files:list, 
                      raw_ndvi_images:list,
                      raw_pan_images:list,
                      input_data_dir:str,
                      output_path:str,
                      bands=[0],
                      show_boundaries_during_preprocessing:bool=False):
    
    r'''
    
    Notes
    -----

    - This method includes my optimized png extraction method from the summer, along with an integration to Jesse's vector burning thing. 
    

    
    '''

    import geopandas as gpd
    from joblib import Parallel 
    import multiprocessing 
    from functools import partial 
    from osgeo import gdal, ogr
    from multiprocessing import Pool
    from os import listdir
    from cnnheights.utilities import extract_overlapping, divide_training_polygons
    import warnings 
    #warnings.filterwarnings("ignore")

    gdal.UseExceptions()
    ogr.UseExceptions()

    gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "TRUE")
    gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")
    gdal.SetConfigOption("NUM_THREADS", "ALL_CPUS")

    n_jobs = 8
    total_jobs = len(area_files)

    def preprocess_single(index:int, area_files=area_files, training_annotations=annotation_files): 
        trainingArea = gpd.read_file(area_files[index])
        trainingPolygon = gpd.read_file(training_annotations[index])

        trainingArea['id'] = range(trainingArea.shape[0])
        
        # areasWithPolygons contains the object polygons and weighted boundaries for each area!
        areasWithPolygons = divide_training_polygons(trainingPolygon, trainingArea, show_boundaries_during_processing=show_boundaries_during_preprocessing)
        
        return areasWithPolygons

    pool = Pool(processes=n_jobs)
    allAreasWithPolygons = pool.map(preprocess_single, range(total_jobs))

    #Parallel(n_jobs=n_jobs)(preprocess_single(index) for index in range(total_jobs))

    inputImages = list(zip(raw_ndvi_images,raw_pan_images))

    # For each raw satellite image, determine if it overlaps with a training area. 
    # If a overlap if found, then extract + write the overlapping part of the raw image, create + write an image from training polygons and create + write an image from boundary weights in the that overlapping region.
        
    # Run the main function for extracting part of ndvi and pan images that overlap with training areas
    pool = Pool(processes=n_jobs)
    partial_func = partial(extract_overlapping, inputImages=inputImages, allAreasWithPolygons=allAreasWithPolygons, writePath=output_path, bands=bands)
    pool.map(partial_func, range(total_jobs))

    ### BELOW MODIFIED FROM JESSE

    def compute_tree_annotation_and_boundary_raster(vector_fp):
        r'''
        
        NOTES
        -----

        - This is Jesse's code that creates the tree annotation and boundary rasters. I still need to create the extracted ndvi and pan images externally first 
        
        '''
        
        #NOTE(Jesse): Find an accompanying raster filename based on the vector_fp.
        vector_fp_split = vector_fp.split('/')
        vector_fn = vector_fp_split[-1]
        raster_fp_base = "/".join(vector_fp_split[:-1]) + "/"
        vector_fn = vector_fn.split('.')[0]
        v_id = vector_fn.split('_')[-1]

        raster_fp = raster_fp_base + f"extracted_ndvi_{int(v_id) - 1}.png"

        raster_disk_ds = gdal.Open(raster_fp)

        #NOTE(Jesse): Create in memory raster of the same geospatial extents as the mask for high performance access.
        raster_mem_ds = gdal.GetDriverByName("MEM").Create('', xsize=raster_disk_ds.RasterXSize, ysize=raster_disk_ds.RasterYSize, bands=1, eType=gdal.GDT_Byte)
        band = raster_mem_ds.GetRasterBand(1)
        band.SetNoDataValue(255)
        raster_mem_ds.SetGeoTransform(raster_disk_ds.GetGeoTransform())
        raster_mem_ds.SetProjection(raster_disk_ds.GetProjection())
        band.Fill(0)
        del raster_disk_ds

        #NOTE(Jesse): Similarly with the vector polygons.  Load from disk and into a memory dataset.
        vector_disk_ds = gdal.OpenEx(vector_fp, gdal.OF_VECTOR)
        vector_mem_ds = gdal.GetDriverByName("Memory").Create('', 0, 0, 0, gdal.GDT_Unknown) #NOTE(Jesse): GDAL has a highly unintuitive API
        vector_mem_ds.CopyLayer(vector_disk_ds.GetLayer(0), 'orig')
        del vector_disk_ds

        #NOTE(Jesse): 'Buffer' extends the geometry out by the geospatial unit amount, approximating 'scaling' by 1.5.
        #             OGR, believe it or not, does not have an easy way to scale geometries like this.
        #             SQL is our only performant recourse to apply these operations to the data within OGR.
        sql_layer = vector_mem_ds.ExecuteSQL("select Buffer(GEOMETRY, 1.5, 5) from orig", dialect="SQLITE")
        vector_mem_ds.CopyLayer(sql_layer, 'scaled') #NOTE(Jesse): The returned 'layer' is not part of the original dataset for some reason? Requires a manual copy.
        del sql_layer

        #NOTE(Jesse): "Burn" the unscaled vector polygons into the raster image.
        opt_orig = gdal.RasterizeOptions(bands=[1], burnValues=1, layers='orig')
        gdal.Rasterize(raster_mem_ds, vector_mem_ds, options=opt_orig)

        #NOTE(Jesse): Track which pixels were burned into (via the '1') here, and reuse the band later.
        orig_arr = band.ReadAsArray()
        orig_arr_mask = orig_arr == 1
        band.Fill(0)

        #NOTE(Jesse): Burn the scaled geometries with the 'add' option, which will add the burn value to the destination pixel
        #             for all geometries which overlap it.  Basically, create a heatmap.
        opt_scaled = gdal.RasterizeOptions(bands=[1], burnValues=1, layers='scaled', add=True)
        gdal.Rasterize(raster_mem_ds, vector_mem_ds, options=opt_scaled)

        #NOTE(Jesse): Retain pixels with burn values > 1 (0 means no polygon overlap, 1 means 1 polygon overlaps, and >2 means multiple overlaps)
        composite_arr = band.ReadAsArray()
        composite_arr[composite_arr > 1] = 2 #NOTE(Jesse): 2 means overlap
        composite_arr[composite_arr == 1] = 0 #NOTE(Jesse): 0 means no polygon coverage
        composite_arr[orig_arr_mask] = 1 #NOTE(Jesse): 1 means original canopy

        #NOTE(Jesse): Save the composite array out to disk.
        raster_disk_ds = gdal.GetDriverByName("GTiff").Create(raster_fp_base + f"annotation_and_boundary_{v_id}.tif", xsize=raster_mem_ds.RasterXSize, ysize=raster_mem_ds.RasterYSize, bands=1, eType=gdal.GDT_Byte)
        raster_disk_ds.GetRasterBand(1).SetNoDataValue(255)
        raster_disk_ds.SetGeoTransform(raster_mem_ds.GetGeoTransform())
        raster_disk_ds.SetProjection(raster_mem_ds.GetProjection())
        raster_disk_ds.GetRasterBand(1).WriteArray(composite_arr)
        del raster_disk_ds

    training_files = listdir(input_data_dir)
    training_files = [input_data_dir + f for f in training_files if f.endswith(".gpkg")]

    with Pool() as p:
        p.map(compute_tree_annotation_and_boundary_raster, training_files, chunksize=1)

def preprocess(area_files:list, 
               annotation_files:list, 
               raw_ndvi_images:list,
               raw_pan_images:list,
               output_path:str,
               bands=[0],
               show_boundaries_during_preprocessing:bool=False): 

    r'''
    _Prepare all the standardized training data for the CNN. Extracts annotations, boundaries, nvdi images, and pan images._  

    Arguments
    ----------      

    area_files : list
    annotation_files : `list`  
    raw_ndvi_images : `list` 
    raw_pan_images : `list` 
    output_path : `str`
        Output path for all the extracted files to be saved to. Should be Linux/Mac style, and last character should be forward slash `/` 
    bands : `list`
        Default is [0]. 
    show_boundaries_during_preprocessing : `bool`

    Notes
    -----

    - The corresponding files in the area_files, annotation_files, raw_ndvi_images, and raw_pan_images lists need to all be in the same order index wise.  

    '''

    import geopandas as gps
    from cnnheights.utilities import extract_overlapping, divide_training_polygons
    import warnings 

    warnings.filterwarnings("ignore")

    allAreasWithPolygons = [] 

    for i in range(len(area_files)): 
        trainingArea = gps.read_file(area_files[i])
        trainingPolygon = gps.read_file(annotation_files[i])

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
        areasWithPolygons = divide_training_polygons(trainingPolygon, trainingArea, show_boundaries_during_processing=show_boundaries_during_preprocessing)
        #print(f'Assigned training polygons in {len(areasWithPolygons)} training areas and created weighted boundaries for polygons')

        allAreasWithPolygons.append(areasWithPolygons)

    #Parallel(n_jobs=n_jobs)(preprocess_single(index) for index in range(total_jobs))

    inputImages = list(zip(raw_ndvi_images,raw_pan_images))
    #print(f'Found a total of {len(input_images)} pair of raw image(s) to process!')

    # For each raw satellite image, determine if it overlaps with a training area. 
    # If a overlap if found, then extract + write the overlapping part of the raw image, create + write an image from training polygons and create + write an image from boundary weights in the that overlapping region.
        
    # Run the main function for extracting part of ndvi and pan images that overlap with training areas
    extract_overlapping(inputImages, allAreasWithPolygons=allAreasWithPolygons, writePath=output_path, ndviFilename='extracted_ndvi',
                                                panFilename='extracted_pan', annotationFilename='extracted_annotation',
                                                boundaryFilename='extracted_boundary', bands=bands)

def train_cnn(ndvi_images:list,
          pan_images:list, 
          annotations:list,
          boundaries:list, 
         logging_dir:str=None,
         epochs:int=200, training_steps:int=1000): 

    r'''
    
    __Train the UNET CNN on the extracted data__

    Arguments 
    ---------

    ndvi_images : list 
        List of full file paths to the extracted ndvi images 

    pan_images : list 
        Same as ndvi_images except for pan 

    boundaries : list 
        List of boundary files extracted during the preproccessing step 

    annotations : list
        List of the full file paths to the extracted annotations outputed during the preproccessing step. 
    
    logging_dir : str
        Passed onto the load_train_test and train_model functions; see load_train_test_split docstring for explanation. 

    '''
        
    from cnnheights.utilities import load_train_test, train_model
    
    train_generator, val_generator, test_generator = load_train_test(ndvi_images=ndvi_images, pan_images=pan_images, annotations=annotations, boundaries=boundaries, logging_dir=logging_dir)

    model, loss_history = train_model(train_generator=train_generator, val_generator=val_generator, logging_dir=logging_dir, NB_EPOCHS=epochs, MAX_TRAIN_STEPS=training_steps)

    return model, loss_history

def predict(model, ndvi_image, pan_image, output_dir:str, crs:str, pyproj_datadir:str='/home/fjuhsd/miniconda3/envs/cnnheights310/share/proj'):
    r'''
    
    NOTES
    -----
    
    TO DO
    -----

    
    '''

    import numpy as np 
    import rasterio
    from rasterio import windows
    from cnnheights.utilities import image_normalize
    from itertools import product

    import pyproj
    pyproj.datadir.set_data_dir(pyproj_datadir)

    ndvi_image = rasterio.open(ndvi_image)
    pan_image = rasterio.open(pan_image)

    def addTOResult(res, prediction, row, col, he, wi, operator = 'MAX'):
        currValue = res[row:row+he, col:col+wi]
        newPredictions = prediction[:he, :wi]
        # IMPORTANT: MIN can't be used as long as the mask is initialed with 0!!!!! If you want to use MIN initial the mask with -1 and handle the case of default value(-1) separately.
        if operator == 'MIN': # Takes the min of current prediction and new prediction for each pixel
            currValue [currValue == -1] = 1 #Replace -1 with 1 in case of MIN
            resultant = np.minimum(currValue, newPredictions) 
        elif operator == 'MAX':
            resultant = np.maximum(currValue, newPredictions)
        else: #operator == 'REPLACE':
            resultant = newPredictions    
        # Alternative approach; Lets assume that quality of prediction is better in the centre of the image than on the edges
        # We use numbers from 1-5 to denote the quality, where 5 is the best and 1 is the worst.In that case, the best result would be to take into quality of prediction based upon position in account
        # So for merge with stride of 0.5, for eg. [12345432100000] AND [00000123454321], should be [1234543454321] instead of [1234543214321] that you will currently get. 
        # However, in case the values are strecthed before hand this problem will be minimized
        res[row:row+he, col:col+wi] =  resultant
        return (res)

    def predict_using_model(model, batch, batch_pos, mask, operator):
        tm = np.stack(batch, axis = 0)
        prediction = model.predict(tm)
        for i in range(len(batch_pos)):
            (col, row, wi, he) = batch_pos[i]
            p = np.squeeze(prediction[i], axis = -1)
            # Instead of replacing the current values with new values, use the user specified operator (MIN,MAX,REPLACE)
            mask = addTOResult(mask, p, row, col, he, wi, operator)
        return mask

    def detect_tree(ndvi_img, pan_img, width=256, height=256, stride = 128, normalize=True):
        ncols, nrows = ndvi_img.meta['width'], ndvi_img.meta['height']
        meta = ndvi_img.meta.copy()
        if 'float' not in meta['dtype']: #The prediction is a float so we keep it as float to be consistent with the prediction. 
            meta['dtype'] = np.float32

        offsets = product(range(0, ncols, stride), range(0, nrows, stride))
        big_window = windows.Window(col_off=0, row_off=0, width=ncols, height=nrows)
        #print(nrows, ncols)

        mask = np.zeros((nrows, ncols), dtype=meta['dtype'])

        # mask = mask -1 # Note: The initial mask is initialized with -1 instead of zero to handle the MIN case (see addToResult)
        batch = []
        batch_pos = [ ]
        for col_off, row_off in offsets:
            window = windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
            transform = windows.transform(window, ndvi_img.transform)
            patch = np.zeros((height, width, 2)) #Add zero padding in case of corner images
            ndvi_sm = ndvi_img.read(window=window)
            pan_sm = pan_img.read(window=window)
            temp_im = np.stack((ndvi_sm, pan_sm), axis = -1)
            temp_im = np.squeeze(temp_im)
            
            if normalize:
                temp_im = image_normalize(temp_im, axis=(0,1)) # Normalize the image along the width and height i.e. independently per channel
                
            patch[:window.height, :window.width] = temp_im
            batch.append(patch)
            batch_pos.append((window.col_off, window.row_off, window.width, window.height))
            if (len(batch) == 8):
                mask = predict_using_model(model, batch, batch_pos, mask, 'MAX')
                batch = []
                batch_pos = []
                
        # To handle the edge of images as the image size may not be divisible by n complete batches and few frames on the edge may be left.
        if batch:
            mask = predict_using_model(model, batch, batch_pos, mask, 'MAX')
            batch = []
            batch_pos = []

        return (mask, meta)
    
    detected_mask, detected_meta = detect_tree(ndvi_img=ndvi_image, pan_img=pan_image)

    import matplotlib.pyplot as plt  # plotting tools
    from matplotlib.patches import Polygon
    import os 
    from shapely.geometry import mapping, shape
    #import fiona 
    import geopandas as gpd

    def mask_to_polygons(maskF, transform):
        import cv2
        from shapely.geometry import Polygon
        from collections import defaultdict

        def transformContoursToXY(contours, transform = None):
            tp = []
            for cnt in contours:
                pl = cnt[:, 0, :]
                cols, rows = zip(*pl)
                x,y = rasterio.transform.xy(transform, rows, cols)
                tl = [list(i) for i in zip(x, y)]
                tp.append(tl)
            return (tp)

        # first, find contours with cv2: it's much faster than shapely
        th = 0.5
        mask = maskF.copy()
        mask[mask < th] = 0
        mask[mask >= th] = 1
        mask = ((mask) * 255).astype(np.uint8)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        #Convert contours from image coordinate to xy coordinate
        contours = transformContoursToXY(contours, transform)
        if not contours: #TODO: Raise an error maybe
            print('Warning: No contours/polygons detected!!')
            return [Polygon()]
        # now messy stuff to associate parent and child contours
        cnt_children = defaultdict(list)
        child_contours = set()
        assert hierarchy.shape[0] == 1
        # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
        for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
            if parent_idx != -1:
                child_contours.add(idx)
                cnt_children[parent_idx].append(contours[idx])

        # create actual polygons filtering by area (removes artifacts)
        all_polygons = []

        for idx, cnt in enumerate(contours):
            if idx not in child_contours: #and cv2.contourArea(cnt) >= min_area: #Do we need to check for min_area??
                try:
                    poly = Polygon(
                        shell=cnt,
                        holes=[c for c in cnt_children.get(idx, [])])
                            #if cv2.contourArea(c) >= min_area]) #Do we need to check for min_area??
                    all_polygons.append(poly)
                except:
                    pass
    #                 print("An exception occurred in createShapefileObject; Polygon must have more than 2 points")
        print(len(all_polygons))
        return all_polygons

    schema = {'geometry': 'Polygon', 'properties': {'id': 'str', 'canopy': 'float:15.2',}}

    def writeMaskToDisk(detected_mask, detected_meta, wp, crs, write_as_type = 'uint8', th = 0.5):
        # Convert to correct required before writing
        if 'float' in str(detected_meta['dtype']) and 'int' in write_as_type:
            print(f'Converting prediction from {detected_meta["dtype"]} to {write_as_type}, using threshold of {th}')
            detected_mask[detected_mask<th]=0
            detected_mask[detected_mask>=th]=1
            detected_mask = detected_mask.astype(write_as_type)
            detected_meta['dtype'] =  write_as_type

        res = mask_to_polygons(detected_mask, detected_meta['transform'])

        d = {'geometry':[i for i in res if i.type == 'Polygon']}
        gdf = gpd.GeoDataFrame(d, crs=crs)
        #gdf = gdf[gdf.geom_type != 'MultiPolygon'] # NOTE THIS FOR FUTURE! HAD TO TAKE OUT GDF!!
        gdf.to_file(predicted_fp, driver='ESRI Shapefile')#, schema=schema)

        '''
        with fiona.open(wp, 'w', crs=crs, driver='ESRI Shapefile', schema=schema) as sink: 
            for idx, mp in enumerate(res):
                try: 
                    sink.write({'geometry':mapping(mp), 'properties':{'id':str(idx), 'canopy':mp.area}})
                except: 
                    print('An exception occurred in createShapefileObject; Polygon must have more than 2 points')
        '''

    predicted_fp = os.path.join(output_dir, f'predicted_polygons.shp')
    writeMaskToDisk(detected_mask=detected_mask, detected_meta=detected_meta, wp=predicted_fp, crs=crs)
