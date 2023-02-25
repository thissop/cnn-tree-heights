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
    from cnnheights.utilities import extract_overlapping, divide_training_polygons
    import warnings 
    import numpy as np
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
    
    def preprocess_single(index:int, area_files=area_files, training_annotations=annotation_files): 
        trainingArea = gpd.read_file(area_files[index])
        trainingPolygon = gpd.read_file(training_annotations[index])

        trainingArea['id'] = range(trainingArea.shape[0])
        
        # areasWithPolygons contains the object polygons and weighted boundaries for each area!
        areasWithPolygons = divide_training_polygons(trainingPolygon, trainingArea, show_boundaries_during_processing=False)
        
        return areasWithPolygons

    pool = Pool(processes=n_jobs)
    allAreasWithPolygons = pool.map(preprocess_single, range(total_jobs))

    inputImages = list(zip(raw_ndvi_images,raw_pan_images))

    pool = Pool(processes=n_jobs)
    partial_func = partial(extract_overlapping, inputImages=inputImages, allAreasWithPolygons=allAreasWithPolygons, writePath=output_data_dir, bands=[0])
    pool.map(partial_func, range(total_jobs))

def old_preprocess(input_data_dir:str, output_data_dir:str): 

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

    Notes
    -----

    - The corresponding files in the area_files, annotation_files, raw_ndvi_images, and raw_pan_images lists need to all be in the same order index wise.  

    '''

    import geopandas as gps
    from cnnheights.utilities import extract_overlapping, divide_training_polygons
    import warnings 
    import numpy as np
    import os 

    warnings.filterwarnings("ignore")

    input_files = [os.path.join(input_data_dir, i) for i in os.listdir(input_data_dir)]
    
    area_files = np.sort([i for i in input_files if 'vector_rectangle' in i]) 
    annotation_files = np.sort([i for i in input_files if 'annotation' in i]) 
    raw_ndvi_images = np.sort([i for i in input_files if 'raw_ndvi' in i])
    raw_pan_images = np.sort([i for i in input_files if 'raw_pan' in i])

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
        areasWithPolygons = divide_training_polygons(trainingPolygon, trainingArea, show_boundaries_during_processing=False)
        #print(f'Assigned training polygons in {len(areasWithPolygons)} training areas and created weighted boundaries for polygons')

        allAreasWithPolygons.append(areasWithPolygons)

    #Parallel(n_jobs=n_jobs)(preprocess_single(index) for index in range(total_jobs))

    inputImages = list(zip(raw_ndvi_images,raw_pan_images))
    #print(f'Found a total of {len(input_images)} pair of raw image(s) to process!')

    # For each raw satellite image, determine if it overlaps with a training area. 
    # If a overlap if found, then extract + write the overlapping part of the raw image, create + write an image from training polygons and create + write an image from boundary weights in the that overlapping region.
        
    # Run the main function for extracting part of ndvi and pan images that overlap with training areas
    extract_overlapping(inputImages, allAreasWithPolygons=allAreasWithPolygons, writePath=output_data_dir, ndviFilename='extracted_ndvi',
                                                panFilename='extracted_pan', annotationFilename='extracted_annotation',
                                                boundaryFilename='extracted_boundary', bands=[0])

def train_cnn(ndvi_images:list,
          pan_images:list, 
          annotations:list,
          boundaries:list, 
         logging_dir:str=None,
         epochs:int=200, training_steps:int=1000, use_multiprocessing:bool=False, confusion_matrix:bool=False, 
         crs:str='EPSG:32628'): 

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

    use_multiprocessing : bool
        Higher level access to turning multiprocessing on or not

    confusion_matrix : bool
        calculate confusion matrix on test set predictions (or not). Note that cm.ravel() for every cm in the returned confusion_matrices will return tn, fp, fn, tp

    crs : str
        e.g. EPSG:32628; should probably be ready from internal files. 

    '''
        
    from cnnheights.utilities import load_train_test, train_model
    import json 
    import os 
    from cnnheights import predict
    import numpy as np

    train_generator, val_generator, test_generator = load_train_test(ndvi_images=ndvi_images, pan_images=pan_images, annotations=annotations, boundaries=boundaries, logging_dir=logging_dir)

    model, loss_history = train_model(train_generator=train_generator, val_generator=val_generator, logging_dir=logging_dir, NB_EPOCHS=epochs, MAX_TRAIN_STEPS=training_steps, use_multiprocessing=use_multiprocessing)

    if confusion_matrix: 
        from PIL import Image
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        with open(os.path.join(logging_dir, 'patches256/frames_list.json')) as json_file:
            
            confusion_matrices = []
            
            data = json.load(json_file)
            test_frames = data['testing_frames']

            for test_frame in test_frames:
                mask = predict(model, ndvi_images[test_frame], pan_images[test_frame], output_dir=logging_dir, crs=crs)

                predictions = mask.flatten()
                predictions[predictions>1] = 1
                predictions[predictions<1] = 0
                predictions = predictions.astype(int)

                image = Image.open(annotations[0])
                annotation_data = np.asarray(image).flatten()
                annotation_data[annotation_data>1] = 1
                annotation_data[annotation_data<1] = 0
                annotation_data = annotation_data.astype(int)

                cm = confusion_matrix(annotation_data, predictions, normalize='pred')
                confusion_matrices.append(cm)

        return model, loss_history, confusion_matrices 
    
    else: 
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
