def mask_to_polygons(maskF, transform):
    import cv2
    from shapely.geometry import Polygon
    from collections import defaultdict
    import rasterio
    import numpy as np
    import geopandas as gpd 

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
    mask[mask >= th] = 1 # already uint8...I already tested it. 

    # GDAL APPROACH: 
    '''
    seems like there is a C function implementation as opposed to a full python file that needs to be run in terminal...

    CPLErr GDALPolygonize(GDALRasterBandH hSrcBand, GDALRasterBandH hMaskBand, OGRLayerH hOutLayer, int iPixValField, char **papszOptions, GDALProgressFunc pfnProgress, void *pProgressArg)
    Create polygon coverage from raster data.

    This function creates vector polygons for all connected regions of pixels in the raster sharing a common pixel value. Optionally each polygon may be labeled with the pixel value in an attribute. Optionally a mask band can be provided to determine which pixels are eligible for processing.
    
    '''
    def gdal_approach(tiff_path:str, output_path:str, crs:str): 
        
        from osgeo import gdal, ogr, osr

        '''
        
        although this code will work itself in polygonizing predicted mask, 
        a bunch of modifications throughout the rest of the pipeline would need to 
        be implemented for the whole system to work with it...and I'm not sure if 
        implementing those far reaching changes would be a wise thing to do. 
        for example, it would add yet another layer of complexity to the data loader method, 
        which would now have to track individual 256x256 tiff files for every single input/output
        being used...this would require me to add steps to preprocessing to save 256x256 tiff "views"
        of any input data that's larger than 256x256, which would add a lot of complexity to 
        our approach. additionally, I would need to modify writeMaskToDisk() to make it more efficient 
        so it doesn't resave this list of polygons back into the gdf that I originally read it from 
        just to read it into a list to keep things consistent. 

        '''

        #  get raster datasource
        src_ds = gdal.Open(tiff_path)
        srcband = src_ds.GetRasterBand(1)
        dst_layername = tiff_path.split('/')[-1].split('.')[0]
        drv = ogr.GetDriverByName("ESRI Shapefile")
        dst_ds = drv.CreateDataSource(output_path)

        sp_ref = osr.SpatialReference()
        sp_ref.SetFromUserInput(crs)

        dst_layer = dst_ds.CreateLayer(dst_layername, srs = sp_ref )

        fld = ogr.FieldDefn("HA", ogr.OFTInteger)
        dst_layer.CreateField(fld)
        dst_field = dst_layer.GetLayerDefn().GetFieldIndex("HA")

        gdal.Polygonize( srcband, None, dst_layer, dst_field, [], callback=None )

        all_polygons = list(gpd.read_file(filename=output_path)['geometry'])
        # above is kinda redundant because we'll re-write it to file right after...but trying to keep things consistent. 

    def cv2_approach(mask): 

        #mask = mask.astype(np.uint8)

        mask = ((mask) * 255).astype(np.uint8) # go from [0,1] to [0,255]

        # mask = mask.astype(np.uint8) # this doesn't help! every time I run as is, no polygons are found...

        #mask = cv2.convertScaleAbs(mask) --> didn't help )

        # ISSUE COMES FROM HERE: 

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        # image Source, an 8-bit single-channel image. Non-zero pixels are treated as 1's. Zero pixels remain 0's, so the image is treated as binary...If mode equals to #RETR_CCOMP or #RETR_FLOODFILL, the input can also be a 32-bit integer image of labels (CV_32SC1)
            # can we just set it to 8-bit? why does it have to be 0 vs 255? I don't know why setting to 1s would help things because limit of 8-bit number is [0,255]

        # CV.CHAIN_APPROX_NONE: store all points around countour (basically, don't simplify shape)
        # what are CV_8UC1 images versus CV_32SC1 images? is this a CV_32SC1 image?

        # doesn't have issue when predicting on binary 0,1 image...compare output polygons? ... feels so extra but I need to do it...need to shift my mentality 
        # 
        # mentality: what gets the job done (minimum viable product, whatever quiets the error messages)...versus diving into root of problems
    
        # additional growth: challenging assumptions...I'm assuming here that the issue is coming from the integers themselves, but what if it's just that after only one epoch of training the predictions are so large there are no contours?

        # update on above: it still throws error about data type overflow when I run 5 epochs on 8 patches per epoch step, so I'm going to repeat above with 0,1

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
                    # print("An exception occurred in createShapefileObject; Polygon must have more than 2 points")

        return all_polygons

    all_polygons = cv2_approach(mask=mask)
    #all_polygons = gdal_approach()

    return all_polygons 

def writeMaskToDisk(detected_mask, detected_meta, save_path:str, write_as_type = 'uint8', th = 0.5):
    import geopandas as gpd 
    from cnnheights.predict import mask_to_polygons
    # Convert to correct required before writing
    if 'float' in str(detected_meta['dtype']) and 'int' in write_as_type:
        print(f'Converting prediction from {detected_meta["dtype"]} to {write_as_type}, using threshold of {th}')
        detected_mask[detected_mask<th]=0
        detected_mask[detected_mask>=th]=1
        detected_mask = detected_mask.astype(write_as_type)
        detected_meta['dtype'] =  write_as_type

    res = mask_to_polygons(detected_mask, detected_meta['transform'])

    d = {'geometry':[i for i in res if i.type == 'Polygon']}
    gdf = gpd.GeoDataFrame(d, crs=detected_meta['crs'])
    #gdf = gdf[gdf.geom_type != 'MultiPolygon'] # NOTE THIS FOR FUTURE! HAD TO TAKE OUT GDF!!
    gdf.to_file(save_path)#, schema=schema)

    return gdf 

def heights_analysis(predicted_gdf, cutlines_shp_file:str, true_gdf=None, d:float=9): 

    '''
    
    Notes
    -----

    haha this is very inefficient with for loops and individual comparisons, I just want to get it working first, and then I'll improve it. 
    
    '''

    from cnnheights.utilities import get_heights
    import geopandas as gpd
    import numpy as np

    if type(predicted_gdf) is str: 
        predicted_gdf = gpd.read_file(predicted_gdf)
    if type(true_gdf) is str: 
        true_gdf = gpd.read_file(true_gdf)

    crs = f'EPSG:{predicted_gdf.crs.to_epsg()}'

    '''
    Three Outcomes: -1 = NC, 0 = FN, 1 = FP, 2 = matched
    -1. Predicted exists, no corresponding true to check  
    0. True Exists, Predicted Doesn't (FN) 
    1. True Doesn't, Predicted Does (FP)
    2. True Exists, Predicted Does (Calculate Loss on Length/Height)
    '''
    
    all_centroids = []
    all_classes = []
    all_predicted_heights = []
    all_true_heights = []

    predicted_heights = get_heights(annotations_gdf=predicted_gdf, cutlines_shp_file=cutlines_shp_file)
    
    if true_gdf is not None: 
    
        true_heights = get_heights(annotations_gdf=true_gdf, cutlines_shp_file=cutlines_shp_file)

        for i, predicted_shadow in enumerate(predicted_gdf['geometry']): 
            idx = np.where(true_gdf.overlaps(predicted_shadow)==True)[0] # need to fix for multiple? @THADDAEUS this is a long term thing to fix
            
            all_centroids.append(predicted_shadow.centroid)
            all_predicted_heights.append(predicted_heights[i])

            if len(idx) == 0: # predicted exists without true equivalent 
                all_classes.append(1) # FP 
                all_true_heights.append(0) 
            
            else:
                all_classes.append(2) 
                if len(idx) > 1:
                    idx = idx[0]
                all_true_heights.append(float(true_heights[idx])) 

        for i, true_shadow in enumerate(true_gdf['geometry']): 
            idx = np.where(predicted_gdf.overlaps(true_shadow)==True)[0]
            if len(idx) == 0: # true exists without predicted equivalent 
                all_centroids.append(true_shadow.centroid)
                all_classes.append(0)
                all_predicted_heights.append(0)
                all_true_heights.append(true_heights[i])

    else:
        all_classes = len(predicted_gdf.index)*[-1]

        for i, predicted_shadow in enumerate(predicted_gdf['geometry']):             
            all_centroids.append(predicted_shadow.centroid)
            all_predicted_heights.append(predicted_heights[i])

        all_true_heights = all_predicted_heights

    results_d = {'geometry':all_centroids, 'class':all_classes, 'P_height':all_predicted_heights, 'T_height':all_true_heights}

    results_gdf = gpd.GeoDataFrame(results_d, crs=crs)

    return results_gdf

def predict(model, output_dir:str, write_counters:list=None, 
            ndvi_paths:list=None, pan_paths:list=None, 
            test_loader=None, meta_infos:list=None, device:str='cpu'):
    
    r'''
    Unified prediction

    depending on pytorch implementation, this function won't take in string and load for pytorch model, because you need to initialize the class first; hence, to load a saved model use following code before supplying model to this function: 
        ```
        model = UNet(*args, **kwargs)
        model.load_state_dict(torch.load(PATH))
        ```

        
    note: with test_generator it would just be
    
    for i in range(1):
        test_images, test_label = next(test_generator)
                        
        #print(test_images.shape())
        #5 images per row: pan, ndvi, label, weight, prediction
        prediction = model.predict(test_images, steps=1)
        prediction[prediction>0.5]=1
        prediction[prediction<=0.5]=0   

    '''
    
    #from cnnheights.loss import torch_calc_loss
    from cnnheights.predict import writeMaskToDisk
    import pandas as pd
    import os 
    from sklearn.metrics import accuracy_score
    import rasterio
    from rasterio import windows
    from cnnheights.preprocessing import image_normalize
    from itertools import product
    import tensorflow as tf
    import numpy as np
        
    predictions  = []
    metrics = []

    if write_counters is None: 
        write_counters = range(len(meta_infos))

    # TENSORFLOW
    if meta_infos is None and test_loader is None: 
        if ndvi_paths is None or pan_paths is None: 
            raise Exception('') 
        
        else: 
            if type(model) is str: 
                from tensorflow import keras 
                model = keras.models.load_model(model, custom_objects={ 'tversky':tf_tversky_loss,
                                                                        'dice_coef':dice_coef, 'dice_loss':dice_loss,
                                                                        'specificity':specificity, 'sensitivity':sensitivity}) 
            
            from cnnheights.original_core.loss import tf_tversky_loss, dice_coef, dice_loss, specificity, sensitivity

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
                batch_pos = []
                for col_off, row_off in offsets:
                    window = windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
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
            
            predictions = []

            for k in range(len(ndvi_paths)): 

                ndvi_image = rasterio.open(ndvi_paths[k])
                pan_image = rasterio.open(pan_paths[k])

                assert ndvi_image.crs == pan_image.crs
                prediction, predicted_meta = detect_tree(ndvi_img=ndvi_image, pan_img=pan_image)

                #prediction_binary = tf.cast(prediction, tf.int8) # ugh these type, data class (np array vs tensor stuff)!! 
                prediction[prediction<0.5]=0 # this is th in the actual write to disk function, but i have to make a binary mask here to for different loss metrics
                prediction[prediction>=0.5]=1

                predicted_fp = os.path.join(output_dir, f'predicted_polygons_{k}.gpkg')    

                gdf = writeMaskToDisk(detected_mask=prediction, detected_meta=predicted_meta, save_path=predicted_fp)

                # NOT WORKING: WIERD ERRORS
                '''
                if annotation_paths is not None: 
                    
                    pass
                    
                    #labels = rasterio.open(annotation_paths[k]).read(1).astype(np.int8)
                    
                    #weights = rasterio.open(weight_paths[k]).read(1).astype(np.int8) # lol issue changing it from int because int was defaulting to int64

                    #labels = tf.Tensor(labels, dtype=tf.int8)
                    #weights = tf.Tensor(weights, dtype=tf.int8)

                    #metrics.append({'':dice_loss(labels, prediction_binary), 'tversky_loss':tf_tversky_loss(labels, prediction_binary, weights), 'accuracy':accuracy_score(labels, prediction_binary)}) # re-add dice loss? 
                    #predictions.append({'gdf':gdf, 'prediction':prediction, 'labels':labels, 'test-loss-weights':test_loss_weights})
                else: 
                    pass 
                    #metrics.append({'dice_loss':None, 'tversky_loss':None, 'accuracy':None})
                '''

                predictions.append({'gdf':gdf, 'prediction':prediction}) # re-add dice loss?  

    # PYTORCH ---> BROKEN!
    elif ndvi_paths is None and pan_paths is None: 
        if meta_infos is None or test_loader is None: 
            raise Exception('')
        else: 
            
            model.eval()   # Set model to evaluate mode 

            for i in range(len(meta_infos)): 
                test_inputs = next(iter(test_loader))
                if len(test_inputs) == 1: 
                    inputs = test_inputs
                    labels, test_loss_weights == (None, None)

                elif len(test_inputs) == 3:  
                    inputs, labels, test_loss_weights = test_inputs
                    labels = labels.to(device)

                else: 
                    raise Exception('')

                inputs = inputs.to(device)
                
                prediction = model(inputs)

                predicted_fp = os.path.join(output_dir, f'predicted_polygons_{i}.gpkg')         
                gdf = writeMaskToDisk(detected_mask=prediction.detach().numpy().squeeze(), detected_meta=meta_infos[i], save_path=predicted_fp)
                predictions.append({'gdf':gdf, 'prediction':prediction.detach().numpy().squeeze(), 'labels':labels, 'test-loss-weights':test_loss_weights})            
        
                if len(test_inputs) == 3: 
                    dice, tversky = torch_calc_loss(y_true=labels, y_pred=prediction, weights=test_loss_weights)[-1]
                    metrics.append({'dice_loss':dice, 'tversky_loss':tversky}) # @THADDAEUS: need to fix these because pytorch doesn't have working acc rn, but tensorflow metrics dicts do have working acc values in their metrics dictionaries.     
                else: 
                    metrics.append({'dice_loss':None, 'tversky_loss':None})
    
    '''
    if annotation_paths is not None: 
        metrics = pd.DataFrame(metrics)
        metrics.to_csv(os.path.join(output_dir, 'prediction-metrics.csv'), index=False)
    '''

    return predictions