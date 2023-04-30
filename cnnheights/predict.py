def mask_to_polygons(maskF, transform):
    import cv2
    from shapely.geometry import Polygon
    from collections import defaultdict
    import rasterio
    import numpy as np

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

def predict(model, output_dir:str, write_counters:list=None, 
            test_loader=None, meta_infos:list=None, device:str='cpu',
            ndvi_paths:list=None, pan_paths:list=None, annotation_paths:list=None, weight_paths:list=None):
    
    r'''
    Unified prediction

    depending on pytorch implementation, this function won't take in string and load for pytorch model, because you need to initialize the class first; hence, to load a saved model use following code before supplying model to this function: 
        ```
        model = UNet(*args, **kwargs)
        model.load_state_dict(torch.load(PATH))
        ```


    '''
    
    from cnnheights.loss import torch_calc_loss
    from cnnheights.predict import writeMaskToDisk
    import pandas as pd
    import os 
    from sklearn.metrics import accuracy_score
    import rasterio
    from rasterio import windows
    from cnnheights.preprocessing import image_normalize
    from itertools import product
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
            metrics = []

            for k in range(len(ndvi_paths)): 

                ndvi_image = rasterio.open(ndvi_paths[k])
                pan_image = rasterio.open(pan_paths[k])

                assert ndvi_image.crs == pan_image.crs

                prediction, predicted_meta = detect_tree(ndvi_img=ndvi_image, pan_img=pan_image)

                predicted_fp = os.path.join(output_dir, f'predicted_polygons_{k}.gpkg')         
                gdf = writeMaskToDisk(detected_mask=prediction, detected_meta=meta_infos[k], save_path=predicted_fp)

                labels, test_loss_weights = (None, None)

                if annotation_paths is not None: 
                    labels = rasterio.open(annotation_paths[k])
                    weights = rasterio.open(weight_paths[k])
                    metrics.append({'dice_loss':dice_loss(labels, prediction), 'tversky_loss':tf_tversky_loss(labels, prediction, weights)})

                else: 
                    metrics.append({'dice_loss':None, 'tversky_loss':None})
                
                predictions.append({'gdf':gdf, 'prediction':prediction, 'labels':labels, 'test-loss-weights':test_loss_weights})            

    # PYTORCH 
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
                    metrics.append({'dice_loss':dice, 'tversky_loss':tversky})         
                else: 
                    metrics.append({'dice_loss':None, 'tversky_loss':None})
    
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(os.path.join(output_dir, 'prediction-metrics.csv'), index=False)

    return predictions, metrics