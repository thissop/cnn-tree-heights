import matplotlib.pyplot as plt 

def predict(model, ndvi_image, pan_image, output_dir:str, crs:str):
    r'''
    
    Arguments
    ---------

    model : 
        - Either a "live" model (i.e. extant in the code already), or a string path to a .h5 saved version of a model (that this function will load with all the custom stuff)
    
    NOTES
    -----

    TO DO
    -----

    
    '''

    import numpy as np 
    import rasterio
    from rasterio import windows
    from cnnheights.preprocessing import image_normalize
    from itertools import product
    import os 
    import geopandas as gpd

    #import pyproj
    #pyproj.datadir.set_data_dir(pyproj_datadir)

    ndvi_image = rasterio.open(ndvi_image)
    pan_image = rasterio.open(pan_image)

    if type(model) is str: 
        from cnnheights.original_core.losses import tversky, dice_coef, dice_loss, specificity, sensitivity
        from tensorflow import keras 
        model = keras.models.load_model(model, custom_objects={ 'tversky':tversky,
                                                                'dice_coef':dice_coef, 'dice_loss':dice_loss,
                                                                'specificity':specificity, 'sensitivity':sensitivity})

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
            transform = windows.transform(window, ndvi_img.transform) # delete? 
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
    
    from cnnheights.utilities import invert_mask
    detected_mask = invert_mask(detected_mask)

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
        gdf.to_parquet(predicted_fp)#, schema=schema)

        '''
        with fiona.open(wp, 'w', crs=crs, driver='ESRI Shapefile', schema=schema) as sink: 
            for idx, mp in enumerate(res):
                try: 
                    sink.write({'geometry':mapping(mp), 'properties':{'id':str(idx), 'canopy':mp.area}})
                except: 
                    print('An exception occurred in createShapefileObject; Polygon must have more than 2 points')
        '''

    predicted_fp = os.path.join(output_dir, f'predicted_polygons.geoparquet')
    writeMaskToDisk(detected_mask=detected_mask, detected_meta=detected_meta, wp=predicted_fp, crs=crs)

    return detected_mask, detected_meta