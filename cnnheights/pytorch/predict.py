import numpy as np

def predict(model, test_loader, meta_infos:list, crs:str, output_dir:str, device:str='cpu'):
    model.eval()   # Set model to evaluate mode
        
    predictions  = []
    for i in range(meta_infos): 
        inputs, labels, test_loss_weights = next(iter(test_loader))
        inputs = inputs.to(device)
        labels = labels.to(device)

        pred = model(inputs)
        pred = pred.detach().numpy().squeeze()

        gdf = save_predictions(prediction=pred, meta_info=meta_infos[i], output_dir=output_dir, crs=crs, i=i)
        predictions.append({'gdf':gdf, 'pred':pred})

def save_predictions(prediction:np.array, meta_info:dict, output_dir:str, crs:str, i:int=None): 
    r'''
    
    Arguments
    ---------

    predictions: Shape is X,Y
    
    '''
    import os 

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
                    # print("An exception occurred in createShapefileObject; Polygon must have more than 2 points")

        return all_polygons

    def writeMaskToDisk(detected_mask, detected_meta, save_path, crs, write_as_type = 'uint8', th = 0.5):
        import geopandas as gpd
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
        gdf.to_parquet(save_path)#, schema=schema)

        return gdf 

    if i is None: 
        i = 0
    predicted_fp = os.path.join(output_dir, f'predicted_polygons_{i}.geoparquet')
    #print(prediction[i])
    gdf = writeMaskToDisk(detected_mask=prediction, detected_meta=meta_info, save_path=predicted_fp, crs=crs)

    return gdf