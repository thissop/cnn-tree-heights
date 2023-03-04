def make_predictions(): 

    from tensorflow import keras
    import tensorflow as tf 
    import numpy as np
    from cnnheights.prediction import predict
    from cnnheights.original_core.losses import tversky, dice_coef, dice_loss, specificity, sensitivity

    model_path = '/ar1/PROJ/fjuhsd/personal/thaddaeus/other/cnn-heights/saved_models/UNet/trees_20230303-1652_AdaDelta_weightmap_tversky_012_256.h5'
    model = keras.models.load_model(model_path, custom_objects={ 'tversky':tversky,
                                                                'dice_coef':dice_coef, 'dice_loss':dice_loss,
                                                                'specificity':specificity, 'sensitivity':sensitivity})

    ndvi_image = '/ar1/PROJ/fjuhsd/personal/thaddaeus/github/cnn-tree-heights/data/input/extracted_ndvi_2.png'
    pan_image = '/ar1/PROJ/fjuhsd/personal/thaddaeus/github/cnn-tree-heights/data/input/extracted_pan_2.png'
    output_dir = '/ar1/PROJ/fjuhsd/personal/thaddaeus/github/cnn-tree-heights/src/monthly/mar2023/wk1/prediction-output'

    detected_mask, detected_meta = predict(model=model, ndvi_image=ndvi_image, pan_image=pan_image, output_dir=output_dir, crs='EPSG:32628')

def make_viz(): 
    from cnnheights.utilities import shadows_from_annotations

    import geopandas as gpd

    predictions_file = 'src/monthly/mar2023/wk1/from-saved-model/prediction-output/predicted_polygons.shp'
    cutlines_shp  = '/ar1/PROJ/fjuhsd/personal/thaddaeus/other/cnn-heights/thaddaeus_cutline/SSAr2_32628_GE01-QB02-WV02-WV03-WV04_PAN_NDVI_010_003_mosaic_cutlines.shp'
    save_path = 'src/monthly/mar2023/wk1/from-saved-model/prediction-output'
    
    gdf = gpd.read_file(predictions_file)
    print(gdf)

    import matplotlib.pyplot as plt 

    fig, ax = plt.subplots()
    gdf.plot(ax=ax)
    plt.savefig(f'{save_path}/polys-only.pdf')

    shadows_gdf_path = f'/ar1/PROJ/fjuhsd/personal/thaddaeus/github/cnn-tree-heights/src/monthly/mar2023/wk1/from-saved-model/prediction-output/shadows-gdf.gpkg'

    shadows_gdf = shadows_from_annotations(predictions_file, cutlines_shp, north=1706575.98, east=446542.55, epsg='32326', save_path=shadows_gdf_path)

make_viz()