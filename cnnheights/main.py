def main(output_dir:str,
         data_dir:str='/ar1/PROJ/fjuhsd/personal/thaddaeus/github/cnn-tree-heights/cnn-input', 
         epochs:int=5, training_steps:int=5, confusion_matrix:bool=False):

    r'''
    
    Arguments
    ---------

    data_dir : absolute path to the local directory of data (should be the './first-shadows-dataset')

    output_dir : path to the folder 'model-output' on local machine

    epochs : number of training epochs

    training_steps : number of training steps per epoch

    pyproj_datadir : see below notes
        - This was something annoying, might throw issues for you. Find it with following: 
         ```
         import pyproj 

         pyproj_datadir = pyproj.datadir.get_data_dir()
         ```    

    Notes 
    -----

        - Multiprocessing is on. 

    '''

    from cnnheights.training import train_cnn
    import os 
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt 
    import json
    from cnnheights.prediction import predict
    from cnnheights.debug_utils import display_images
    import geopandas as gpd
    import warnings                  # ignore annoying warnings
    warnings.filterwarnings("ignore")
    import time 

    annotations = [os.path.join(data_dir, f'extracted_annotation_{i}.png') for i in range(10)]
    boundaries = [i.replace('annotation', 'boundary') for i in annotations]
    ndvi_images = [i.replace('annotation', 'ndvi') for i in annotations]
    pan_images = [i.replace('annotation', 'pan') for i in annotations]

    print('about to run train_cnn')

    # why is it only getting four? 

    model, hist, test_generator = train_cnn(ndvi_images, pan_images, annotations, boundaries, logging_dir=output_dir, epochs=epochs, training_steps=training_steps, 
                            make_confusion_matrix=confusion_matrix)

    test_images, real_label = next(test_generator)
    #5 images per row: pan, ndvi, label, weight, prediction
    prediction = model.predict(test_images, steps=1)
    prediction[prediction>0.5]=1
    prediction[prediction<=0.5]=0
    display_images(np.concatenate((test_images, prediction), axis = -1), 'debugging-take-2/output/plots/predictions_display_image.pdf')

    hist_df = pd.DataFrame().from_dict(hist)
    hist_df.to_csv(os.path.join(output_dir, 'history-df.csv'), index=False)

    ## MY PREDICT METHOD ##
    test_frame = None 
    with open(os.path.join(output_dir, 'patches256/frames_list.json')) as json_file:
        data = json.load(json_file)
        test_frame = data['testing_frames'][0]

    temp_pan_images = np.array([i.split('/')[-1] for i in pan_images])
    idx = np.where(temp_pan_images==f'extracted_pan_{test_frame}.png')[0][0]

    predictions_dir = os.path.join(output_dir, 'predictions')
    if not os.path.exists(predictions_dir):
        os.mkdir(predictions_dir)

    predict(model, ndvi_image=ndvi_images[idx], pan_image=pan_images[idx],
            output_dir=predictions_dir, crs='EPSG:32628')

    import geopandas as gpd
    import matplotlib.pyplot as plt 

    fig, ax = plt.subplots()
    gdf = gpd.read_parquet('debugging-take-2/output/pipeline-output/predictions/predicted_polygons.geoparquet')
    print(gdf)
    gdf.plot(ax=ax)
    plt.savefig('debugging-take-2/output/plots/final-predictions.pdf')

if __name__ == '__main__':
    main(output_dir='/ar1/PROJ/fjuhsd/personal/thaddaeus/github/cnn-tree-heights/debugging-take-2/output/pipeline-output')