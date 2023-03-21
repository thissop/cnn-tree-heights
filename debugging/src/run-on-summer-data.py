import pyproj 

def main(output_dir:str,
         data_dir:str='data/cnn-input', 
         epochs:int=1, training_steps:int=15, confusion_matrix:bool=False):
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
    from cnnheights.prediction import predict
    import os 
    import pandas as pd
    import numpy as np
    import json
    import warnings                  # ignore annoying warnings
    warnings.filterwarnings("ignore")
    import time 

    annotations = []
    boundaries = []
    ndvi_images = []
    pan_images = []

    for file in np.sort(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, file)
        if '.png' in file: 
            if 'annotation' in file: 
                annotations.append(full_path) 
    
            elif 'boundary' in file: 
                boundaries.append(full_path) 

            elif 'extracted_ndvi' in file: 
                ndvi_images.append(full_path) 

            elif 'extracted_pan' in file: 
                pan_images.append(full_path) 

    print('about to run train_cnn')
    model, hist = train_cnn(ndvi_images, pan_images, annotations, boundaries, logging_dir=output_dir, epochs=epochs, training_steps=training_steps, 
                            make_confusion_matrix=confusion_matrix)

    hist_df = pd.DataFrame().from_dict(hist)
    hist_df.to_csv(os.path.join(output_dir, 'history-df.csv'), index=False)

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
    gdf = gpd.read_parquet('/ar1/PROJ/fjuhsd/personal/thaddaeus/github/cnn-tree-heights/debugging/src/temp-output/predictions/predicted_polygons.geoparquet')
    print(gdf)
    print(len(gdf.index))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    gdf.plot(ax=ax)
    plt.tight_layout()
    plt.savefig('/ar1/PROJ/fjuhsd/personal/thaddaeus/github/cnn-tree-heights/debugging/src/temp-output/predicted_polygons.pdf')

if __name__ == '__main__':
    main(data_dir='/ar1/PROJ/fjuhsd/personal/thaddaeus/github/cnn-tree-heights/cnn-input',
         output_dir='/ar1/PROJ/fjuhsd/personal/thaddaeus/github/cnn-tree-heights/debugging/src/temp-output')