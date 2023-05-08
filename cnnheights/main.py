def main(paradigm:str, input_data_dir:str, output_dir:str,
         model:str=None, test_input_dir:str=None, 
         num_epochs:int=1, num_patches:int=8, batch_size:int=2):
    r'''
    
    Arguments
    ---------

    paradigm : str
        Options include 'tensorflow-1' (optimized tensorflow), 'pytorch-0' (first pytorch implementation), 'pytorch-1' (second pytorch implementation), 'pytorch-2' (third pytorch implementation)
    
    input_data_dir : str
        Full path to directory with (ndvi_i.tiff, pan_i.tiff, raster_annotation_i.tiff, and raster_boundary_i.tiff).     
    
    output_dir : str
        Full path to where all the output should go to 

    model : str
        If you want to skip retraining a model, just provide the string to a saved version of a model here. if you've still provided a input_data_dir, it will provide metrics on how it performs on the predictions in the input data directory, and then will go straight to predicting on data for the stand_alone_input data directory, if that path is provided. 

    test_input_dir : str
        If you want to predict on inputs without doing loss analysis on them (i.e. inputs without corresponding annotations), provide a directory to them here. 
        
    num_epochs : int 
        Number of training epochs 

    num_patches : int 
        Number of 256x256 chunks that should be generated from input images (relevant for pytorch paradigms only)

    NOTES
    -----

    - CURRENT PROCESS/PROCEDURE: 

        1. PREPROCESSING: sample_background/preprocess from preprocessing.py

        2. TRAINING/PREDICTIONS: main.py

            a. Load data
            b. Train model
            c. Report training metrics, plots, etc. 
            d. Make predictions for test data that has annotation data 
            e. Report tree heights for predictions from step 2.d., compare these tree heights to ground truth tree heights calculated from corresponding annotations. (save GDF, save metrics like mae/mse, as well as sample image wide rasterized accuracy/tversky, etc. show histograms scaled to same xscale and yscale for predicted vs true heights.)
            
            NOTE: steps a.-c. are skipped if a path to an already trained model is provided. 

        3. STAND ALONE PREDICTION: main.py
 
            a. Load data
            b. Make predictions for data that doesn't have annotation data to compare to as ground truth
            c. Report tree heights for these predictions 

            NOTE: step 2. can be skipped if you don't provide input data dir but provide a trained model and a standalone predictions input directory. 

    - BIG CHANGES TO NOMENCLATURE/PROCESS: 
        1. "vector_" and "raster_" vs. "raw_" and "extracted_": switching from later to former because it's more specific/descriptive/appropriate language. "vector_" will refer to polygonized/gpkg sets (annotations, boundaries, etc.), "raster_" will refer to .png files, of course. Images will just be ndvi_i.png or ndvi_i.tiff
        2. Need to include vector and raster annotation files in the input_data_dir, so the model can get the tree heights (I don't want to re-polygonize everything in the middle of running it if the user doesn't provide these values.)
        3. Preprocessing is no longer going to write png copies of the image files it draws from (because Jesse said a while ago that we might as well just read from tiff because the main purpose in creating the png is that those png were originally getting normalized pre-neural net stage, which is something we don't want. hence, no need to make the png copies)
        4. rasterized annotations and boundaries are getting saved as tiff (to keep one file format per Jesse's request)
        5. predict (and main obviously) have been modified to incorporate standalone predictions. 
        6. before test, made sure to test ALL INPUTS (opened them up in qgis, DESCRIBE STEPS TAKEN)

        - had to reprocess the files and check them twice when converting to tiff for standard. 

    '''

    import numpy as np 
    import pandas as pd 
    import os 
    from cnnheights.predict import predict
    from cnnheights.tensorflow.train import train_model, load_train_val
    import json
    from tensorflow import keras
    from cnnheights.predict import heights_analysis
    from cnnheights.original_core.loss import tf_tversky_loss, dice_coef, dice_loss, specificity, sensitivity
    import warnings
    import matplotlib.pyplot as plt 
    from cnnheights.plotting import plot_height_histograms, plot_train_history
    #warnings.filterwarnings('ignore', message='.*initial implementation of Parquet.*')

    model_paradigms = ['tensorflow-1', 'pytorch-0', 'pytorch-1', 'pytorch-2']

    train_plot_dir = os.path.join(output_dir, 'plots')
    test_output_dir = os.path.join(output_dir, 'test')
    test_output_predictions_dir = os.path.join(test_output_dir, 'predictions')
    test_plot_dir = os.path.join(test_output_dir, 'plots')

    for i in [train_plot_dir, test_output_dir, test_plot_dir, test_output_predictions_dir]:
        if not os.path.exists(i):
            os.mkdir(i)

    if paradigm in model_paradigms: 
        if paradigm == 'tensorflow-1': 

            if input_data_dir is not None: 

                raster_annotations = [os.path.join(input_data_dir, i) for i in os.listdir(input_data_dir) if 'raster_annotation' in i]
                boundaries = [i.replace('annotation', 'boundary') for i in raster_annotations]
                ndvi_images = [i.replace('raster_annotation', 'ndvi').replace('gpkg', 'tiff') for i in raster_annotations]
                pan_images = [i.replace('ndvi', 'pan') for i in ndvi_images]
                vector_annotations = [i.replace('raster', 'vector').replace('tiff', 'gpkg') for i in raster_annotations]

                # new nomenclature: raster and vector vs raw and extracted 

                train_generator, val_generator = load_train_val(ndvi_images=ndvi_images, pan_images=pan_images, annotations=raster_annotations, boundaries=boundaries, output_dir=output_dir, batch_size=batch_size)
            
                if model is not None: 
                    if type(model) is str: 
                        model = keras.models.load_model(model, custom_objects={ 'tversky':tf_tversky_loss,
                                                                        'dice_coef':dice_coef, 'dice_loss':dice_loss,
                                                                        'specificity':specificity, 'sensitivity':sensitivity}) 

                else: 

                    model, hist = train_model(train_generator=train_generator, val_generator=val_generator, logging_dir=output_dir, epochs=num_epochs, batch_size=batch_size)
                    hist_df = pd.DataFrame().from_dict(hist)
                    hist_df.insert(0, 'epoch', np.array(range(len(hist_df.index)))+1)
                    hist_df.to_csv(os.path.join(output_dir, 'train-val-history.csv'), index=False)
                
            # 3. STAND ALONE PREDICTION

            # NEED TO ADD CALCULATION OF STATS FOR TEST IF ANNOTATIONS ARE PRESENT!

            if test_input_dir is not None: # will return heights analysis if annotations are present in input dir for test
                
                test_ndvi_images = [os.path.join(test_input_dir, i) for i in os.listdir(test_input_dir) if 'ndvi_' in i]
                test_pan_images = [i.replace('ndvi', 'pan') for i in ndvi_images]
                test_write_counters = [int(i.split('.')[0].split('_')[-1]) for i in ndvi_images]
                
                test_raster_annotations = [os.path.join(test_input_dir, i) for i in os.listdir(test_input_dir) if 'raster_annotation' in i]
                test_vector_annotations = [i.replace('raster', 'vector').split('.')[0]+'.gpkg' for i in test_raster_annotations]
                test_raster_boundaries = [i.replace('annotation', 'boundary') for i in test_raster_annotations]

                if len(test_raster_annotations) != len(test_ndvi_images):
                    test_raster_annotations = None
                    test_vector_annotations = None
                    test_raster_boundaries = None

                predictions = predict(model, output_dir=test_output_predictions_dir, ndvi_paths=test_ndvi_images, pan_paths=test_pan_images, write_counters=test_write_counters) 

                all_predicted_heights = []
                all_true_heights = []

                for idx, prediction_dict in enumerate(predictions): 
                    if test_vector_annotations is not None: 
                        true_gdf = test_vector_annotations[idx] 
                    else: 
                        true_gdf = None 

                    heights_gdf = heights_analysis(predicted_gdf=prediction_dict['gdf'], true_gdf=true_gdf, cutlines_shp_file='/ar1/PROJ/fjuhsd/personal/thaddaeus/github/cnn-tree-heights/SSAr2_32628_GE01-QB02-WV02-WV03-WV04_PAN_NDVI_010_003_mosaic_cutlines.shp') 

                    #from shapely.validation import make_valid

                    #valid_shape = make_valid(invalid_shape)

                    heights_gdf.to_file(os.path.join(test_output_dir, f'heights_analysis_output_{idx}.gpkg'))

                    heights_df = pd.DataFrame()
                    heights_df['class'] = heights_gdf['class']
                    heights_df['predicted_height'] = heights_gdf['P_height'] 
                    heights_df['true_height'] = heights_gdf['T_height']
                    heights_df.to_csv(os.path.join(test_output_dir, f'heights_analysis_output_{idx}.csv'), index=False)

                    all_predicted_heights = np.concatenate((all_predicted_heights, heights_gdf['P_height']))
                    all_true_heights = np.concatenate((all_true_heights, heights_gdf['T_height']))

                    plot_path = os.path.join(test_plot_dir, f'heights_histogram_{test_write_counters[idx]}.png')
                    plot_height_histograms(heights_gdf=heights_gdf, plot_path=plot_path)

                plot_path = os.path.join(test_plot_dir, f'all_heights_histogram.png')
                plot_height_histograms(true_heights=all_true_heights, predicted_heights=all_predicted_heights, plot_path=plot_path)

        else: 
            
            # everything below this point is not production worthy...once I get this working I'll take this back into dev, delete it out of dev, and then save it in pytorch dev branch

            import torch 
            from cnnheights.pytorch.train import load_data, train_model

            if paradigm == 'pytorch-0': 
                from cnnheights.pytorch.unet_0 import UNet
                model = UNet()

            elif paradigm == 'pytorch-1':
                from cnnheights.pytorch.unet_1 import UNet 
                model = UNet()

            elif paradigm == 'pytorch-2': 
                from cnnheights.pytorch.unet_2 import UNet
                model = UNet()

            device = 'cpu' 
            if torch.backends.mps.is_available(): 
                device = 'mps'

            elif torch.cuda.is_available(): 
                device = 'cuda'
            device = 'cpu' # fix later...issue somewhere with some of the tensors being cpu, some being mps, even after setting here 
            #print(f"Using device: {device}")

            model = model.to(device)

            #summary(model, input_size=(2, 1056, 1056))# input_size=(channels, H, W)) # Really mps, but this old summary doesn't support it for some reason
            train_loader, val_loader, test_loader, meta_infos = load_data(input_data_dir=input_data_dir, num_patches=num_patches, batch_size=batch_size)
            test_meta_infos = meta_infos[-1]
            model, train_val_metrics = train_model(model = model, train_loader = train_loader, val_loader=val_loader, num_epochs=num_epochs, device=device, output_dir=output_dir)

            predictions, predicted_metrics = predict(model=model, test_loader=test_loader, meta_infos=test_meta_infos, output_dir=predictions_dir) 
    
            torch.save(model.state_dict(), os.path.join(output_dir, 'unet_pytorch_model.pt'))

        # PLOTTING / POST PREDICTION ANALYSIS

        # 1. Plot Train/Val History ---> MOVE THIS TO PLOTTING FUNCTION? 

        hist_path = os.path.join(output_dir, 'train-val-history.csv')
        if os.path.exists(hist_path): 
            plot_train_history(train_val_hist_df=hist_path, plot_dir=train_plot_dir)

        # 2. 

        # 2. Plot 

        # Calculate Tree Heights

        # Calculate Loss on Tree Heights? 
        
        from cnnheights.plotting import plot_predictions

        plot_predictions(gdf=predictions[0]['gdf'])

    else: 
        raise Exception('Illegal value for paradigm argument. See documentation string.') 

if __name__ == "__main__": 
    
    input_data_dir = 'data/test-dataset' 
    output_dir = 'temp/tensorflow' 
    standalone_input_dir = 'data/standalone-fake'
    main(paradigm='tensorflow-1', input_data_dir=input_data_dir, output_dir=output_dir, test_input_dir=standalone_input_dir, 
         num_epochs=4) 