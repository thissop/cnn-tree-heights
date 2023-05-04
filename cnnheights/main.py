def main(paradigm:str, input_data_dir:str, output_dir:str,
         model:str=None, standalone_input_dir:str=None, 
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

    standalone_input_dir : str
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
    from cnnheights.tensorflow.train import train_model, load_train_test
    import json
    from tensorflow import keras
    from cnnheights.predict import heights_analysis
    from cnnheights.original_core.loss import tf_tversky_loss, dice_coef, dice_loss, specificity, sensitivity
    import warnings
    from cnnheights.plotting import plot_height_histograms
    #warnings.filterwarnings('ignore', message='.*initial implementation of Parquet.*')

    model_paradigms = ['tensorflow-1', 'pytorch-0', 'pytorch-1', 'pytorch-2']

    predictions_dir = os.path.join(output_dir, 'predictions')
    predictions_plot_dir = os.path.join(predictions_dir, 'plots')
    standalone_predictions_dir = os.path.join(output_dir, 'standalone_predictions')

    for i in [predictions_dir, predictions_plot_dir, standalone_predictions_dir]:
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

                train_generator, val_generator, test_generator = load_train_test(ndvi_images=ndvi_images, pan_images=pan_images, annotations=raster_annotations, boundaries=boundaries, logging_dir=output_dir, batch_size=batch_size)
            
                if model is not None: 
                    if type(model) is str: 
                        model = keras.models.load_model(model, custom_objects={ 'tversky':tf_tversky_loss,
                                                                        'dice_coef':dice_coef, 'dice_loss':dice_loss,
                                                                        'specificity':specificity, 'sensitivity':sensitivity}) 

                else: 

                    model, hist = train_model(train_generator=train_generator, val_generator=val_generator, logging_dir=output_dir, epochs=num_epochs, batch_size=batch_size)
                    hist_df = pd.DataFrame().from_dict(hist)
                    hist_df.to_csv(os.path.join(output_dir, 'history-df.csv'), index=False)

                ## MY PREDICT METHOD ## --> COULD GET REPLACED BY USING TEST GENERATOR INSTEAD? 
                test_frames = [] 
                with open(os.path.join(output_dir, 'patches256/frames_list.json')) as json_file:
                    data = json.load(json_file)
                    test_frames = data['testing_frames']

                #for test_frame_index in test_frames: 
                #    temp_pan_images = np.array([i.split('/')[-1] for i in pan_images])
                #    idx = np.where(temp_pan_images==f'extracted_pan_{test_frame_index}.png')[0][0]

                test_ndvi, test_pan, test_annotations, test_boundaries = (np.array(i)[test_frames] for i in (ndvi_images, pan_images, raster_annotations, boundaries))
                write_counters = [int(i.split('.')[0].split('_')[-1]) for i in test_ndvi]
                
                predictions, predicted_metrics = predict(model, output_dir=predictions_dir,
                                                            ndvi_paths=test_ndvi,
                                                            pan_paths=test_pan,
                                                            annotation_paths=test_annotations,
                                                            weight_paths=test_boundaries, 
                                                            write_counters=write_counters)

                print('###')
                print('###')
                print(predicted_metrics)
                print('###')
                print('###')
                quit()

                all_predicted_heights = []
                all_true_heights = []

                for idx, predicted_gdf in enumerate(predictions): 
                    true_gdf = vector_annotations[idx]

                    heights_gdf = heights_analysis(predicted_gdf=predicted_gdf, true_gdf=true_gdf)

                    heights_gdf.to_file(os.path.join(output_dir, f'heights_analysis_output_{idx}.gpkg'))

                    all_predicted_heights = np.concatenate((all_predicted_heights, heights_gdf['predicted_height']))
                    all_true_heights = np.concatenate((all_true_heights, heights_gdf['true_height']))

                    plot_path = os.path.join(predictions_plot_dir, f'heights_histogram_{write_counters[i]}.png')
                    plot_height_histograms(heights_gdf=heights_gdf, plot_path=plot_path)

                plot_path = os.path.join(predictions_plot_dir, f'all_heights_histogram_{write_counters[i]}.png')
                plot_height_histograms(true_heights=all_true_heights, predicted_heights=all_predicted_heights, plot_path=plot_path)

            # 3. STAND ALONE PREDICTION

            if standalone_input_dir is not None: 

                raster_annotations = [os.path.join(input_data_dir, i) for i in os.listdir(input_data_dir) if 'raster_annotation' in i]
                boundaries = [i.replace('annotation', 'boundary') for i in raster_annotations]
                ndvi_images = [i.replace('raster_annotation', 'ndvi').replace('gpkg', 'tiff') for i in raster_annotations]
                pan_images = [i.replace('ndvi', 'pan') for i in ndvi_images]
                write_counters = [int(i.split('.')[0].split('_')[-1]) for i in ndvi_images]

                predictions = predict(model, output_dir=standalone_predictions_dir, ndvi_paths=test_ndvi, pan_paths=test_pan, write_counters=write_counters) 


        else: 
            
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

        # Calculate Tree Heights

        # Calculate Loss on Tree Heights? 
        
        from cnnheights.plotting import plot_predictions

        plot_predictions(gdf=predictions[0]['gdf'])

    else: 
        raise Exception('Illegal value for paradigm argument. See documentation string.') 

if __name__ == "__main__": 
    
    input_data_dir = '/ar1/PROJ/fjuhsd/personal/thaddaeus/github/cnn-tree-heights/data/test-dataset' 
    output_dir = '/ar1/PROJ/fjuhsd/personal/thaddaeus/github/cnn-tree-heights/temp/tensorflow' 
    standalone_input_dir = '/ar1/PROJ/fjuhsd/personal/thaddaeus/github/cnn-tree-heights/data/standalone-fake'
    main(paradigm='tensorflow-1', input_data_dir=input_data_dir, output_dir=output_dir, standalone_input_dir=standalone_input_dir, 
         num_epochs=1) 