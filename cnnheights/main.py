def main(paradigm:str, input_data_dir:str, output_dir:str, num_epochs:int=5, num_patches:int=8, batch_size:int=8):
    r'''
    
    Arguments
    ---------

    paradigm : str
        Options include 'tensorflow-1' (optimized tensorflow), 'pytorch-0' (first pytorch implementation), 'pytorch-1' (second pytorch implementation), 'pytorch-2' (third pytorch implementation)
    
    input_data_dir : str
        Full path to directory with (extracted_ndvi_i.png, extracted_pan_i.png, extractd_annotation_i.png, and extracted_boundary_i.png) files    
    
    output_dir : str
        Full path to where all the output should go to 

    num_epochs : int 
        Number of training epochs 

    num_patches : int 
        Number of 256x256 chunks that should be generated from input images (relevant for pytorch paradigms only)

    '''

    import numpy as np 
    import pandas as pd 
    import os 
    from cnnheights.predict import predict
    from cnnheights.tensorflow.train import train_model
    import json
    from cnnheights.predict import heights_analysis
    import warnings
    #warnings.filterwarnings('ignore', message='.*initial implementation of Parquet.*')

    model_paradigms = ['tensorflow-1', 'pytorch-0', 'pytorch-1', 'pytorch-2']

    predictions_dir = os.path.join(output_dir, 'predictions')
    if not os.path.exists(predictions_dir):
        os.mkdir(predictions_dir)

    if paradigm in model_paradigms: 
        if paradigm == 'tensorflow-1': 

            raster_annotations = [os.path.join(input_data_dir, i) for i in os.listdir(input_data_dir) if 'extracted_annotation' in i]
            boundaries = [i.replace('annotation', 'boundary') for i in raster_annotations]
            ndvi_images = [i.replace('annotation', 'ndvi') for i in raster_annotations]
            pan_images = [i.replace('annotation', 'pan') for i in raster_annotations]

            vector_annotations = [i.replace('raster', 'vector').replace('png', 'gpkg') for i in raster_annotations]
            # new nomenclature: raster and vector vs raw and extracted 

            model, hist, test_generator = train_model(ndvi_images, pan_images, raster_annotations, boundaries, logging_dir=output_dir, epochs=num_epochs, batch_size=batch_size)
        
            hist_df = pd.DataFrame().from_dict(hist)
            hist_df.to_csv(os.path.join(output_dir, 'history-df.csv'), index=False)

            ## MY PREDICT METHOD ## --> COULD GET REPLACED BY USING TEST GENERATOR INSTEAD? 
            test_frames = [] 
            with open(os.path.join(output_dir, 'patches256/frames_list.json')) as json_file:
                data = json.load(json_file)
                test_frames = data['testing_frames']

            for test_frame_index in test_frames: 
                temp_pan_images = np.array([i.split('/')[-1] for i in pan_images])
                idx = np.where(temp_pan_images==f'extracted_pan_{test_frame_index}.png')[0][0]

            test_ndvi, test_pan, test_annotations, test_boundaries = (np.array(i)[test_frames] for i in (ndvi_images, pan_images, annotations, boundaries))
            
            predictions, predicted_metrics = predict(model, output_dir=predictions_dir,
                                                         ndvi_paths=test_ndvi,
                                                         pan_paths=test_pan,
                                                         annotation_paths=test_annotations,
                                                         weight_paths=test_boundaries)

            for idx, predicted_gdf in enumerate(predictions): 
                true_gdf = vector_annotations[idx]

                heights_gdf = heights_analysis(predicted_gdf=predicted_gdf, true_gdf=true_gdf)

                heights_gdf.to_file(os.path.join(output_dir, f'heights_analysis_output_{idx}.gpkg'))

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
    
    input_data_dir = '/ar1/PROJ/fjuhsd/personal/thaddaeus/github/cnn-tree-heights/data/first-shadows-dataset' 
    output_dir = '/ar1/PROJ/fjuhsd/personal/thaddaeus/github/cnn-tree-heights/temp/tensorflow' 
    
    for paradigmn in ['tensorflow-1']:#, 'pytorch-0']: #, 'pytorch-1']: 
        print(f'Paradigm: {paradigmn}') 
        main(paradigm=paradigmn, input_data_dir=input_data_dir, output_dir=output_dir) 