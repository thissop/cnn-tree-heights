def main(paradigm:str, input_data_dir:str, output_dir:str, num_epochs:int=1, num_patches:int=8):
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

    model_paradigms = ['tensorflow-1', 'pytorch-0', 'pytorch-1', 'pytorch-2']

    if paradigm in model_paradigms: 
        if paradigm == 'tensorflow-1': 

            from cnnheights.tensorflow.train import train_model
            from cnnheights.tensorflow.predict import predict
            import json

            annotations = [i for i in os.listdir(input_data_dir) if 'extracted_annotation' in i]
            boundaries = [i.replace('annotation', 'boundary') for i in annotations]
            ndvi_images = [i.replace('annotation', 'ndvi') for i in annotations]
            pan_images = [i.replace('annotation', 'pan') for i in annotations]

            model, hist, test_generator = train_model(ndvi_images, pan_images, annotations, boundaries, logging_dir=output_dir, epochs=epochs, training_steps=training_steps)
        
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

                predictions_dir = os.path.join(output_dir, 'predictions')
                if not os.path.exists(predictions_dir):
                    os.mkdir(predictions_dir)

                predict(model, ndvi_fp=ndvi_images[idx], pan_fp=pan_images[idx],
                        output_dir=predictions_dir, counter=test_frame_index) 

        else: 
            
            import torch 
            from cnnheights.pytorch.train import load_data

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
            train_loader, val_loader, test_loader, meta_infos = load_data(input_data_dir=input_data_dir, num_patches=num_patches)
            
            model = train_model(train_loader = train_loader, val_loader=val_loader, num_epochs=num_epochs)

            predictions = predict(model=model, test_loader=test_loader, meta_info=meta_infos)

    else: 
        raise Exception('Illegal value for paradigm argument. See documentation string.')

if __name__ == "__main__": 
    main()