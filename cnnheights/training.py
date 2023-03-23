def load_train_test(ndvi_images:list, pan_images:list, annotations:list, boundaries:list, logging_dir:str=None):
    
    r'''
    
    Arguments 
    ---------

    area_files : list 
        List of the area files 

    annotations : list
        List of the full file paths to the extracted annotations that got outputed by the earlier preproccessing step. 

    ndvi_images : list 
        List of full file paths to the extracted ndvi images 

    pan_images : list 
        Same as ndvi_images except for pan 

    boundaries : list
        List of boundary files extracted by previous preprocessing step 

    logging_dir : str
        the directory all the logging stuff should be saved into. defaults to none, which will make all the directories in directory that the python file that executes this function is run in. 
    
    '''

    import os
    import rasterio 
    import numpy as np
    from PIL import Image
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    from cnnheights.original_core.frame_utilities import FrameInfo, split_dataset
    from cnnheights.original_core.dataset_generator import DataGenerator

    normalize = 0.4
    BATCH_SIZE = 8
    patch_size=(256,256,4)
    input_shape = (256,256,2)
    input_image_channel = [0,1]
    input_label_channel = [2]
    input_weight_channel = [3]

    if logging_dir is not None: 
        patch_dir = os.path.join(logging_dir, f'patches{patch_size[0]}/')
    else: 
        patch_dir = './patches{}'.format(patch_size[0])
    
    frames_json = os.path.join(patch_dir,'frames_list.json')

    # Read all images/frames into memory
    frames = []

    # problem is not in this for loop
    for i in range(len(ndvi_images)):
        ndvi_img = rasterio.open(ndvi_images[i])
        pan_img = rasterio.open(pan_images[i])
        read_ndvi_img = ndvi_img.read()
        read_pan_img = pan_img.read()
        comb_img = np.concatenate((read_ndvi_img, read_pan_img), axis=0)
        comb_img = np.transpose(comb_img, axes=(1,2,0)) #Channel at the end
        annotation_im = Image.open(annotations[i])
        annotation = np.array(annotation_im)
        
        weight_im = Image.open(boundaries[i])
        weight = np.array(weight_im)
        f = FrameInfo(img=comb_img, annotations=annotation, weight=weight) # problem is not with how this is ordered
        frames.append(f)
    
    # @THADDAEUS maybe problem is with generator? --> I don't think so...most likely prediction tbh??

    training_frames, validation_frames, testing_frames  = split_dataset(frames, frames_json, patch_dir)

    annotation_channels = input_label_channel + input_weight_channel
    train_generator = DataGenerator(input_image_channel, patch_size, training_frames, frames, annotation_channels, augmenter = 'iaa').random_generator(BATCH_SIZE, normalize = normalize) # set augmenter from ''iaa'' to None in case that's messing with things? 
    val_generator = DataGenerator(input_image_channel, patch_size, validation_frames, frames, annotation_channels, augmenter= None).random_generator(BATCH_SIZE, normalize = normalize)
    test_generator = DataGenerator(input_image_channel, patch_size, testing_frames, frames, annotation_channels, augmenter= None).random_generator(BATCH_SIZE, normalize = normalize)

    '''
    # do the for _ in range() here to check if issue is before or after
    from cnnheights.debug_utils import display_images
    train_images, real_label = next(train_generator)
    ann = real_label[:,:,:,0]
    wei = real_label[:,:,:,1]
    #overlay of annotation with boundary to check the accuracy
    #5 images in each row are: pan, ndvi, annotation, weight(boundary), overlay of annotation with weight
    overlay = ann + wei
    overlay = overlay[:,:,:,np.newaxis]
    display_images(np.concatenate((train_images,real_label), axis = -1))
    '''

    return train_generator, val_generator, test_generator

# not in train_cnn (unless the function call to generators causes the problem)
def train_cnn(ndvi_images:list, pan_images:list, annotations:list, boundaries:list, 
              epochs:int=200, training_steps:int=1000, use_multiprocessing:bool=False, 
              logging_dir:str=None,
              crs:str='EPSG:32628'): 

    r'''
    
    __Train the UNET CNN on the extracted data__

    Arguments 
    ---------

    ndvi_images : list 
        List of full file paths to the extracted ndvi images 

    pan_images : list 
        Same as ndvi_images except for pan 

    boundaries : list 
        List of boundary files extracted during the preproccessing step 

    annotations : list
        List of the full file paths to the extracted annotations outputed during the preproccessing step. 
    
    logging_dir : str
        Passed onto the load_train_test and train_model functions; see load_train_test_split docstring for explanation. 

    use_multiprocessing : bool
        Higher level access to turning multiprocessing on or not

    confusion_matrix : bool
        calculate confusion matrix on test set predictions (or not). Note that cm.ravel() for every cm in the returned confusion_matrices will return tn, fp, fn, tp

    crs : str
        e.g. EPSG:32628; should probably be ready from internal files. 

    '''
        
    from cnnheights.training import load_train_test
    import os 
    import numpy as np
    import matplotlib.pyplot as plt 
    import seaborn as sns
    import pandas as pd
    from cnnheights.original_core.losses import tversky, accuracy, dice_coef, dice_loss, specificity, sensitivity 
    from cnnheights.original_core.optimizers import adaDelta 
    import time 
    from functools import reduce 
    from cnnheights.original_core.UNet import UNet 
    from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

    train_generator, val_generator, test_generator = load_train_test(ndvi_images=ndvi_images, pan_images=pan_images, annotations=annotations, boundaries=boundaries, logging_dir=logging_dir)

    BATCH_SIZE = 8
    VALID_IMG_COUNT = 200 # used in validation
    input_shape = (256,256,2)
    input_image_channel = [0,1]
    input_label_channel = [2]
    input_weight_channel = [3]

    OPTIMIZER = adaDelta
    LOSS = tversky 

    # Only for the name of the model in the very end
    OPTIMIZER_NAME = 'AdaDelta'
    LOSS_NAME = 'weightmap_tversky'

    # Declare the path to the final model
    # If you want to retrain an exising model then change the cell where model is declared. 
    # This path is for storing a model after training.

    timestr = time.strftime("%Y%m%d-%H%M")
    chf = input_image_channel + input_label_channel
    chs = reduce(lambda a,b: a+str(b), chf, '')

    if logging_dir is not None: 
        model_dir= os.path.join(logging_dir, 'saved_models/UNet/')
        tensorboard_log_dir = os.path.join(logging_dir, 'logs/')

    else: 
        model_dir = './saved_models/UNet/'
        tensorboard_log_dir = './logs'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(tensorboard_log_dir): 
        os.mkdir(tensorboard_log_dir)

    model_path = os.path.join(model_dir,'trees_{}_{}_{}_{}_{}.h5'.format(timestr,OPTIMIZER_NAME,LOSS_NAME,chs,input_shape[0]))

    # The weights without the model architecture can also be saved. Just saving the weights is more efficent.

    # weight_path="./saved_weights/UNet/{}/".format(timestr)
    # if not os.path.exists(weight_path):
    #     os.makedirs(weight_path)
    # weight_path=weight_path + "{}_weights.best.hdf5".format('UNet_model')
    # print(weight_path)

    # Define the model and compile it
    print('\n')
    print([BATCH_SIZE, *input_shape])
    print('\n')

    model = UNet([BATCH_SIZE, *input_shape],input_label_channel)
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=[dice_coef, dice_loss, specificity, sensitivity, accuracy])

    # Define callbacks for the early stopping of training, LearningRateScheduler and model checkpointing

    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = False)


    log_dir = os.path.join('./logs','UNet_{}_{}_{}_{}_{}'.format(timestr,OPTIMIZER_NAME,LOSS_NAME,chs, input_shape[0]))
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

    callbacks_list = [checkpoint, tensorboard] #reduceLROnPlat is not required with adaDelta

    # do training  

    loss_history = [model.fit(train_generator, 
                            steps_per_epoch=training_steps, 
                            epochs=epochs, 
                            validation_data=val_generator,
                            validation_steps=VALID_IMG_COUNT,
                            callbacks=callbacks_list, use_multiprocessing=use_multiprocessing)] # the generator is not very thread safe

    return model, loss_history[0].history, test_generator
