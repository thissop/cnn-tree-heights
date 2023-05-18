def load_train_val(ndvi_images:list, pan_images:list, annotations:list, boundaries:list, output_dir:str=None, batch_size:int=8):
   
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

    '''

    import os
    import rasterio
    import numpy as np
    from PIL import Image
    from PIL import ImageFile
    from cnnheights.original_core.frame_utilities import FrameInfo, split_dataset
    from cnnheights.original_core.dataset_generator import DataGenerator
    from cnnheights.original_core.config import normalize, patch_size, input_image_channel, input_label_channel, input_weight_channel
    # ImageFile.LOAD_TRUNCATED_IMAGES = True --> THIS IS BAD! 

    if output_dir is not None:
        patch_dir = os.path.join(output_dir, f'patches{patch_size[0]}/')
        plot_dir = os.path.join(output_dir, 'plots')
    else:
        patch_dir = './patches{}'.format(patch_size[0])


   
    frames_json = os.path.join(patch_dir,'frames_list.json')

    # Read all images/frames into memory
    frames = []
    meta_infos = []

    # problem is not in this for loop
    for i in range(len(ndvi_images)):
        ndvi_img = rasterio.open(ndvi_images[i])
        meta_infos.append(ndvi_img.meta)
        pan_img = rasterio.open(pan_images[i])
        read_ndvi_img = ndvi_img.read()
        read_pan_img = pan_img.read()
        comb_img = np.concatenate((read_ndvi_img, read_pan_img), axis=0)
        comb_img = np.transpose(comb_img, axes=(1,2,0)) #Channel at the end
        annotation_im = rasterio.open(annotations[i]).read(1)
        annotation = np.array(annotation_im)
       
        weight_im = rasterio.open(boundaries[i]).read(1)
        weight = np.array(weight_im)
        f = FrameInfo(img=comb_img, annotations=annotation, weight=weight) # problem is not with how this is ordered
        frames.append(f)
   
    (training_frames, validation_frames) = split_dataset(frames, frames_json, patch_dir)

    annotation_channels = input_label_channel + input_weight_channel
    train_generator = DataGenerator(input_image_channel, patch_size, training_frames, frames, annotation_channels, augmenter = 'iaa').random_generator(batch_size, normalize = normalize) # set augmenter from ''iaa'' to None in case that's messing with things?
    val_generator = DataGenerator(input_image_channel, patch_size, validation_frames, frames, annotation_channels, augmenter= None).random_generator(batch_size, normalize = normalize)
    
    # do the for _ in range() here to check if issue is before or after	
    from cnnheights.original_core.visualize import display_images	
    train_images, real_label = next(train_generator)	
    ann = real_label[:,:,:,0]	
    wei = real_label[:,:,:,1]	
    #overlay of annotation with boundary to check the accuracy	
    #5 images in each row are: pan, ndvi, annotation, weight(boundary), overlay of annotation with weight	
    overlay = ann + wei	
    overlay = overlay[:,:,:,np.newaxis]	
    display_images(np.concatenate((train_images,real_label), axis = -1), plot_path=plot_dir)
    	
    return train_generator, val_generator

# not in train_cnn (unless the function call to generators causes the problem)
def train_model(train_generator, val_generator, 
                epochs:int=2, batch_size:int=8, use_multiprocessing:bool=False,
                logging_dir:str=None):

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

    if model is provided, skips training and just returns it. 

    '''
       
    import os
    from cnnheights.original_core.loss import tf_tversky_loss, accuracy, dice_coef, dice_loss, specificity, sensitivity
    from cnnheights.original_core.optimizers import adaDelta
    import time
    from functools import reduce
    from cnnheights.original_core.UNet import UNet
    from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
    from cnnheights.original_core.config import validation_image_count, input_shape, input_image_channel, input_label_channel

    OPTIMIZER = adaDelta
    LOSS = tf_tversky_loss

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
    print([batch_size, *input_shape])
    print('\n')
    model = UNet([batch_size, *input_shape],input_label_channel)
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=[dice_coef, dice_loss, specificity, sensitivity, accuracy])

    # Define callbacks for the early stopping of training, LearningRateScheduler and model checkpointing

    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min', save_weights_only = False)

    log_dir = os.path.join('./logs','UNet_{}_{}_{}_{}_{}'.format(timestr,OPTIMIZER_NAME,LOSS_NAME,chs, input_shape[0]))
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

    callbacks_list = [checkpoint, tensorboard] #reduceLROnPlat is not required with adaDelta

    # do training
    start = time.time()
    # When training with input tensors such as TensorFlow data tensors, the default None is equal to the number of samples in your dataset divided by the batch size
    
    steps_per_epoch = max([int(epochs/batch_size), 2])

    loss_history = model.fit(train_generator,
                            epochs=epochs, steps_per_epoch=steps_per_epoch, # SET THIS! 
                            validation_data=val_generator,
                            validation_steps=validation_image_count,
                            callbacks=callbacks_list, use_multiprocessing=use_multiprocessing) # the generator is not very thread safe

    elapsed = time.time()-start

    print(f'Elapsed: {elapsed}; Average: {round(elapsed/100, 3)}')

    return model, loss_history.history