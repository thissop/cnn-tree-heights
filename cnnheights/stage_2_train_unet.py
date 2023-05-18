#NOTE(Jesse): The premise of this script is to train a CNN with the provided prepared training data.
# The inputs are assumed to have been produced using the stage_1 script

UNTESTED

training_data_fp = "/path/to/training/data"
model_fp = None #NOTE(Jesse): Set this to the directory of a previously trained UNET for post-training, otherwise we output to the training data fp the trained model.

if __name__ != "__main__":
    print(f"This script {__name__} must be called directly and not imported to be used as a library.  Early exiting.")
    exit()

def main():
    from time import time
    start = time / 60

    from os import listdir
    from os.path import join, isdir, isfile

    global training_data_fp
    global model_fp

    #NOTE(Jesse): Early failure for bad inputs.
    assert isdir(training_data_fp)

    if model_fp:
        assert isfile(model_fp), model_fp

    import rasterio
    from json import dump
    from unet.frame_utilities import FrameInfo
    from numpy import concatenate, transpose, newaxis
    from numpy.random import shuffle

    training_files = listdir(training_data_fp)

    ndvi_fn_template = "extracted_ndvi_{}.png"
    pan_fn_template = "extracted_pan_{}.png"
    boundary_fn_template = "extracted_boundary_{}.png"
    annotation_fn_template = "extracted_annotation_{}.png"

    frames = []
    print("Loading training data")
    for tf in training_files:
        if tf.startswith('.'): #NOTE(Jesse): Skip hidden files
            continue

        if tf.endswith('.vrt'):
            continue

        tf_fn = tf.split('.')[0]
        tf_id = tf_fn.split('_')[-1]
        assert tf_id.isdigit(), f"{tf} the ID number was expected after the last '_' in the file name."

        #NOTE(Jesse): From the ID number we can grab all associated training data.
        #TODO(Jesse): Support .TIF and new training data specification
        ndvi_fp = join(training_data_fp, ndvi_fn_template.format(tf_id))
        pan_fp = join(training_data_fp, pan_fn_template.format(tf_id))
        boundary_fp = join(training_data_fp, boundary_fn_template.format(tf_id))
        annotation_fp = join(training_data_fp, annotation_fn_template.format(tf_id))

        ndvi_img = rasterio.open(ndvi_fp).read(1)
        pan_img = rasterio.open(pan_fp).read(1)
        boundary_img = rasterio.open(boundary_fp).read(1)
        annotation_img = rasterio.open(annotation_fp).read(1)

        comb_img = concatenate((ndvi_img, pan_img), axis=0)
        comb_img = transpose(comb_img, axes=(1,2,0)) #Channel at the end

        frames.append(FrameInfo(img=comb_img, annotations=annotation_img, weight=boundary_img))
    assert len(frames) > 0

    from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
    from unet.dataset_generator import DataGenerator
    from unet.config import input_shape, normalize, patch_size, input_image_channel, input_label_channel, input_weight_channel, batch_size
    from unet.loss import tf_tversky_loss, dice_coef, dice_loss, specificity, sensitivity
    from unet.UNet import UNet
    from unet.optimizers import adaDelta
    from keras.models import load_model

    training_frames_count = int(len(frames) * 0.8)
    shuffled_indices = shuffle(range(len(frames)))
    training_frames = frames[shuffled_indices[:training_frames_count]]
    validation_frames = frames[shuffled_indices[training_frames_count:]]

    frames_fp = join(training_data_fp,'frames_list.json')
    with open(frames_fp, 'w') as f:
        dump({
        'training_frames': training_frames,
        'validation_frames': validation_frames
        }, f)

    annotation_channels = [input_label_channel, input_weight_channel]
    training_augmenter = None #'iaa' #NOTE(Jesse): We'll see if iaa helps later on.
    train_generator = DataGenerator(input_image_channel, patch_size, training_frames, frames, annotation_channels, augmenter =training_augmenter).random_generator(batch_size, normalize = normalize)
    val_generator = DataGenerator(input_image_channel, patch_size, validation_frames, frames, annotation_channels, augmenter= None).random_generator(batch_size, normalize = normalize)

    debug = True
    if debug:
        from unet.visualize import display_images
        train_images, real_label = next(train_generator)
        ann = real_label[:,:,:,0]
        wei = real_label[:,:,:,1]
        #overlay of annotation with boundary to check the accuracy
        #5 images in each row are: pan, ndvi, annotation, weight(boundary), overlay of annotation with weight
        overlay = ann + wei
        overlay = overlay[:,:,:,newaxis]
        display_images(concatenate((train_images, real_label), axis = -1))

    model = None
    if model_fp:
        #TODO(Jesse): Replace load_model with passing a weights file to UNet()
        model = load_model(model_fp, custom_objects={'tversky':tf_tversky_loss,
                                                     'dice_coef':dice_coef, 'dice_loss':dice_loss,
                                                     'specificity':specificity, 'sensitivity':sensitivity})
    else:
        model_fp = join(training_data_fp, "unet_cnn_model.h5")
        model = UNet([batch_size, *input_shape], input_label_channel)

    model.compile(optimizer=adaDelta, loss=tf_tversky_loss, metrics=[dice_coef, dice_loss, specificity, sensitivity, accuracy])

    checkpoint = ModelCheckpoint(model_fp, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only = False)
    tensorboard = TensorBoard(log_dir=training_data_fp, histogram_freq=0, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

    print("Start training")
    model.fit(train_generator,
                epochs=2, steps_per_epoch=10,
                validation_data=val_generator,
                validation_steps=len(validation_frames) * 10,
                callbacks=[checkpoint, tensorboard], use_multiprocessing=False)

    stop = time() / 60
    print(f"Took {stop - start} minutes.")

main()
