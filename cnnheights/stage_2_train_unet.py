#NOTE(Jesse): The premise of this script is to train a CNN with the provided prepared training data.
# The inputs are assumed to have been produced using the stage_1 script

training_data_fp = "C:\\Users\\jrmeyer3\\Documents\\cnn-input"
model_weights_fp = "C:\\Users\\jrmeyer3\\Documents\\cnn-input\\unet_cnn_model_weights.h5" #NOTE(Jesse): Set this to the directory of a previously trained UNET for post-training, otherwise we output to the training data fp the trained model.

if __name__ != "__main__":
    print(f"This script {__name__} must be called directly and not imported to be used as a library.  Early exiting.")
    exit()

def main():
    from time import time
    start = time() / 60

    from os import listdir
    from os.path import join, isdir, isfile, normpath

    global training_data_fp
    global model_weights_fp

    #NOTE(Jesse): Early failure for bad inputs.
    training_data_fp = normpath(training_data_fp)
    assert isdir(training_data_fp)

    if model_weights_fp:
        model_weights_fp = normpath(model_weights_fp)
        assert isfile(model_weights_fp), model_weights_fp

    import rasterio
    from json import dump
    from unet.frame_utilities import FrameInfo
    from numpy import newaxis, zeros, uint16, float32, arange, concatenate
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

        if '.' not in tf:
            continue #NOTE(Jesse): Skip directories

        if tf.endswith('.vrt'):
            continue

        if tf.endswith('.json'):
            continue

        if tf.endswith('.h5'):
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

        #TODO(Jesse): As referred to above, support the 1024x1024 uint16 pan_ndvi version.
        assert pan_img.shape == (1056, 1056), f"Expected shape of 1056x1056, got {pan_img.shape}"
        assert pan_img.shape == ndvi_img.shape == boundary_img.shape == annotation_img.shape

        arr = zeros((1056, 1056, 2), dtype=float32)
        arr[..., 0] = pan_img
        arr[..., 1] = ndvi_img

        frames.append(FrameInfo(img=arr, annotations=annotation_img, weight=boundary_img))
    assert len(frames) > 0

    print("Loading Tensorflow")
    from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
    from unet.dataset_generator import DataGenerator
    from unet.config import input_shape, normalize, patch_size, input_image_channel, input_label_channel, input_weight_channel, batch_size
    from unet.loss import tversky, dice_coef, dice_loss, specificity, sensitivity, accuracy
    from unet.UNet import UNet
    from unet.optimizers import adaDelta

    shuffled_indices = arange(len(frames))
    shuffle(shuffled_indices)

    training_frames_count = int((len(frames) * 0.8) + 0.5)
    training_frame_indices   = shuffled_indices[:training_frames_count]
    validation_frame_indices = shuffled_indices[training_frames_count:]

    if False:
        #TODO(Jesse): These frame lists are meaningless.  They don't record which image they are from, nor which patch subset is ultimately fed to the CNN.
        # The prior is vital, the latter is maybe situationally useful.
        # So, remember which *image* the frame index refers to.
        frames_fp = join(training_data_fp, 'frames_list.json')
        with open(frames_fp, 'w') as f:
            dump({
            'training_frames_indices': training_frame_indices,
            'validation_frames_indices': validation_frame_indices
            }, f)

    #TODO(Jesse): I don't know why the DataGenerator needs a separate list of indices and the whole frames list.  Just send the frames it needs.
    annotation_channels = [input_label_channel, input_weight_channel]
    training_augmenter = None #'iaa' #NOTE(Jesse): We'll see if iaa helps later on.
    train_generator = DataGenerator(input_image_channel, patch_size, training_frame_indices, frames, annotation_channels, augmenter =training_augmenter).random_generator(batch_size, normalize = normalize)
    val_generator = DataGenerator(input_image_channel, patch_size, validation_frame_indices, frames, annotation_channels, augmenter= None).random_generator(batch_size, normalize = normalize)

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

    model = UNet([batch_size, *input_shape], 1, weight_file=model_weights_fp)
    post_train = False
    if model_weights_fp:
        post_train = True
    else:
        model_weights_fp = join(training_data_fp, "unet_cnn_model_weights.h5")

    model.compile(optimizer=adaDelta, loss=tversky, metrics=[dice_coef, dice_loss, specificity, sensitivity, accuracy])

    checkpoint = ModelCheckpoint(model_weights_fp, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only = True)
    tensorboard = TensorBoard(log_dir=training_data_fp, histogram_freq=0, write_graph=True, write_grads=False, write_images=False,
                              embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

    print("Start training")
    model.fit(train_generator,
                epochs=10, steps_per_epoch=25,
                initial_epoch = 9 if post_train else 0,
                validation_data=val_generator,
                validation_steps=len(validation_frame_indices) * 10,
                callbacks=[checkpoint, tensorboard], use_multiprocessing=False)

    stop = time() / 60
    print(f"Took {stop - start} minutes.")

    if debug:
        batch = zeros((batch_size, 256, 256, 2), dtype=float32)
        for i in range(batch_size):
            batch[i::] = frames[i].img[:256, :256, :] #TODO(Jesse): Use newer input specification & use standardize

        predictions = model.predict(batch)
        for p in predictions:
            p[p > 0.5] = 1
            p[p <= 0.5] = 0

        display_images(predictions)

main()
