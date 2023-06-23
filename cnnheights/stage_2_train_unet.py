#NOTE(Jesse): The premise of this script is to train a CNN with the provided prepared training data.
# The inputs are assumed to have been produced using the stage_1 script

training_data_fp = "C:\\Users\\jrmeyer3\\cnn-tree-heights\\training_data"
model_weights_fp = "C:\\Users\\jrmeyer3\\Documents\\cnn-input\\unet_cnn_model_weights.h5" #NOTE(Jesse): Set this to the directory of a previously trained UNET for post-training, otherwise we output to the training data fp the trained model.

model_weights_fp = None

if __name__ != "__main__":
    print(f"This script {__name__} must be called directly and not imported to be used as a library.  Early exiting.")
    exit()

def main():
    from time import time
    start = time() / 60

    from os import listdir, environ
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
    from random import shuffle
    from unet.frame_utilities import standardize_without_resize
    from numpy import newaxis, zeros, float32, uint16, uint8, concatenate, where, sqrt
    from numpy.random import default_rng

    training_files = listdir(training_data_fp)
    assert len(training_files) > 0

    print("Loading training data")

    frames = []
    for tf in training_files:
        if not tf.endswith('.tif'):
            continue

        if not tf[-5].isdigit():
            continue

        assert "cutout" in tf, tf

        pan_ndvi  = zeros((1024, 1024, 2), dtype=float32)
        anno_boun = zeros((1024, 1024, 2), dtype=float32)

        with rasterio.open(join(training_data_fp, tf)) as r_ds:
            assert r_ds.dtypes == ('uint16', 'uint16')
            assert r_ds.shape == (1024, 1024), f"Expected shape of 1024x1024, got {r_ds.shape}"

            pan_ndvi[..., 0] = standardize_without_resize(r_ds.read(1))
            pan_ndvi[..., 1] = standardize_without_resize(r_ds.read(2))
            
        annotation_boundary_img = rasterio.open(join(training_data_fp, tf.replace(".tif", "_annotation_and_boundary.tif"))).read(1)
        assert annotation_boundary_img.shape == (1024, 1024)

        anno_boun[..., 0] = where(annotation_boundary_img == 1, 1, 0)  #NOTE(Jesse): Is there a simpler computation for this?  It's just masking with a scale.
        anno_boun[..., 1] = where(annotation_boundary_img == 2, 10, 1) #NOTE(Jesse): Set boundary pixels to 10, otherwise 1 (as per Ankit's specification)

        frames.append((pan_ndvi, anno_boun, tf))

    frames_count = len(frames)
    assert frames_count > 0

    shuffle(frames)

    training_ratio = 0.8
    validation_ratio = 0.1
    test_ratio = 0.1
    assert training_ratio + validation_ratio + test_ratio == 1.0

    training_frames_count = int((frames_count * training_ratio))
    validation_frames_count = int((frames_count * validation_ratio))
    #NOTE(Jesse): The test frame count is just what's left over from training and validation.

    training_frames   = frames[:training_frames_count]
    validation_frames = frames[training_frames_count:training_frames_count + validation_frames_count]
    test_frames = frames[training_frames_count + validation_frames_count:]

    with open(join(training_data_fp, 'frames_list.json'), 'w') as f:
        dump({
        'base_file_path': training_data_fp,
        'training': [i[2] for i in training_frames],
        'validation': [i[2] for i in validation_frames],
        'test': [i[2] for i in test_frames],
        }, f)

    def uniform_random_patch_generator(patches, normalization_odds=0.0):
        random_gen_count = 1024
        rng = default_rng()
        randint = rng.integers

        patches_count = len(patches)

        batch_xy_size = 128
        patch_offsets = randint(0, 1024 - batch_xy_size, (random_gen_count, 2), dtype=uint16)
        patch_indices = randint(0, patches_count, random_gen_count, dtype=uint16)

        batch_count = 96
        batches_pan_ndvi = zeros((batch_count, batch_xy_size, batch_xy_size, 2), dtype=float32)
        batches_anno_bound = zeros((batch_count, batch_xy_size, batch_xy_size, 2), dtype=float32)

        normalization_odds = int(normalization_odds * 100)
        assert normalization_odds <= 100, normalization_odds

        norm_odds = None
        if normalization_odds > 0:
            norm_odds = randint(0, 100, random_gen_count, dtype=uint8)

        i = 0
        while True:
            for b in range(batch_count):
                (y, x) = patch_offsets[i]
                pan_ndvi, anno_boun, _ = patches[patch_indices[i]]
                pan_ndvi  =  pan_ndvi[y : y + batch_xy_size, x: x + batch_xy_size, ...]
                anno_boun = anno_boun[y : y + batch_xy_size, x: x + batch_xy_size, ...]

                if norm_odds is not None:
                    if norm_odds[i] > normalization_odds:
                        pan_ndvi[..., 0] = standardize_without_resize(pan_ndvi[..., 0])
                        pan_ndvi[..., 1] = standardize_without_resize(pan_ndvi[..., 1])

                batches_pan_ndvi[b, ...]   = pan_ndvi
                batches_anno_bound[b, ...] = anno_boun

                i += 1
                if i == random_gen_count:
                    patch_offsets = randint(0, 1024 - batch_xy_size, (random_gen_count, 2), dtype=uint16)
                    patch_indices = randint(0, patches_count, random_gen_count, dtype=uint16)
                    norm_odds = randint(0, 100, random_gen_count, dtype=uint8)

                    i = 0

            yield batches_pan_ndvi, batches_anno_bound

    train_generator = uniform_random_patch_generator(training_frames)
    val_generator = uniform_random_patch_generator(validation_frames)
    test_generator = uniform_random_patch_generator(test_frames)

    from unet.visualize import display_images
    debug = False
    if debug:
        train_images, real_label = next(train_generator)
        ann = real_label[..., 0]
        wei = real_label[..., 1]
        #overlay of annotation with boundary to check the accuracy
        #5 images in each row are: pan, ndvi, annotation, weight(boundary), overlay of annotation with weight
        overlay = ann + wei
        overlay = overlay[..., newaxis]
        display_images(concatenate((train_images, real_label, overlay), axis = -1), training_data_fp)

    print("Loading Tensorflow")

    environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    environ["TF_GPU_THREAD_MODE"] = "gpu_private" #NOTE(Jesse): Seperate I/O and Compute CPU thread scheduling.

    #TODO(Jesse): Are these necessary if jit_compile=True is specified in model.compile?
    #environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices:--xla_gpu_persistent_cache_dir=C:/Users/jrmeyer3/Desktop/NASA/trees/:"
    #environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"

    from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
    from unet.loss import tversky, dice_coef, dice_loss, specificity, sensitivity, accuracy
    from unet.UNet import UNet
    from unet.optimizers import adaDelta
    from unet.config import batch_size, normalize, input_shape

    model = UNet((batch_size, *input_shape), 1, weight_file=model_weights_fp)
    post_train = False
    if model_weights_fp:
        post_train = True
    else:
        model_weights_fp = join(training_data_fp, "unet_cnn_model_weights.h5")

    model.compile(optimizer=adaDelta, loss=tversky, metrics=[dice_coef, dice_loss, specificity, sensitivity, accuracy], jit_compile=True)

    checkpoint = ModelCheckpoint(model_weights_fp, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only = True)
    tensorboard = TensorBoard(log_dir=training_data_fp, histogram_freq=0, write_graph=True, write_grads=False, write_images=False,
                              embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

    print("Start training")
    model.fit(train_generator,
                epochs=10, steps_per_epoch=100,
                initial_epoch = 9 if post_train else 0,
                validation_data=val_generator,
                validation_steps=int(sqrt(frames_count) + 0.5),
                callbacks=[checkpoint, tensorboard], use_multiprocessing=False)

    stop = time() / 60
    print(f"Took {stop - start} minutes to train.")

    #TODO(Jesse): Analyze test set
    batch, anno = next(test_generator)
    predictions = model.predict(batch, batch_size=batch_size)
    for p in predictions:
        p[p > 0.5] = 1
        p[p <= 0.5] = 0

    display_images(concatenate((batch, predictions, anno), axis = -1), training_data_fp)

main()
