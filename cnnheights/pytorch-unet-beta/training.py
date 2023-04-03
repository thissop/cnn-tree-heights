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
    from cnnheights.original_core.frame_utilities import FrameInfo, split_dataset
    from cnnheights.original_core.dataset_generator import DataGenerator
    from cnnheights.original_core.config import normalize, batch_size, patch_size, input_image_channel, input_label_channel, input_weight_channel
    ImageFile.LOAD_TRUNCATED_IMAGES = True

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
    train_generator = DataGenerator(input_image_channel, patch_size, training_frames, frames, annotation_channels, augmenter = 'iaa').random_generator(batch_size, normalize = normalize) # set augmenter from ''iaa'' to None in case that's messing with things?
    val_generator = DataGenerator(input_image_channel, patch_size, validation_frames, frames, annotation_channels, augmenter= None).random_generator(batch_size, normalize = normalize)
    test_generator = DataGenerator(input_image_channel, patch_size, testing_frames, frames, annotation_channels, augmenter= None).random_generator(batch_size, normalize = normalize)
        
    return train_generator, val_generator, test_generator

# not in train_cnn (unless the function call to generators causes the problem)
def train_cnn(ndvi_images:list, pan_images:list, annotations:list, boundaries:list,
              epochs:int=200, training_steps:int=1000, use_multiprocessing:bool=False,
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

    '''
       
    from cnnheights.tensorflow.training import load_train_test
    import os
    from cnnheights.original_core.losses import tversky, accuracy, dice_coef, dice_loss, specificity, sensitivity
    from cnnheights.original_core.optimizers import adaDelta
    import time
    from functools import reduce
    from cnnheights.original_core.UNet import UNet
    from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
    from cnnheights.original_core.config import validation_image_count, batch_size, input_shape, input_image_channel, input_label_channel

    train_generator, val_generator, test_generator = load_train_test(ndvi_images=ndvi_images, pan_images=pan_images, annotations=annotations, boundaries=boundaries, logging_dir=logging_dir)

    OPTIMIZER = adaDelta
    LOSS = tversky

    # Only for the name of the model in the very end
    OPTIMIZER_NAME = 'AdaDelta'
    LOSS_NAME = 'weightmap_tversky'

    # do training
    start = time.time()
   
    loss_history = model.fit(train_generator, steps_per_epoch=training_steps, epochs=epochs, validation=val_generator, validation_steps=validation_image_count)

    from cnnheights.pytorch.training import train_model

    train_model(model=model, epochs=epochs, batch_size=args.batch_size, learning_rate=args.lr, device=device, img_scale=args.scale, val_percent=args.val / 100, amp=args.amp)

    elapsed = time.time()-start

    print(f'Elapsed: {elapsed}; Average: {round(elapsed/100, 3)}')

    return model, loss_history.history, test_generator

def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # 1. Create dataset
    
    dataset = None 

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)