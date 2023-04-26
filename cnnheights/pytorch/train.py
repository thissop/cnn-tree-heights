def load_data(input_data_dir:str, num_patches:int=8, batch_size:int=8): 

    '''
    
    Arguments
    ---------

    num_patches : int
        - number of 256x256 patches to randomly sample from input images. 
    
    Notes 
    -----

    '''

    from torch.utils.data import DataLoader
    import os 
    import numpy as np
    import torch
    import rasterio 
    from sklearn.model_selection import train_test_split
    from torchvision import transforms
    from torch.utils.data import TensorDataset, DataLoader
    from cnnheights.original_core.frame_utilities import image_normalize
    # ImageFile.LOAD_TRUNCATED_IMAGES = True --> THIS IS BAD!! 

    annotations = [os.path.join(input_data_dir, i) for i in os.listdir(input_data_dir) if 'extracted_annotation' in i]
    boundaries = [i.replace('extracted_annotation', 'extracted_boundary') for i in annotations]
    ndvi_images = [i.replace('extracted_annotation', 'extracted_ndvi') for i in annotations]
    pan_images = [i.replace('extracted_annotation', 'extracted_pan') for i in annotations]

    #trans = transforms.Compose([transforms.ToTensor()])

    dim = rasterio.open(ndvi_images[0]).read().shape[1:]

    N = num_patches*len(ndvi_images)

    X = np.zeros(shape=(N, 2, 256, 256)) # (N, C, H, W)
    y = np.zeros(shape=(N, 1, 256, 256)) # nice! more efficient...NOTE: used to be 2*dim[-1] for this W to account for weights, but to simplify, I'm just doing annotation on output for now. issue I need to fix later. 
    loss_weight = np.zeros(shape=(N, 256, 256))
    meta_infos = []

    # 1. 
    idx = 0 
    for i in range(len(ndvi_images)):

        ndvi_img = rasterio.open(ndvi_images[i])
        meta_info = ndvi_img.meta 
        detected_meta = ndvi_img.meta.copy()
        if 'float' not in detected_meta['dtype']: #The prediction is a float so we keep it as float to be consistent with the prediction. 
            detected_meta['dtype'] = np.float32
        pan_img = rasterio.open(pan_images[i])
        read_ndvi_img = ndvi_img.read()
        read_pan_img = pan_img.read()
      
        # Some augmentations can change the value of y, so we re-assign values just to be sure.
        annotation_im = rasterio.open(annotations[i]).read(1)
        annotation = np.array(annotation_im)
        annotation[annotation<0.5] = 0
        annotation[annotation>=0.5] = 1
        
        #boundaries have a weight of 10 other parts of the image has weight 1
        weight_im = rasterio.open(boundaries[i]).read(1)
        weight = np.array(weight_im)
        weight[weight>=0.5] = 10
        weight[weight<0.5] = 1

        # Sample 256x256 Patches
        combined_for_crop = np.zeros(shape=(4, dim[-2], dim[-1]))
        combined_for_crop[0] = read_ndvi_img 
        combined_for_crop[1] = read_pan_img 
        combined_for_crop[2] = annotation 
        combined_for_crop[3] = weight 

        cropper = transforms.RandomCrop(size=(256, 256))
        combined_for_crop = torch.Tensor(combined_for_crop)

        for _ in range(num_patches):
            cropped = cropper(combined_for_crop)
            X[idx][0] = image_normalize(cropped[0])
            X[idx][1] = image_normalize(cropped[1])
            y[idx] = cropped[2]
            loss_weight[idx] = cropped[3]
            meta_infos.append(meta_info)
            
            idx+=1 

    # 2. Split into train / validation partitions
    test_size = 0.1
    val_size = 0.2

    X_train, X_test, y_train, y_test, loss_weight_train, loss_weight_test, meta_infos_train, meta_infos_test = train_test_split(X, y, loss_weight, meta_infos, test_size=test_size)
    X_train, X_val, y_train, y_val, loss_weight_train, loss_weight_val, meta_infos_train, meta_infos_val = train_test_split(X_train, y_train, loss_weight_train, meta_infos_train, test_size=val_size)

    # 3. Apply Image Augmentation to training data 

    # 4. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=1, pin_memory=True) # os.cpu_count()
    train_loader = DataLoader(TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train), torch.Tensor(loss_weight_train)), shuffle=True, **loader_args) # Yikes this is messy. clean later. 
    val_loader = DataLoader(TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val), torch.Tensor(loss_weight_val)), shuffle=False, **loader_args)
    test_loader = DataLoader(TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test), torch.Tensor(loss_weight_test)), shuffle=False, **loader_args)

    return train_loader, val_loader, test_loader, (meta_infos_train, meta_infos_val, meta_infos_test)

def train_model(model, train_loader, val_loader, num_epochs:int=25, device:str='cpu', output_dir:str=None): 
    r'''
    
    '''
    
    from torchsummary import summary # used to be torchsummary
    import torch
    from collections import defaultdict
    from cnnheights.loss import torch_calc_loss
    import torch.optim as optim
    from torch.optim import lr_scheduler
    import copy
    import numpy as np
    import pandas as pd
    import os

    #model = pytorch_unet.UNet(2) # shouldn't it only be n_class =2? 
    #model = model.to(device)

    #summary(model, input_size=(2, 1056, 1056))# input_size=(channels, H, W)) # Really mps, but this old summary doesn't support it for some reason

    def conduct_training(model, optimizer, scheduler, num_epochs=num_epochs):
        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = 1e10

        # The batch size is a number of samples processed before the model is updated. The number of epochs is the number of complete passes through the training dataset.

        metrics = {'epoch':list(range(num_epochs)), 'train_dice_loss':[], 'train_tversky_loss':[], 'val_dice_loss':[], 'val_tversky_loss':[]}

        for epoch in range(num_epochs):
            #print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            #print('-' * 10)
            
            #since = time.time()
            # you may also want to shuffle the entire dataset on each epoch so no two batch would be the same in the entire training loop
            # Each epoch has a training and validation phase
                
            def temp(phase): 

                if phase == 'train':
                    if epoch>0: 
                        scheduler.step()
                    #for param_group in optimizer.param_groups:
                        #print("LR", param_group['lr'])
                        
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                if phase == 'train':
                    dataloader = train_loader
                else: 
                    dataloader = val_loader

                '''
                for epoch in range(n_epochs):
                    for X_batch, y_batch in loader:
                '''

                tversky_losses = []
                dice_losses = []

                for X_batch, y_batch, loss_weights in dataloader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)  

                    #epoch_samples += X_batch.size(0)

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        y_pred = model(X_batch)
                        loss, epoch_metrics = torch_calc_loss(y_true=y_batch, y_pred=y_pred, weights=loss_weights)
                        tversky_losses.append(epoch_metrics[0])
                        dice_losses.append(epoch_metrics[1])
                        # zero the parameter gradients
                        optimizer.zero_grad()
                        
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                return np.mean(tversky_losses), np.mean(dice_losses)

                # statistics
                #epoch_samples += X_batch.size(0)

            train_losses = temp(phase='train')
            metrics['train_tversky_loss'].append(train_losses[0])
            metrics['train_dice_loss'].append(train_losses[1])
            val_losses = temp(phase='val')
            metrics['val_tversky_loss'].append(val_losses[0])
            metrics['val_dice_loss'].append(val_losses[1])

            # deep copy the model
            if metrics['val_tversky_loss'][-1] < best_loss:
                #print("saving best model")
                best_loss = metrics['val_tversky_loss'][-1] 
                best_model_wts = copy.deepcopy(model.state_dict())

            #time_elapsed = time.time() - since
            #print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        #print('Best val loss: {:4f}'.format(best_loss))

        # load best model weights
        model.load_state_dict(best_model_wts)

        return model, metrics

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adadelta(params=model.parameters(), lr=1.0, rho=0.95)#adaDelta = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=25, gamma=0.1) # tf equivalent? 

    model, metrics = conduct_training(model, optimizer=optimizer_ft, scheduler=exp_lr_scheduler, num_epochs=num_epochs)

    metrics = pd.DataFrame(metrics)
    if output_dir is not None: 
        metrics.to_csv(os.path.join(output_dir, 'train-val-metrics.csv'), index=False)

    return model, metrics