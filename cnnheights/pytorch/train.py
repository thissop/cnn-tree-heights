def load_data(input_data_dir:str, num_patches:int=8): 

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
    from PIL import Image
    from PIL import ImageFile
    from sklearn.model_selection import train_test_split
    from torchvision import transforms
    from torch.utils.data import TensorDataset, DataLoader
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
            X[idx][0] = cropped[0]
            X[idx][1] = cropped[1]
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
    batch_size = 1
    loader_args = dict(batch_size=batch_size, num_workers=1, pin_memory=True) # os.cpu_count()
    train_loader = DataLoader(TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train), torch.Tensor(loss_weight_train)), shuffle=True, **loader_args) # Yikes this is messy. clean later. 
    val_loader = DataLoader(TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val), torch.Tensor(loss_weight_val)), shuffle=False, **loader_args)
    test_loader = DataLoader(TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test), torch.Tensor(loss_weight_test)), shuffle=False, **loader_args)

    return train_loader, val_loader, test_loader, (meta_infos_train, meta_infos_val, meta_infos_test)

def train_model(model, train_loader, val_loader, num_epochs:int=25, device:str='cpu'): 
    r'''
    
    '''
    
    from torchsummary import summary # used to be torchsummary
    import torch
    from collections import defaultdict
    from cnnheights.pytorch.loss import calc_loss
    import torch.optim as optim
    from torch.optim import lr_scheduler
    import copy

    #model = pytorch_unet.UNet(2) # shouldn't it only be n_class =2? 
    #model = model.to(device)

    #summary(model, input_size=(2, 1056, 1056))# input_size=(channels, H, W)) # Really mps, but this old summary doesn't support it for some reason

    def print_metrics(metrics, epoch_samples, phase):    
        outputs = []
        for k in metrics.keys():
            outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
            
        print("{}: {}".format(phase, ", ".join(outputs)))    

    def conduct_training(model, optimizer, scheduler, num_epochs=num_epochs):
        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = 1e10

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            
            #since = time.time()

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    if epoch>0: 
                        scheduler.step()
                    for param_group in optimizer.param_groups:
                        print("LR", param_group['lr'])
                        
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                metrics = defaultdict(float)
                epoch_samples = 0
                
                if phase == 'train':
                    dataloader = train_loader
                else: 
                    dataloader = val_loader

                for inputs, labels, loss_weights in dataloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)             

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss, metrics = calc_loss(y_true=labels, y_pred=outputs, weights=loss_weights, metrics=metrics)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward() 
                            optimizer.step()

                    # statistics
                    epoch_samples += inputs.size(0)

                print_metrics(metrics, epoch_samples, phase)
                epoch_loss = metrics['tversky_loss'] / epoch_samples

                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    print("saving best model")
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

            #time_elapsed = time.time() - since
            #print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val loss: {:4f}'.format(best_loss))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adadelta(params=model.parameters(), lr=1.0, rho=0.95)#adaDelta = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=25, gamma=0.1) # tf equivalent? 

    model = conduct_training(model, optimizer=optimizer_ft, scheduler=exp_lr_scheduler, num_epochs=num_epochs)

    return model 