def load_data(input_data_dir:str): 

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
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    annotations = [os.path.join(input_data_dir, f'extracted_annotation_{i}.png') for i in range(4)]
    boundaries = [i.replace('extracted_annotation', 'extracted_boundary') for i in annotations]
    ndvi_images = [i.replace('extracted_annotation', 'extracted_ndvi') for i in annotations]
    pan_images = [i.replace('extracted_annotation', 'extracted_pan') for i in annotations]

    #trans = transforms.Compose([transforms.ToTensor()])

    dim = rasterio.open(ndvi_images[0]).read().shape[1:]

    X = np.zeros(shape=(len(ndvi_images), 2, dim[-2], dim[-1])) # (N, C, X, Y)
    y = np.zeros(shape=(len(ndvi_images), 2, dim[-2], 2*dim[-1])) # nice! more efficient 

    for i in range(len(ndvi_images)):
        ndvi_img = rasterio.open(ndvi_images[i])
        pan_img = rasterio.open(pan_images[i])
        read_ndvi_img = ndvi_img.read()
        read_pan_img = pan_img.read()

        # y would have two channels, i.e. annotations and weights.
        comb_img = np.concatenate((read_ndvi_img, read_pan_img), axis=0)
        #print(comb_img.shape)
        #comb_img = np.transpose(comb_img, axes=(1,2,0)) # original channel was at the end, but I'm putting it towards the begginning to fit pytorch practice? 
        # 
        annotation_im = Image.open(annotations[i])
        annotation = np.array(annotation_im)
        annotation[annotation<0.5] = 0
        annotation[annotation>=0.5] = 1
        
        #boundaries have a weight of 10 other parts of the image has weight 1
        weight_im = Image.open(boundaries[i])
        weight = np.array(weight_im)
        weight[weight>=0.5] = 10
        weight[weight<0.5] = 1

        ann_joint = np.concatenate((annotation,weight), axis=-1) # why does it create them as next to each other left/right? 
        #print(ann_joint.shape)

        X[i] = comb_img
        y[i] = ann_joint

    # 2. Split into train / validation partitions
    test_size = 0.1
    val_size = 0.2

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size)
    
    # 3. Create data loaders
    batch_size = 8 
    loader_args = dict(batch_size=batch_size, num_workers=1, pin_memory=True) # os.cpu_count()
    train_loader = DataLoader(TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train)), shuffle=True, **loader_args) # Yikes this is messy. clean later. 
    val_loader = DataLoader(TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val)), shuffle=False, **loader_args)
    test_loader = DataLoader(TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test)), shuffle=False, **loader_args)

    return train_loader, val_loader, test_loader

#train_loader, val_loader, test_loader = load_data('/Users/yaroslav/Documents/Work/NASA/old/data/first-annotations-push/first-shadows-dataset/')

#inputs, masks = next(iter(train_loader)) 

