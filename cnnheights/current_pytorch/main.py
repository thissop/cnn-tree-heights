def main(output_dir:str,
         input_data_dir:str='/ar1/PROJ/fjuhsd/personal/thaddaeus/github/cnn-tree-heights/first-shadows-dataset', 
         num_epochs:int=1, num_patches:int=8):

    from torchsummary import summary # used to be torchsummary
    import torch
    import matplotlib.pyplot as plt 
    import pytorch_unet
    import os 
    from cnnheights.current_pytorch.utils import load_data
    from cnnheights.current_pytorch.train import train_model
    from cnnheights.current_pytorch.prediction import predict 

    device = 'cpu' 
    if torch.backends.mps.is_available(): 
        device = 'mps'

    elif torch.cuda.is_available(): 
        device = 'cuda'
    device = 'cpu' # fix later...issue somewhere with some of the tensors being cpu, some being mps, even after setting here 
    #print(f"Using device: {device}")

    model = pytorch_unet.UNet(2) # shouldn't it only be n_class =2? 
    model = model.to(device)

    #summary(model, input_size=(2, 1056, 1056))# input_size=(channels, H, W)) # Really mps, but this old summary doesn't support it for some reason
    train_loader, val_loader, test_loader, meta_infos = load_data(input_data_dir=input_data_dir, num_patches=num_patches)
    
    model = train_model(train_loader = train_loader, val_loader=val_loader, num_epochs=num_epochs)

    predictions = predict(model=model, test_loader=test_loader, meta_info=meta_infos)

    plot_dir = os.path.join(output_dir, 'plots')

    for i in range(len(meta_infos)):
        
        inputs, labels, test_loss_weights = next(iter(test_loader))
        inputs = inputs.to(device)
        labels = labels.to(device)
        pred = predictions[i]['pred']
        
        fig, axs = plt.subplots(2,2)

        a = axs[0,0].imshow(pred)
        plt.colorbar(a, ax=axs[0,0])
        axs[0,0].set(title='Prediction')
        b = axs[0,1].imshow(labels[0][0])
        axs[0,1].set(title='Label')
        plt.colorbar(b, ax=axs[0,1])


        c = axs[1,0].imshow(inputs[0][0])
        plt.colorbar(c, ax=axs[1,0])
        axs[1,0].set(title='input[0] (NDVI)')
        d = axs[1,1].imshow(inputs[0][1])
        plt.colorbar(d, ax=axs[1,1])
        axs[1,1].set(title='input[1] (PAN)')

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'prediction_{i}.pdf'))

        plt.clf()
        plt.close()

if __name__ == '__main__':
    main()