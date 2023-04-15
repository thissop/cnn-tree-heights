def train_model(num_epochs:int=25): 
    from torchsummary import summary # used to be torchsummary
    import torch
    import torch.nn as nn
    import pytorch_unet

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

    def conduct_training(model, optimizer, scheduler, num_epochs=25):
        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = 1e10

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            
            #since = time.time()

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
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
                        loss = calc_loss(y_true=labels, y_pred=outputs, weights=loss_weights, metrics=metrics)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward() # plz work...I think because it's returning torch tensor, I don't need a 
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

    import torch
    import torch.optim as optim
    from torch.optim import lr_scheduler
    import copy

    num_class = 2 # maybe num class was the problem? it was 6 before, and I just changed it to 2, and it seems to be working haha. 

    model = pytorch_unet.UNet(num_class).to(device)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adadelta(lr=1.0, rho=0.95, eps=None)#adaDelta = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=25, gamma=0.1) # tf equivalent? 

    model = conduct_training(model, optimizer=optimizer_ft, scheduler=exp_lr_scheduler, num_epochs=num_epochs)

    return model 