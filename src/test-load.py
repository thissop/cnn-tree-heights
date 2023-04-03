from torch.utils.data import DataLoader
import os 
from cnnheights.tensorflow.training import load_train_test

train_generator, val_generator, test_generator = load_train_test()

loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
train_loader = DataLoader(train_set, shuffle=True, **loader_args)
val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)