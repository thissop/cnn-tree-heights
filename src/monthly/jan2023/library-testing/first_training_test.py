import os
from cnnheights import train_cnn
import numpy as np

ndvi_images = []
pan_images = [] 
annotations = []
boundaries = []


computer = input('"wh1", "wsl", or "m2"')

if computer == 'm2': 
    data_dir = '/Users/yaroslav/Documents/Work/NASA/data/first_mosaic/rebuilt_approach/output/'

elif computer == 'wh1': 
    data_dir = '/ar1/PROJ/fjuhsd/personal/thaddaeus/github/cnn-tree-heights/data/cnn-input/'

for file in np.sort(os.listdir(data_dir)):
    full_path = data_dir+file
    if '.png' in file: 
        if 'annotation' in file: 
            annotations.append(full_path) 

        elif 'boundary' in file: 
            boundaries.append(full_path) 

        elif 'ndvi' in file: 
            ndvi_images.append(full_path) 

        elif 'extracted_pan' in file: 
            pan_images.append(full_path) 

for i in [ndvi_images, pan_images, annotations, boundaries]: 
    print(len(i))

train_cnn(ndvi_images, pan_images, annotations, boundaries)