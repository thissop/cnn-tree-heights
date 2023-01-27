import os
from cnnheights import train_cnn
import numpy as np

ndvi_images = []
pan_images = [] 
annotations = [] 
boundaries = []

data_dir = '/Users/yaroslav/Documents/Work/NASA/data/first_mosaic/rebuilt_approach/output/'
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

train_cnn(ndvi_images, pan_images, annotations, boundaries, logging_dir='/Users/yaroslav/Documents/GitHub/cnn-tree-heights/src/monthly/jan2023/library-testing/cnn-training-output')