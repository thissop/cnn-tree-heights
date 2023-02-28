#NB_EPOCHS, MAX_TRAIN_STEPS
import os
from cnnheights.training import train_cnn
from cnnheights.prediction import predict
import numpy as np

ndvi_images = []
pan_images = [] 
annotations = [] 
boundaries = []

# [1] 
# python /ar1/PROJ/fjuhsd/personal/thaddaeus/github/cnn-tree-heights/src/monthly/jan2023/library-testing/first_training_test.py > /ar1/PROJ/fjuhsd/personal/thaddaeus/other/cnn-heights/output/log.txt &

computer = 'm2' # input('m2, wh1, or wsl: ')

if computer == 'm2': 
    data_dir = '/Users/yaroslav/Documents/Work/NASA/data/old/ready-for-cnn/cnn-input/'
    logging_dir = '/Users/yaroslav/Documents/GitHub/cnn-tree-heights/src/monthly/jan2023/library-testing/cnn-training-output'

elif computer == 'wh1': 
    data_dir = '/ar1/PROJ/fjuhsd/personal/thaddaeus/github/cnn-tree-heights/data/cnn-input/'
    logging_dir = '/ar1/PROJ/fjuhsd/personal/thaddaeus/other/cnn-heights/output'

elif computer == 'wsl': 
    data_dir = ''

else:
    raise Exception('Choose correct computer to work on!')

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

model, hist = train_cnn(ndvi_images, pan_images, annotations, boundaries, logging_dir=logging_dir, epochs=1, training_steps=5)

mask = predict(model, ndvi_images[0], pan_images[0])

print(mask)

# [1] 