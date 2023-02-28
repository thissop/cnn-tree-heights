import rasterio 
import numpy 
from PIL import Image
import numpy as np
from PIL import ImageFile
import os 
ImageFile.LOAD_TRUNCATED_IMAGES = True
from cnnheights.original_core.frame_utilities import FrameInfo, split_dataset
from cnnheights.original_core.dataset_generator import DataGenerator


BATCH_SIZE = 8
NB_EPOCHS = 21
VALID_IMG_COUNT = 1
MAX_TRAIN_STEPS = 500
input_shape = (256,256,2)
input_image_channel = [0,1]
input_label_channel = [2]
input_weight_channel = [3]
normalize:float = 0.4 
patch_size=(256,256,4)

frames_json = '/Users/yaroslav/Documents/GitHub/cnn-tree-heights/src/monthly/feb2023/misc/confusion-matrix/boilerplate-output/frames_list.json'

patch_dir = '/Users/yaroslav/Documents/GitHub/cnn-tree-heights/src/monthly/feb2023/misc/confusion-matrix/boilerplate-output'

cnn_input_dir = '/Users/yaroslav/Documents/Work/NASA/data/old/ready-for-cnn/cnn-input'

ndvi_images = [os.path.join(cnn_input_dir, f'extracted_ndvi_{i}.png') for i in range(10)]
pan_images = [os.path.join(cnn_input_dir, f'extracted_pan_{i}.png') for i in range(10)]
annotations = [os.path.join(cnn_input_dir, f'extracted_annotation_{i}.png') for i in range(10)]
boundaries = [os.path.join(cnn_input_dir, f'extracted_boundary_{i}.png') for i in range(10)]


frames = []

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
    f = FrameInfo(comb_img, annotation, weight)
    frames.append(f)

training_frames, validation_frames, testing_frames  = split_dataset(frames, frames_json, patch_dir)

annotation_channels = input_label_channel + input_weight_channel
train_generator = DataGenerator(input_image_channel, patch_size, training_frames, frames, annotation_channels, augmenter = 'iaa').random_generator(BATCH_SIZE, normalize = normalize)
val_generator = DataGenerator(input_image_channel, patch_size, validation_frames, frames, annotation_channels, augmenter= None).random_generator(BATCH_SIZE, normalize = normalize)
test_generator = DataGenerator(input_image_channel, patch_size, testing_frames, frames, annotation_channels, augmenter= None).random_generator(BATCH_SIZE, normalize = normalize)

print(test_generator)