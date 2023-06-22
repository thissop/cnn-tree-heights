#    Edited by Sizhuo Li
#    Author: Ankit Kariryaa, University of Bremen

import numpy as np
from numpy import nan, nanmean, std, nanstd, mean, float32

import os
import json
#from sklearn.model_selection import train_test_split, KFold

# FROM SPLIT FRAMES #

#Divide the frames into n-splits
if False: #NOTE(Jesse): Unused
    def cross_validation_split(frames, frames_json, patch_dir, n=10):
        """ n-times divide the frames into training, validation and test.

        Args:
            frames: list(FrameInfo)
                list of the all the frames.
            frames_json: str
                Filename of the json where data is written.
            patch_dir: str
                Path to the directory where frame_json is stored.
        """
        if os.path.isfile(frames_json):
            print("Reading n-splits from file")
            with open(frames_json, 'r') as file:
                fjson = json.load(file)
                splits = fjson['splits']
        else:
            print("Creating and writing n-splits to file")
            frames_list = list(range(len(frames)))
            # Divide into n-split, each containing training and test set
            kf = KFold(n_splits=n, shuffle=True, random_state=1117)
            print("Number of spliting iterations:", kf.get_n_splits(frames_list))
            splits = []
            for train_index, test_index in kf.split(frames_list):
                splits.append([train_index.tolist(), test_index.tolist()])
            frame_split = {
                'splits': splits
            }
            if not os.path.exists(patch_dir):
                os.makedirs(patch_dir)
            with open(frames_json, 'w') as f:
                json.dump(frame_split, f)

        return splits

    def split_dataset(frames, frames_json, patch_dir, val_size = 0.2):
        """Divide the frames into training, validation and test.

        Args:
            frames: list(FrameInfo)
                list of the all the frames.
            frames_json: str
                Filename of the json where data is written.
            patch_dir: str
                Path to the directory where frame_json is stored.
            test_size: float, optional
                Percentage of the test set.
            val_size: float, optional
                Percentage of the val set.

        NOTES
        -----

        - Need to incorporate random initialized number
        - Need to not delete frames_json every time? lol

        """

        if os.path.isfile(frames_json):
            #os.remove(frames_json)

            print("Reading train-test split from file")
            with open(frames_json, 'r') as file:
                fjson = json.load(file)
                training_frames = fjson['training_frames']
                validation_frames = fjson['validation_frames']


        else:
            print("Creating and writing train-test split from file")
            frames_list = list(range(len(frames)))
            # Divide into training and val set
            training_frames, validation_frames = train_test_split(frames_list, test_size=val_size)

            frame_split = {
                'training_frames': training_frames,
                'validation_frames': validation_frames
            }

            if not os.path.exists(patch_dir):
                os.makedirs(patch_dir)
            with open(frames_json, 'w') as f:
                json.dump(frame_split, f)

            print('training_frames', training_frames)
            print('validation_frames', validation_frames)

        print(training_frames)
        print(validation_frames)

        return (training_frames, validation_frames)

# FROM FRAME INFO #

def image_normalize(im, axis = (0,1), c = 1e-8):
    '''
    Normalize to zero mean and unit standard deviation along the given axis'''
    return (im - im.mean(axis)) / (im.std(axis) + c)

def standardize_without_resize(i): #NOTE(Jesse): Copied from stage 3
    f_i = i.astype(float32)
    has_nans = f_i == 0
    if has_nans.any():
        f_i[has_nans] = nan

        s_i = (f_i - nanmean(f_i)) / nanstd(f_i)
        s_i[has_nans] = 0
    else:
        s_i = (f_i - mean(f_i)) / std(f_i)

    return s_i

def standardize(i): #NOTE(Jesse): Copied from stage 3
    f_i = i.astype(float32)
    has_nans = f_i == 0
    if has_nans.any():
        f_i[has_nans] = nan

        s_i = (f_i - nanmean(f_i)) / nanstd(f_i)
        s_i[has_nans] = 0
    else:
        s_i = (f_i - mean(f_i)) / std(f_i)

    if s_i.shape != (256, 256): #NOTE(Jesse): Occurs on the last xy step (a partial final step)
        s_i.resize((256, 256), refcheck=False)

    return s_i

# Each area (ndvi, pan, annotation, weight) is represented as an Frame
class FrameInfo:
    """ Defines a frame, includes its constituent images, annotation and weights (for weighted loss).
    """

    def __init__(self, img, annotations, weight, density=None, dtype=np.float32, nonorm=False):
        """FrameInfo constructor.

        Args:
            img: ndarray
                3D array containing various input channels.
            annotations: ndarray
                3D array containing human labels, height and width must be same as img.
            weight: ndarray
                3D array containing weights for certain losses.
            dtype: np.float32, optional
                datatype of the array.
        """
        self.img = img
        self.annotations = annotations
        self.weight = weight
        self.density = density
        self.dtype = dtype
        self.nonorm = nonorm

    # Normalization takes a probability between 0 and 1 that an image will be locally normalized.
    def getPatch(self, i, j, patch_size, img_size, normalize=1.0):
        """Function to get patch from the given location of the given size.

        Args:
            i: int
                Starting location on first dimension (x axis).
            y: int
                Starting location on second dimension (y axis).
            patch_size: tuple(int, int)
                Size of the patch.
            img_size: tuple(int, int)
                Total size of the images from which the patch is generated.
        """
        patch = np.zeros(patch_size, dtype=self.dtype)
    
        im = self.img[i:i + img_size[0], j:j + img_size[1]]
        r = np.random.random(1)
        if self.nonorm is True:
            im = im
        else:
            if normalize >= r[0]:
                im = image_normalize(im, axis=(0, 1))
        an = self.annotations[i:i + img_size[0], j:j + img_size[1]]
        an = np.expand_dims(an, axis=-1)
        we = self.weight[i:i + img_size[0], j:j + img_size[1]]
        we = np.expand_dims(we, axis=-1)
        if self.density is not None:
            den = self.density[i:i + img_size[0], j:j + img_size[1]]
            den = np.expand_dims(den, axis=-1)
            comb_img = np.concatenate((im, an, we, den), axis=-1)
        else:
            comb_img = np.concatenate((im, an, we), axis=-1)
        patch[:img_size[0], :img_size[1], ] = comb_img
        return (patch)

    # Returns all patches in a image, sequentially generated
    def sequential_patches(self, patch_size, step_size, normalize):
        """All sequential patches in this frame.

        Args:
            patch_size: tuple(int, int)
                Size of the patch.
            step_size: tuple(int, int)
                Total size of the images from which the patch is generated.
            normalize: float
                Probability with which a frame is normalized.
        """
        img_shape = self.img.shape
        x = range(0, img_shape[0] - patch_size[0], step_size[0])
        y = range(0, img_shape[1] - patch_size[1], step_size[1])
        if (img_shape[0] <= patch_size[0]):
            x = [0]
        if (img_shape[1] <= patch_size[1]):
            y = [0]

        ic = (min(img_shape[0], patch_size[0]), min(img_shape[1], patch_size[1]))
        xy = [(i, j) for i in x for j in y]
        img_patches = []
        for i, j in xy:
            img_patch = self.getPatch(i, j, patch_size, ic, normalize)
            img_patches.append(img_patch)
        # print(len(img_patches))
        return (img_patches)

    # Returns a single patch, starting at a random image
    def random_patch(self, patch_size, normalize):
        """A random from this frame.

        Args:
            patch_size: tuple(int, int)
                Size of the patch.
            normalize: float
                Probability with which a frame is normalized.
        """
        img_shape = self.img.shape
        if (img_shape[0] <= patch_size[0]):
            x = 0
        else:
            x = np.random.randint(0, img_shape[0] - patch_size[0])
        if (img_shape[1] <= patch_size[1]):
            y = 0
        else:
            y = np.random.randint(0, img_shape[1] - patch_size[1])
        ic = (min(img_shape[0], patch_size[0]), min(img_shape[1], patch_size[1]))
        img_patch = self.getPatch(x, y, patch_size, ic, normalize)
        return (img_patch)
