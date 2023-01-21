import os 
from cnnheights import preprocess
from time import time 
import numpy as np

top_dir = '/Users/yaroslav/Documents/GitHub/cnn-tree-heights/data/old/july2022-testing-input/'

ndvi = []
pan = []
backgrounds = []
annotations = [] 

for i in np.sort(os.listdir(top_dir)): 
    path = top_dir+i
    if 'ndvi' in i: 
        ndvi.append(path)

    elif 'pan' in i: 
        pan.append(path)

    elif 'vector' in i:
        backgrounds.append(path)

    elif 'annotations' in i: 
        annotations.append(path) 

def file_only(fp): 
    return fp.split('/')[-1]

#for i in range(len(ndvi)):
#    print([file_only(j) for j in [ndvi[i], pan[i], backgrounds[i], annotations[i]]])

out_dir = '/Users/yaroslav/Documents/GitHub/cnn-tree-heights/src/monthly/jan2023/library-testing/output/'

for i in os.listdir(out_dir):
    os.remove(out_dir+i)

s = time()
preprocess(area_files=backgrounds, annotation_files=annotations, raw_ndvi_images=ndvi, raw_pan_images=pan, output_path=out_dir)
print('New Time:', time()-s)