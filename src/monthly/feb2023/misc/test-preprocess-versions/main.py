from cnnheights import better_preprocess, old_preprocess
import os 
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import time 
import shutil 
from tqdm import tqdm 

input_dir = '/Users/yaroslav/Documents/GitHub/cnn-tree-heights/src/monthly/feb2023/misc/test-preprocess-versions/input'
output_dir = '/Users/yaroslav/Documents/GitHub/cnn-tree-heights/src/monthly/feb2023/misc/test-preprocess-versions/output'

annotations = '/Users/yaroslav/Documents/Work/NASA/old/data/first-annotations-push/individual_cutouts/cutout_3/annotations_3.gpkg'
ndvi = '/Users/yaroslav/Documents/Work/NASA/old/data/first-annotations-push/individual_cutouts/cutout_3/raw_ndvi_3.tif'
pan = '/Users/yaroslav/Documents/Work/NASA/old/data/first-annotations-push/individual_cutouts/cutout_3/raw_pan_3.tif'
vector_rectangle = '/Users/yaroslav/Documents/Work/NASA/old/data/first-annotations-push/individual_cutouts/cutout_3/vector_rectangle_3.gpkg'

test_numbers = [2, 4, 8, 16, 32, 64]
old_times = []
new_times = []

for i in tqdm(test_numbers): 
    n = 0

    for f in os.listdir(input_dir): 
        if f != '.gitignore':
            os.remove(os.path.join(input_dir, f))

    for j in range(i):
        
        shutil.copyfile(annotations, os.path.join(input_dir, f'annotations_{n}.gpkg'))
        shutil.copyfile(ndvi, os.path.join(input_dir, f'ndvi_{n}.tif'))
        shutil.copyfile(pan, os.path.join(input_dir, f'pan_{n}.tif'))
        shutil.copyfile(vector_rectangle, os.path.join(input_dir, f'vector_rectangle_{n}.gpkg'))
        
        n+=1
    
    if i!=test_numbers[-1]:
        for f in os.listdir(output_dir): 
            if f != '.gitignore':
                pass
                #os.remove(os.path.join(output_dir, f))

    old_start = time.time()
    old_preprocess(input_data_dir=input_dir, output_data_dir=output_dir)
    old_times.append(time.time()-old_start)

    if i!=test_numbers[-1]:
        for f in os.listdir(output_dir): 
            if f != '.gitignore':
                pass
                #os.remove(os.path.join(output_dir, f))
    
    new_start = time.time()
    better_preprocess(input_data_dir=input_dir, output_data_dir=output_dir)
    new_times.append(time.time()-new_start)

results_df = pd.DataFrame()
results_df['n']=test_numbers
results_df['old_time'] = old_times
results_df['new_time'] = new_times 

results_df.to_csv('src/monthly/feb2023/misc/test-preprocess-versions/results.csv', index=False)