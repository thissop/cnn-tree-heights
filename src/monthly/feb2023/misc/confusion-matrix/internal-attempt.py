import os 

boilerplate_output_dir = '/Users/yaroslav/Documents/GitHub/cnn-tree-heights/src/monthly/feb2023/misc/confusion-matrix/boilerplate-output'
cnn_input_dir = '/Users/yaroslav/Documents/Work/NASA/data/old/ready-for-cnn/cnn-input'
frames_json = os.path.join(boilerplate_output_dir, 'frames_list.json')

ndvi_images = [os.path.join(cnn_input_dir, f'extracted_ndvi_{i}.png') for i in range(1,11)]
pan_images = [os.path.join(cnn_input_dir, f'extracted_pan_{i}.png') for i in range(1,11)]
annotations = [os.path.join(cnn_input_dir, f'extracted_annotation_{i}.png') for i in range(1,11)]
boundaries = [os.path.join(cnn_input_dir, f'extracted_boundary_{i}.png') for i in range(1,11)]



