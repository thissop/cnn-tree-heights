# cnn-tree-heights

## Usage Notes

### Running main.py

Running as background process (so you can close connection to remote):
1. nohup python main.py > "/ar1/PROJ/fjuhsd/personal/thaddaeus/github/cnn-tree-heights/routine-nohup-log.txt" & 

### Installing the Library

This can be skipped if done before, but I reccomend pulling changes from main and rebuilding your local version of the cnnheights python library every time you do a training run to update it with any chances I might have made. 

1. Clone this repository to machine
2. Make new miniconda environment (Python 3.10). If miniconda is installed, you can do this with the command `conda create --name name_of_env python=python_version` e.g. `conda create --name cnnheights310 python=3.10` Whenever working with my code, please activate this miniconda environment (i.e. `conda activate name_of_env`)
3. Conda install as many libraries as you can in the requirements.txt file; pip install the remaining. 
3. cd into the repository you've cloned. Execute `pip install -e .` Execute `conda develop .` These commands build a local copy of the `cnnheights` library and put it on path for the miniconda environment you're working in. This will also install all the pre-req conda and pip module requirements for everything to work. 

    a. Note that to *uninstall* this Python library, execute `pip uninstall .` and or `conda develop -u .` 
    
4. Make sure you're using correct python environment in VSCode 

## Conventions

### docstrings

```
r'''
_Short Description_  

Parameters
----------      

variable_name : `type`
    Description

Returns
-------

variable_name : `type`
    Description 
'''
```

## Repository Directory

```
.
├── README.md
├── cnnheights
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-310.pyc
│   │   ├── __init__.cpython-38.pyc
│   │   ├── main.cpython-310.pyc
│   │   ├── main.cpython-38.pyc
│   │   ├── plotting.cpython-310.pyc
│   │   ├── plotting.cpython-38.pyc
│   │   ├── utilities.cpython-310.pyc
│   │   └── utilities.cpython-38.pyc
│   ├── main.py
│   ├── original_core
│   │   ├── UNet.py
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── UNet.cpython-310.pyc
│   │   │   ├── UNet.cpython-38.pyc
│   │   │   ├── __init__.cpython-310.pyc
│   │   │   ├── __init__.cpython-38.pyc
│   │   │   ├── dataset_generator.cpython-310.pyc
│   │   │   ├── dataset_generator.cpython-38.pyc
│   │   │   ├── frame_utilities.cpython-310.pyc
│   │   │   ├── frame_utilities.cpython-38.pyc
│   │   │   ├── losses.cpython-310.pyc
│   │   │   ├── losses.cpython-38.pyc
│   │   │   ├── optimizers.cpython-310.pyc
│   │   │   └── optimizers.cpython-38.pyc
│   │   ├── dataset_generator.py
│   │   ├── frame_utilities.py
│   │   ├── losses.py
│   │   └── optimizers.py
│   ├── plotting.py
│   └── utilities.py
├── data
│   └── cnn-input
│       ├── extracted_annotation_0.png
│       ├── extracted_annotation_1.png
│       ├── extracted_annotation_2.png
│       ├── extracted_annotation_3.png
│       ├── extracted_annotation_4.png
│       ├── extracted_annotation_5.png
│       ├── extracted_annotation_6.png
│       ├── extracted_annotation_7.png
│       ├── extracted_annotation_8.png
│       ├── extracted_annotation_9.png
│       ├── extracted_boundary_0.png
│       ├── extracted_boundary_1.png
│       ├── extracted_boundary_2.png
│       ├── extracted_boundary_3.png
│       ├── extracted_boundary_4.png
│       ├── extracted_boundary_5.png
│       ├── extracted_boundary_6.png
│       ├── extracted_boundary_7.png
│       ├── extracted_boundary_8.png
│       ├── extracted_boundary_9.png
│       ├── extracted_ndvi_0.png
│       ├── extracted_ndvi_1.png
│       ├── extracted_ndvi_2.png
│       ├── extracted_ndvi_3.png
│       ├── extracted_ndvi_4.png
│       ├── extracted_ndvi_5.png
│       ├── extracted_ndvi_6.png
│       ├── extracted_ndvi_7.png
│       ├── extracted_ndvi_8.png
│       ├── extracted_ndvi_9.png
│       ├── extracted_pan_0.png
│       ├── extracted_pan_1.png
│       ├── extracted_pan_2.png
│       ├── extracted_pan_3.png
│       ├── extracted_pan_4.png
│       ├── extracted_pan_5.png
│       ├── extracted_pan_6.png
│       ├── extracted_pan_7.png
│       ├── extracted_pan_8.png
│       └── extracted_pan_9.png
├── notes
│   ├── annotation-and-extraction.MD
│   └── qgis-notes.md
├── requirements.txt
├── setup.py
├── src
│   ├── ankit
│   │   ├── SampleAnnotations
│   │   │   ├── training_areas_example.cpg
│   │   │   ├── training_areas_example.dbf
│   │   │   ├── training_areas_example.prj
│   │   │   ├── training_areas_example.shp
│   │   │   ├── training_areas_example.shx
│   │   │   ├── training_polygons_example.cpg
│   │   │   ├── training_polygons_example.dbf
│   │   │   ├── training_polygons_example.prj
│   │   │   ├── training_polygons_example.shp
│   │   │   └── training_polygons_example.shx
│   │   ├── SampleResults-Preprocessing
│   │   │   ├── annotation_0.json
│   │   │   ├── annotation_0.png
│   │   │   ├── annotation_1.json
│   │   │   ├── annotation_1.png
│   │   │   ├── boundary_0.json
│   │   │   ├── boundary_0.png
│   │   │   ├── boundary_1.json
│   │   │   └── boundary_1.png
│   │   └── notebooks
│   │       ├── 1-Preprocessing.ipynb
│   │       ├── 2-UNetTraining.ipynb
│   │       ├── 3-RasterAnalysis.ipynb
│   │       ├── Auxiliary-1-UNetEvaluation.ipynb
│   │       ├── Auxiliary-2-SplitRasterToAnalyse.ipynb
│   │       ├── configTemplate
│   │       │   ├── Preprocessing.py
│   │       │   ├── RasterAnalysis.py
│   │       │   ├── UNetTraining.py
│   │       │   └── __init__.py
│   │       ├── core
│   │       │   ├── UNet.py
│   │       │   ├── __init__.py
│   │       │   ├── dataset_generator.py
│   │       │   ├── frame_info.py
│   │       │   ├── losses.py
│   │       │   ├── optimizers.py
│   │       │   ├── split_frames.py
│   │       │   └── visualize.py
│   │       └── scripts
│   │           ├── compress.sh
│   │           └── gdal_polygonize.py
│   ├── cnnheights.egg-info
│   │   ├── PKG-INFO
│   │   ├── SOURCES.txt
│   │   ├── dependency_links.txt
│   │   ├── requires.txt
│   │   └── top_level.txt
│   ├── jesse
│   │   ├── note.txt
│   │   ├── stage_1.py
│   │   └── tree_heights
│   │       ├── cy_determine_tree_heights.pyx
│   │       └── determine_tree_heights.py
│   ├── monthly
│   │   ├── feb2023
│   │   │   ├── annotation_gallery
│   │   │   │   ├── dev.py
│   │   │   │   ├── first_multi_gallery.pdf
│   │   │   │   └── first_single_gallery.pdf
│   │   │   ├── annotations_over_tiff
│   │   │   │   ├── dev.py
│   │   │   │   ├── first_plot.pdf
│   │   │   │   └── first_plot.png
│   │   │   ├── autosegment-exploring
│   │   │   │   ├── thoughts.txt
│   │   │   │   └── Снимок экрана 2023-02-01 в 13.51.44.png
│   │   │   ├── heights-distribution
│   │   │   │   ├── first_demo.png
│   │   │   │   └── first_demo.py
│   │   │   └── sample_generation
│   │   │       ├── cutout_0.pdf
│   │   │       ├── cutout_1.pdf
│   │   │       ├── cutout_2.pdf
│   │   │       ├── cutout_3.pdf
│   │   │       ├── cutout_4.pdf
│   │   │       ├── dev.py
│   │   │       ├── get-dim.py
│   │   │       └── key_gdf_file.feather
│   │   │           ├── key_gdf_file.cpg
│   │   │           ├── key_gdf_file.dbf
│   │   │           ├── key_gdf_file.prj
│   │   │           ├── key_gdf_file.shp
│   │   │           └── key_gdf_file.shx
│   │   └── jan2023
│   │       ├── compute-shadow-length
│   │       │   ├── dummy.txt
│   │       │   └── first-try.py
│   │       ├── height-from-shadow-length
│   │       │   ├── first-try.py
│   │       │   └── height-from-shadow.py
│   │       ├── library-testing
│   │       │   ├── cnn-training-output
│   │       │   │   └── saved_models
│   │       │   │       └── UNet
│   │       │   ├── first-layer
│   │       │   │   ├── basic.py
│   │       │   │   ├── cutline.py
│   │       │   │   ├── inkling_of_success.png
│   │       │   │   ├── intersection_test.py
│   │       │   │   ├── notes.MD
│   │       │   │   ├── notes.txt
│   │       │   │   ├── plot_tif.png
│   │       │   │   ├── plot_tif.py
│   │       │   │   ├── plotting.py
│   │       │   │   ├── preproccessing.py
│   │       │   │   ├── shadow_lengths.py
│   │       │   │   ├── shadow_lengths_demo_approach.pdf
│   │       │   │   ├── shadow_lengths_demo_approach.png
│   │       │   │   └── shadow_lengths_demo_approach.svg
│   │       │   ├── first_preprocess_test.py
│   │       │   ├── first_training_test.py
│   │       │   ├── preprocessing-output
│   │       │   │   ├── extracted_annotation_0.png
│   │       │   │   ├── extracted_annotation_1.png
│   │       │   │   ├── extracted_annotation_2.png
│   │       │   │   ├── extracted_annotation_3.png
│   │       │   │   ├── extracted_annotation_4.png
│   │       │   │   ├── extracted_annotation_5.png
│   │       │   │   ├── extracted_annotation_6.png
│   │       │   │   ├── extracted_annotation_7.png
│   │       │   │   ├── extracted_annotation_8.png
│   │       │   │   ├── extracted_annotation_9.png
│   │       │   │   ├── extracted_boundary_0.png
│   │       │   │   ├── extracted_boundary_1.png
│   │       │   │   ├── extracted_boundary_2.png
│   │       │   │   ├── extracted_boundary_3.png
│   │       │   │   ├── extracted_boundary_4.png
│   │       │   │   ├── extracted_boundary_5.png
│   │       │   │   ├── extracted_boundary_6.png
│   │       │   │   ├── extracted_boundary_7.png
│   │       │   │   ├── extracted_boundary_8.png
│   │       │   │   ├── extracted_boundary_9.png
│   │       │   │   ├── extracted_ndvi_0.png
│   │       │   │   ├── extracted_ndvi_1.png
│   │       │   │   ├── extracted_ndvi_2.png
│   │       │   │   ├── extracted_ndvi_3.png
│   │       │   │   ├── extracted_ndvi_4.png
│   │       │   │   ├── extracted_ndvi_5.png
│   │       │   │   ├── extracted_ndvi_6.png
│   │       │   │   ├── extracted_ndvi_7.png
│   │       │   │   ├── extracted_ndvi_8.png
│   │       │   │   ├── extracted_ndvi_9.png
│   │       │   │   ├── extracted_pan_0.png
│   │       │   │   ├── extracted_pan_1.png
│   │       │   │   ├── extracted_pan_2.png
│   │       │   │   ├── extracted_pan_3.png
│   │       │   │   ├── extracted_pan_4.png
│   │       │   │   ├── extracted_pan_5.png
│   │       │   │   ├── extracted_pan_6.png
│   │       │   │   ├── extracted_pan_7.png
│   │       │   │   ├── extracted_pan_8.png
│   │       │   │   └── extracted_pan_9.png
│   │       │   ├── training-diagnostic-plots
│   │       │   │   ├── big-batch
│   │       │   │   │   ├── accuracy.png
│   │       │   │   │   ├── dice_coef.png
│   │       │   │   │   ├── dice_loss.png
│   │       │   │   │   ├── loss.png
│   │       │   │   │   ├── sensitivity.png
│   │       │   │   │   └── specificity.png
│   │       │   │   └── first-mini-batch
│   │       │   │       ├── accuracy.png
│   │       │   │       ├── dice_coef.png
│   │       │   │       ├── dice_loss.png
│   │       │   │       ├── loss.png
│   │       │   │       ├── sensitivity.png
│   │       │   │       └── specificity.png
│   │       │   ├── training.ipynb
│   │       │   ├── training_log.txt
│   │       │   └── training_log_big_batch.txt
│   │       ├── meta-info-play
│   │       │   └── first.py
│   │       ├── standardizing-extant-work
│   │       │   ├── checking-annotations.py
│   │       │   ├── fully-dated.py
│   │       │   └── main.py
│   │       └── tree-heights-additional-work
│   └── other
│       └── sort-requirements.py
└── temp.txt

43 directories, 224 files
```