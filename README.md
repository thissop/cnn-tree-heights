# cnn-tree-heights

## Usage Notes

### Installing the Library

1. Make sure you've activated the correct conda library and previously installed `conda-build` (you can do this via `conda install conda-build`)
2. Execute `conda develop .` and `pip install -e .` from `./cnn-tree-heights`
    
    a. Note that to *uninstall* this Python library, execute `pip uninstall .` and or `conda develop -u .` 
    
3. Make sure you're using correct python environment in VSCode 

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
│   │   ├── __init__.cpython-38.pyc
│   │   ├── main.cpython-38.pyc
│   │   └── utilities.cpython-38.pyc
│   ├── admin
│   │   └── sort-requirements.py
│   ├── main.py
│   ├── old main.py
│   ├── original_core
│   │   ├── UNet.py
│   │   ├── __init__.py
│   │   ├── frame_utilities.py
│   │   ├── losses.py
│   │   └── optimizers.py
│   └── utilities.py
├── data
│   ├── jesse
│   │   ├── example_traning
│   │   │   ├── arthur_training_data
│   │   │   │   ├── arthur
│   │   │   │   │   ├── annotation_and_boundary_1.tif
│   │   │   │   │   ├── annotation_and_boundary_10.tif
│   │   │   │   │   ├── annotation_and_boundary_2.tif
│   │   │   │   │   ├── annotation_and_boundary_2.tif.aux.xml
│   │   │   │   │   ├── annotation_and_boundary_3.tif
│   │   │   │   │   ├── annotation_and_boundary_4.tif
│   │   │   │   │   ├── annotation_and_boundary_5.tif
│   │   │   │   │   ├── annotation_and_boundary_6.tif
│   │   │   │   │   ├── annotation_and_boundary_7.tif
│   │   │   │   │   ├── annotation_and_boundary_8.tif
│   │   │   │   │   ├── annotation_and_boundary_9.tif
│   │   │   │   │   ├── arthur_training_area_1.gpkg
│   │   │   │   │   ├── arthur_training_area_1.tif
│   │   │   │   │   ├── arthur_training_area_1.tif.aux.xml
│   │   │   │   │   ├── arthur_training_area_10.gpkg
│   │   │   │   │   ├── arthur_training_area_10.tif
│   │   │   │   │   ├── arthur_training_area_10.tif.aux.xml
│   │   │   │   │   ├── arthur_training_area_2.gpkg
│   │   │   │   │   ├── arthur_training_area_2.tif
│   │   │   │   │   ├── arthur_training_area_2.tif.aux.xml
│   │   │   │   │   ├── arthur_training_area_3.gpkg
│   │   │   │   │   ├── arthur_training_area_3.tif
│   │   │   │   │   ├── arthur_training_area_3.tif.aux.xml
│   │   │   │   │   ├── arthur_training_area_4.gpkg
│   │   │   │   │   ├── arthur_training_area_4.tif
│   │   │   │   │   ├── arthur_training_area_4.tif.aux.xml
│   │   │   │   │   ├── arthur_training_area_5.gpkg
│   │   │   │   │   ├── arthur_training_area_5.tif
│   │   │   │   │   ├── arthur_training_area_5.tif.aux.xml
│   │   │   │   │   ├── arthur_training_area_6.gpkg
│   │   │   │   │   ├── arthur_training_area_6.tif
│   │   │   │   │   ├── arthur_training_area_6.tif.aux.xml
│   │   │   │   │   ├── arthur_training_area_7.gpkg
│   │   │   │   │   ├── arthur_training_area_7.tif
│   │   │   │   │   ├── arthur_training_area_7.tif.aux.xml
│   │   │   │   │   ├── arthur_training_area_8.gpkg
│   │   │   │   │   ├── arthur_training_area_8.tif
│   │   │   │   │   ├── arthur_training_area_8.tif.aux.xml
│   │   │   │   │   ├── arthur_training_area_9.gpkg
│   │   │   │   │   ├── arthur_training_area_9.tif
│   │   │   │   │   ├── arthur_training_area_9.tif.aux.xml
│   │   │   │   │   ├── arthur_training_area_patch_1.gpkg
│   │   │   │   │   ├── arthur_training_area_patch_10.gpkg
│   │   │   │   │   ├── arthur_training_area_patch_2.gpkg
│   │   │   │   │   ├── arthur_training_area_patch_3.gpkg
│   │   │   │   │   ├── arthur_training_area_patch_4.gpkg
│   │   │   │   │   ├── arthur_training_area_patch_5.gpkg
│   │   │   │   │   ├── arthur_training_area_patch_6.gpkg
│   │   │   │   │   ├── arthur_training_area_patch_7.gpkg
│   │   │   │   │   ├── arthur_training_area_patch_8.gpkg
│   │   │   │   │   ├── arthur_training_area_patch_9.gpkg
│   │   │   │   │   ├── getting_annotations.qgz
│   │   │   │   │   ├── ndvi_arthur_training_area_1.tif
│   │   │   │   │   ├── ndvi_arthur_training_area_10.tif
│   │   │   │   │   ├── ndvi_arthur_training_area_10.tif.aux.xml
│   │   │   │   │   ├── ndvi_arthur_training_area_2.tif
│   │   │   │   │   ├── ndvi_arthur_training_area_2.tif.aux.xml
│   │   │   │   │   ├── ndvi_arthur_training_area_3.tif
│   │   │   │   │   ├── ndvi_arthur_training_area_3.tif.aux.xml
│   │   │   │   │   ├── ndvi_arthur_training_area_4.tif
│   │   │   │   │   ├── ndvi_arthur_training_area_4.tif.aux.xml
│   │   │   │   │   ├── ndvi_arthur_training_area_5.tif
│   │   │   │   │   ├── ndvi_arthur_training_area_5.tif.aux.xml
│   │   │   │   │   ├── ndvi_arthur_training_area_6.tif
│   │   │   │   │   ├── ndvi_arthur_training_area_6.tif.aux.xml
│   │   │   │   │   ├── ndvi_arthur_training_area_7.tif
│   │   │   │   │   ├── ndvi_arthur_training_area_7.tif.aux.xml
│   │   │   │   │   ├── ndvi_arthur_training_area_8.tif
│   │   │   │   │   ├── ndvi_arthur_training_area_8.tif.aux.xml
│   │   │   │   │   ├── ndvi_arthur_training_area_9.tif
│   │   │   │   │   ├── ndvi_arthur_training_area_9.tif.aux.xml
│   │   │   │   │   ├── pan_arthur_training_area_1.tif
│   │   │   │   │   ├── pan_arthur_training_area_1.tif.aux.xml
│   │   │   │   │   ├── pan_arthur_training_area_10.tif
│   │   │   │   │   ├── pan_arthur_training_area_10.tif.aux.xml
│   │   │   │   │   ├── pan_arthur_training_area_2.tif
│   │   │   │   │   ├── pan_arthur_training_area_2.tif.aux.xml
│   │   │   │   │   ├── pan_arthur_training_area_3.tif
│   │   │   │   │   ├── pan_arthur_training_area_3.tif.aux.xml
│   │   │   │   │   ├── pan_arthur_training_area_4.tif
│   │   │   │   │   ├── pan_arthur_training_area_4.tif.aux.xml
│   │   │   │   │   ├── pan_arthur_training_area_5.tif
│   │   │   │   │   ├── pan_arthur_training_area_5.tif.aux.xml
│   │   │   │   │   ├── pan_arthur_training_area_6.tif
│   │   │   │   │   ├── pan_arthur_training_area_6.tif.aux.xml
│   │   │   │   │   ├── pan_arthur_training_area_7.tif
│   │   │   │   │   ├── pan_arthur_training_area_7.tif.aux.xml
│   │   │   │   │   ├── pan_arthur_training_area_8.tif
│   │   │   │   │   ├── pan_arthur_training_area_8.tif.aux.xml
│   │   │   │   │   ├── pan_arthur_training_area_9.tif
│   │   │   │   │   ├── pan_arthur_training_area_9.tif.aux.xml
│   │   │   │   │   └── site.gpkg
│   │   │   │   └── test
│   │   │   │       ├── arthur_preprocess_patches.gpkg
│   │   │   │       └── arthur_preprocess_polygons.gpkg
│   │   │   └── vector_rectangles
│   │   │       ├── annotation_and_boundary_1.tif
│   │   │       ├── annotation_and_boundary_10.tif
│   │   │       ├── annotation_and_boundary_2.tif
│   │   │       ├── annotation_and_boundary_3.tif
│   │   │       ├── annotation_and_boundary_4.tif
│   │   │       ├── annotation_and_boundary_5.tif
│   │   │       ├── annotation_and_boundary_6.tif
│   │   │       ├── annotation_and_boundary_7.tif
│   │   │       ├── annotation_and_boundary_8.tif
│   │   │       ├── annotation_and_boundary_9.tif
│   │   │       ├── extracted_annotation_0.png
│   │   │       ├── extracted_annotation_1.png
│   │   │       ├── extracted_annotation_1.png.aux.xml
│   │   │       ├── extracted_annotation_2.png
│   │   │       ├── extracted_annotation_3.png
│   │   │       ├── extracted_annotation_4.png
│   │   │       ├── extracted_annotation_5.png
│   │   │       ├── extracted_annotation_6.png
│   │   │       ├── extracted_annotation_7.png
│   │   │       ├── extracted_annotation_8.png
│   │   │       ├── extracted_annotation_9.png
│   │   │       ├── extracted_boundary_0.png
│   │   │       ├── extracted_boundary_1.png
│   │   │       ├── extracted_boundary_2.png
│   │   │       ├── extracted_boundary_3.png
│   │   │       ├── extracted_boundary_4.png
│   │   │       ├── extracted_boundary_5.png
│   │   │       ├── extracted_boundary_6.png
│   │   │       ├── extracted_boundary_7.png
│   │   │       ├── extracted_boundary_8.png
│   │   │       ├── extracted_boundary_9.png
│   │   │       ├── extracted_ndvi_0.png
│   │   │       ├── extracted_ndvi_1.png
│   │   │       ├── extracted_ndvi_1.png.aux.xml
│   │   │       ├── extracted_ndvi_2.png
│   │   │       ├── extracted_ndvi_3.png
│   │   │       ├── extracted_ndvi_4.png
│   │   │       ├── extracted_ndvi_5.png
│   │   │       ├── extracted_ndvi_6.png
│   │   │       ├── extracted_ndvi_7.png
│   │   │       ├── extracted_ndvi_8.png
│   │   │       ├── extracted_ndvi_9.png
│   │   │       ├── extracted_pan_0.png
│   │   │       ├── extracted_pan_1.png
│   │   │       ├── extracted_pan_2.png
│   │   │       ├── extracted_pan_3.png
│   │   │       ├── extracted_pan_4.png
│   │   │       ├── extracted_pan_5.png
│   │   │       ├── extracted_pan_6.png
│   │   │       ├── extracted_pan_7.png
│   │   │       ├── extracted_pan_8.png
│   │   │       ├── extracted_pan_9.png
│   │   │       ├── thaddaeus_vector_rectangle_1.gpkg
│   │   │       ├── thaddaeus_vector_rectangle_10.gpkg
│   │   │       ├── thaddaeus_vector_rectangle_2.gpkg
│   │   │       ├── thaddaeus_vector_rectangle_3.gpkg
│   │   │       ├── thaddaeus_vector_rectangle_4.gpkg
│   │   │       ├── thaddaeus_vector_rectangle_5.gpkg
│   │   │       ├── thaddaeus_vector_rectangle_6.gpkg
│   │   │       ├── thaddaeus_vector_rectangle_7.gpkg
│   │   │       ├── thaddaeus_vector_rectangle_8.gpkg
│   │   │       └── thaddaeus_vector_rectangle_9.gpkg
│   │   └── example_traning.zip
│   └── old
│       └── july2022-testing-input
│           ├── ndvi_thaddaeus_training_area_1.tif
│           ├── ndvi_thaddaeus_training_area_10.tif
│           ├── ndvi_thaddaeus_training_area_2.tif
│           ├── ndvi_thaddaeus_training_area_3.tif
│           ├── ndvi_thaddaeus_training_area_4.tif
│           ├── ndvi_thaddaeus_training_area_5.tif
│           ├── ndvi_thaddaeus_training_area_6.tif
│           ├── ndvi_thaddaeus_training_area_7.tif
│           ├── ndvi_thaddaeus_training_area_8.tif
│           ├── ndvi_thaddaeus_training_area_9.tif
│           ├── pan_thaddaeus_training_area_1.tif
│           ├── pan_thaddaeus_training_area_10.tif
│           ├── pan_thaddaeus_training_area_2.tif
│           ├── pan_thaddaeus_training_area_3.tif
│           ├── pan_thaddaeus_training_area_4.tif
│           ├── pan_thaddaeus_training_area_5.tif
│           ├── pan_thaddaeus_training_area_6.tif
│           ├── pan_thaddaeus_training_area_7.tif
│           ├── pan_thaddaeus_training_area_8.tif
│           ├── pan_thaddaeus_training_area_9.tif
│           ├── thaddaeus_training_annotations_1.gpkg
│           ├── thaddaeus_training_annotations_10.gpkg
│           ├── thaddaeus_training_annotations_2.gpkg
│           ├── thaddaeus_training_annotations_3.gpkg
│           ├── thaddaeus_training_annotations_4.gpkg
│           ├── thaddaeus_training_annotations_5.gpkg
│           ├── thaddaeus_training_annotations_6.gpkg
│           ├── thaddaeus_training_annotations_7.gpkg
│           ├── thaddaeus_training_annotations_8.gpkg
│           ├── thaddaeus_training_annotations_9.gpkg
│           ├── thaddaeus_vector_rectangle_1.gpkg
│           ├── thaddaeus_vector_rectangle_10.gpkg
│           ├── thaddaeus_vector_rectangle_2.gpkg
│           ├── thaddaeus_vector_rectangle_3.gpkg
│           ├── thaddaeus_vector_rectangle_4.gpkg
│           ├── thaddaeus_vector_rectangle_5.gpkg
│           ├── thaddaeus_vector_rectangle_6.gpkg
│           ├── thaddaeus_vector_rectangle_7.gpkg
│           ├── thaddaeus_vector_rectangle_8.gpkg
│           └── thaddaeus_vector_rectangle_9.gpkg
├── notes
│   └── 1-20-2023.txt
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
│   └── monthly
│       └── jan2023
│           ├── height-from-zennith
│           │   ├── first-zennith.py
│           │   └── height-from-shadow.py
│           └── library-testing
│               ├── first_preprocess_test.py
│               ├── first_training_test.py
│               └── output
│                   ├── extracted_annotation_0.png
│                   ├── extracted_annotation_1.png
│                   ├── extracted_annotation_2.png
│                   ├── extracted_annotation_3.png
│                   ├── extracted_annotation_4.png
│                   ├── extracted_annotation_5.png
│                   ├── extracted_annotation_6.png
│                   ├── extracted_annotation_7.png
│                   ├── extracted_annotation_8.png
│                   ├── extracted_annotation_9.png
│                   ├── extracted_boundary_0.png
│                   ├── extracted_boundary_1.png
│                   ├── extracted_boundary_2.png
│                   ├── extracted_boundary_3.png
│                   ├── extracted_boundary_4.png
│                   ├── extracted_boundary_5.png
│                   ├── extracted_boundary_6.png
│                   ├── extracted_boundary_7.png
│                   ├── extracted_boundary_8.png
│                   ├── extracted_boundary_9.png
│                   ├── extracted_ndvi_0.png
│                   ├── extracted_ndvi_1.png
│                   ├── extracted_ndvi_2.png
│                   ├── extracted_ndvi_3.png
│                   ├── extracted_ndvi_4.png
│                   ├── extracted_ndvi_5.png
│                   ├── extracted_ndvi_6.png
│                   ├── extracted_ndvi_7.png
│                   ├── extracted_ndvi_8.png
│                   ├── extracted_ndvi_9.png
│                   ├── extracted_pan_0.png
│                   ├── extracted_pan_1.png
│                   ├── extracted_pan_2.png
│                   ├── extracted_pan_3.png
│                   ├── extracted_pan_4.png
│                   ├── extracted_pan_5.png
│                   ├── extracted_pan_6.png
│                   ├── extracted_pan_7.png
│                   ├── extracted_pan_8.png
│                   └── extracted_pan_9.png
└── temp.txt

31 directories, 305 files
```