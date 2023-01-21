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
│   ├── __init__.py
│   ├── main.py
│   └── utilities.py
├── data
├── requirements.txt
├── setup.py
└── src
    └── monthly
        └── jan2023
            └── height-from-zennith
                ├── first-zennith.py
                └── height-from-shadow.py
```