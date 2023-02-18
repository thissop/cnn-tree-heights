# Instructions for Running Training Pipeline

## Setup Environment and Library

This can be skipped if done before, but I reccomend pulling changes from main and rebuilding your local version of the cnnheights python library every time you do a training run to update it with any chances I might have made. 

1. Clone this repository to machine
2. Make new miniconda environment (Python 3.10). If miniconda is installed, you can do this with the command `conda create --name name_of_env python=python_version` e.g. `conda create --name cnnheights310 python=3.10` Whenever working with my code, please activate this miniconda environment (i.e. `conda activate name_of_env`)
3. Conda install as many libraries as you can in the requirements.txt file; pip install the remaining. 
3. cd into the repository you've cloned. Execute `pip install -e .` Execute `conda develop .` These commands build a local copy of the `cnnheights` library and put it on path for the miniconda environment you're working in. This will also install all the pre-req conda and pip module requirements for everything to work. 

```
This issue is addressed in their own documentation here: https://pyproj4.github.io/pyproj/stable/api/datadir.html

Using commands mentioned in their website to identify the path that has been set:

pyproj.datadir.get_data_dir()
pyproj.datadir.set_data_dir()
The correct path in my environment, which took a fair bit to find is as follows:

c:\\Users\\myname\\Anaconda3\\envs\\main\\lib\\site-packages\\pyproj\\proj_dir\\share\\proj
```

## Run Training Pipeline
