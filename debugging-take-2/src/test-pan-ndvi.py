import geopandas as gpd
import rasterio 
import matplotlib.pyplot as plt 

annotation = '/ar1/PROJ/fjuhsd/personal/thaddaeus/github/cnn-tree-heights/cnn-input/extracted_annotation_0.png'
boundary = annotation.replace('annotation', 'boundary')
ndvi = annotation.replace('annotation', 'ndvi')
pan = annotation.replace('annotation', 'pan')

fig, axs = plt.subplots(4,1, figsize=(2,7))

files = [annotation, boundary, ndvi, pan]
titles = ['annotation', 'boundary', 'ndvi', 'pan']
for i in range(len(files)):
    img = rasterio.open(files[i]).read(1)
    axs[i].imshow(img, cmap='magma')
    axs[i].set(title=titles[i])

plt.tight_layout()
plt.savefig('/ar1/PROJ/fjuhsd/personal/thaddaeus/github/cnn-tree-heights/debugging-take-2/output/plots/data-check.png', dpi=250)
