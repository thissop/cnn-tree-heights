import geopandas as gpd
import rasterio 
import matplotlib.pyplot as plt 

fig, axs = plt.subplots(4,4, figsize=(7,7))

for i in range(4): 

    annotation = f'/ar1/PROJ/fjuhsd/personal/thaddaeus/github/cnn-tree-heights/data/input/extracted_annotation_{i}.png'
    boundary = annotation.replace('annotation', 'boundary')
    ndvi = annotation.replace('annotation', 'ndvi')
    pan = annotation.replace('annotation', 'pan')

    files = [annotation, boundary, ndvi, pan]
    titles = ['annotation', 'boundary', 'ndvi', 'pan']
    for j in range(len(files)):
        img = rasterio.open(files[j]).read(1)
        axs[i, j].imshow(img, cmap='magma')
        axs[i, j].set(title=titles[j])

plt.tight_layout()
plt.savefig('/ar1/PROJ/fjuhsd/personal/thaddaeus/github/cnn-tree-heights/debugging-take-2/output/plots/data-check.png', dpi=250)
