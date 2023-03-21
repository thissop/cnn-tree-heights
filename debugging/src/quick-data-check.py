def check_preprocess_with_plots(): 
    import geopandas as gpd
    import rasterio 
    import matplotlib.pyplot as plt 

    annotation = '/ar1/PROJ/fjuhsd/personal/thaddaeus/github/cnn-tree-heights/data/input/extracted_annotation_0.png'
    boundary = annotation.replace('annotation', 'boundary')
    ndvi = annotation.replace('annotation', 'ndvi')
    pan = annotation.replace('annotation', 'pan')

    fig, axs = plt.subplots(4,1, figsize=(2,7))

    files = [annotation, boundary, ndvi, pan]
    titles = ['annotation', 'boundary', 'ndvi', 'pan']
    for i in range(len(files)):
        img = rasterio.open(files[i]).read(1)
        axs[i].imshow(img)
        axs[i].set(title=titles[i])

    plt.tight_layout()
    plt.savefig('/ar1/PROJ/fjuhsd/personal/thaddaeus/github/cnn-tree-heights/debugging/plots/data-check.png', dpi=250)

def re_preprocess(): 
    from cnnheights.preprocessing import preprocess

    input_dir = ''
    output_dir = '/ar1/PROJ/fjuhsd/personal/thaddaeus/github/cnn-tree-heights/data/input'
    preprocess(input_data_dir=input_dir, output_data_dir=output_dir)