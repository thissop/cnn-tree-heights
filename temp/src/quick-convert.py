def convert(): 

    files = ['data/standalone-fake/raster_annotation_1.png',
            'data/standalone-fake/raster_boundary_1.png',
            'data/test-dataset/raster_annotation_1.png',
            'data/test-dataset/raster_boundary_1.png',
            'data/test-dataset/raster_annotation_4.png',
            'data/test-dataset/raster_boundary_4.png']

    from PIL import Image
    import os 
    import cv2

    for i in files: 
        image = cv2.imread(i)
        cv2.imwrite(i.replace('.png', '.tiff'),image)
        os.remove(i)

def plot(): 

    import matplotlib.pyplot as plt 
    import rasterio 

    files = ['data/standalone-fake/raster_annotation_1.png',
            'data/standalone-fake/raster_boundary_1.png',
            'data/test-dataset/raster_annotation_1.png',
            'data/test-dataset/raster_boundary_1.png',
            'data/test-dataset/raster_annotation_4.png',
            'data/test-dataset/raster_boundary_4.png']
    
    files = [i.replace('.png', '.tiff') for i in files]

    fig, axs = plt.subplots(1,6, figsize=(36,3))
    
    for i in range(len(files)): 
        axs[i].imshow(rasterio.open(files[i]).read(1))

    plt.savefig('temp/src/temp.png')

plot()