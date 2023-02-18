from cnnheights.utilities import sample_background
import os 

input_tif = '/Users/yaroslav/Documents/Work/NASA/data/jesse/big mosaic/big mosaic.tif'
key = 'src/monthly/feb2023/first-annotation-push/sample_backgrounds_key.gpkg'

for i in range(1,5):
    output_path = f'/Users/yaroslav/Documents/Work/NASA/data/first-annotations-push/cutout_{i}'
    os.mkdir(output_path)
    plot_path = f'src/monthly/feb2023/first-annotation-push/background_{i}.pdf'
    sample_background(input_tif=input_tif, output_path=output_path, crs='EPSG:32628', counter=i, key=key, plot_path=plot_path)