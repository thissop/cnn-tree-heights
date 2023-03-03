from cnnheights.preprocessing import sample_background
import os 
import geopandas as gpd 
import rasterio 
import warnings

warnings.filterwarnings("ignore")

def make_sample(): 

    input_tif = '/Users/yaroslav/Documents/Work/NASA/data/jesse/big mosaic/big mosaic.tif'

    key = 'src/monthly/feb2023/misc/fix-overlap-log/sample_backgrounds_key.gpkg'

    for i in range(0,3):
        output_dir = f'/Users/yaroslav/Downloads/temp/'
        plot_path = f'src/monthly/feb2023/misc/fix-overlap-log/background_{i}.pdf'
        sample_background(input_tif=input_tif, output_dir=output_dir, crs='EPSG:32628', counter=i, key=key, plot_path=plot_path)

    gdf = gpd.read_file(key) 
    print(gdf)

#make_sample()

# check coords: 
def check_coords(): 
    for i in range(0,3):
        output_dir = f'/Users/yaroslav/Downloads/temp/cutout_{i}/'
        ndvi = rasterio.open(output_dir+f'raw_ndvi_{i}.tif')
        pan = rasterio.open(output_dir+f'raw_pan_{i}.tif')
        cutout = rasterio.open(output_dir+f'cutout_{i}.tif')

        vector_gdf = gpd.read_file(output_dir+f'vector_rectangle_{i}.gpkg')

        print(3*'####\n')
        print([i.bounds for i in [ndvi, pan, cutout]])
        print(vector_gdf['geometry'])
        print(3*'####\n')

        ndvi.close()
        pan.close()
        cutout.close()
check_coords()

