def sample_background(input_tif:str, output_path:str, crs:str, key:str=None, counter:int=0, plot_path:str=None, sample_dim:tuple=(1056,1056), dpi=350):
    r'''

    TO DO 
    -----

    Notes
    -----
    - output_path is used for output_tif and output vector rectangle.
    - note for jesse: opening from this window saves crazy memory. 

    '''
    import rasterio 

    # randomly choose left corner 
    # eventually, make sure no overlap with other regions 

    import matplotlib.pyplot as plt
    from rasterio import plot
    from rasterio.windows import transform
    import rasterio
    from shapely.geometry import Polygon
    import os 
    import numpy as np
    import geopandas as gpd 
    from cnnheights.plotting import save_fig
    from blend_modes import lighten_only

    dataset = rasterio.open(input_tif) 

    dims = (dataset.width, dataset.height) 

    extant_key = False 
    if key is not None and os.path.exists(key):
        extant_key = True 

    overlapping = True 
    while overlapping: 

        x1, y1 = (np.random.randint(0,dims[0]-sample_dim[0]), np.random.randint(0,dims[1]-sample_dim[1])) ## FIXX!! 
        x2, y2 = (x1+sample_dim[0], y1+sample_dim[1])

        window = rasterio.windows.Window(x1, y1, sample_dim[0], sample_dim[1]) 
        img = dataset.read(window=window) 
        # read both layers with dataset.read(window=window)
        # read first layer with dataset.read(1, window=window)
        # read second layer with dataset.read(2, window=window)
     
        window_transform = transform(window, dataset.transform) 
        extent = plot.plotting_extent(img, window_transform) 

        extracted_polygon = Polygon(((x1,y1), (x2,y1), (x2, y2), (x1,y2))) 

        if extant_key: 
            key_gdf = gpd.read_file(key)
            extracted_geometries = key_gdf['geometry']

            overlapping = extracted_geometries.overlaps(extracted_polygon)[0]

        else: 
            overlapping = False      
            
    centroid = [np.median(extent[0:2]), np.median(extent[3:5])]

    if extant_key: 
        key_gdf = gpd.read_file(key)
        d = {'geometry':[extracted_polygon]+key_gdf['geometry'].to_list(),
             'centroidsx':[centroid[0]]+key_gdf['centroidsx'].to_list(), 
             'centroidsy':[centroid[1]]+key_gdf['centroidsy'].to_list()}

        key_gdf = gpd.GeoDataFrame(d, crs=crs)

    else: 
        d = {'geometry':[extracted_polygon], 'centroidsx':[centroid[0]], 'centroidsy':[centroid[1]]}
        key_gdf = gpd.GeoDataFrame(d, crs=crs) #lines_gdf = gpd.GeoDataFrame({'geometry':lines}, geometry='geometry', crs=annotations_gdf.crs)
        print(key_gdf)

    if plot_path is not None: 
        # load img here using rasterio
        fig, ax = plt.subplots(figsize=(5,5))

        plot.show(img[0], origin='upper', transform=window_transform, extent=extent, interpolation=None, ax=ax, vmin=250, vmax=750, cmap='Greys_r')
        plot.show(img[1], origin='upper', transform=window_transform, extent=extent, interpolation=None, cmap='Reds', ax=ax, alpha=0.5)
        
        #blended = lighten_only(np.array([img[0].astype(float)]), np.array([img[1].astype(float)]), opacity=0.5)
        #ax.plot(blended)
        
        ax.set_aspect('equal', 'box')
        
        save_fig(plot_path, dpi)

    output_tif = os.path.join(output_path, f'cutout_{counter}.tiff')

    with rasterio.open(output_tif, 'w',
            driver='GTiff', width=sample_dim[0], height=sample_dim[1], count=2,
            dtype=img.dtype, crs=dataset.crs, transform=window_transform) as new_data_set:
        new_data_set.write(img)#, indexes=2)

    vector_rect_path = os.path.join(output_path, f'vector_rectangle_{counter}.gpkg')
    vector_rectangle = gpd.GeoDataFrame({'geometry':[extracted_polygon]}, crs=crs)
    vector_rectangle.to_file(vector_rect_path, driver='GPKG')#, layer='name')

    return key_gdf

'''
with rasterio.open('tests/data/RGB.byte.tif') as src:
    r, g, b = src.read()
'''

background_tif = '/Users/yaroslav/Documents/Work/NASA/data/jesse/big mosaic/big mosaic.tif'
key_path = 'src/monthly/feb2023/sample_generation/key_gdf_file.feather'

for i in range(5):

    output_path = f'/Users/yaroslav/Documents/Work/NASA/data/sampled-cutouts/'
    plot_path = f'src/monthly/feb2023/sample_generation/cutout_{i}.pdf'
    key_gdf = sample_background(background_tif, output_path=output_path, counter=i, key=key_path, crs='epsg:32628', plot_path=plot_path)
    key_gdf.to_file(key_path)

import geopandas as gpd 
gdf = gpd.read_file(key_path)
print(gdf)