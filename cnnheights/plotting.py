import matplotlib.pyplot as plt 
import seaborn as sns

#plt.style.use('https://gist.githubusercontent.com/thissop/d1967ecb352011a4580e2b2274959a89/raw/fe22f835ecb734523e88884bd30c751ca6511cf2/stylish.mplstyle')

sns.set_context("paper") # font_scale=
sns.set_palette('deep') #
seaborn_colors = sns.color_palette('deep') #

plt.rcParams['font.family'] = 'serif'
plt.rcParams["mathtext.fontset"] = "dejavuserif"

dpi = 350

def save_fig(save_path, dpi:float=350):
    if save_path is not None: 
        if save_path.split('.')[0] == 'png': 
            plt.savefig(save_path, dpi=dpi)
        else: 
            plt.savefig(save_path)

        plt.close()

    else: 
        plt.show()

# Preprocessing Related

def plot_shadow_lengths(shadows_gdf, background_tif:str=None, show_lines:bool=True, save_path:str=None, dpi:int=dpi):
    r'''
    
    Notes 
    -----

    TO DO 
    -----

    - give option for 
    

    {'shadow_geometry':annotations_gdf['geometry'], 
         'centroids':centroids,
         'bounds_geometry':[box(*i) for i in square_bounds],
         'heights':heights, 
         'line_geometries':shadow_lines, 
         'lengths':shadow_lengths}

    ''' 

    import rasterio 
    import rasterio.plot 

    fig, ax = plt.subplots()

    if background_tif is not None: 
        raster = rasterio.open(background_tif)
        rasterio.plot.show(raster, ax=ax, cmap='Greys_r') # change raster dim based on 

    shadows_gdf['shadow_geometry'].plot(ax=ax, color='#408ee0')
    #lines_gdf.plot(ax=ax, color='black', linewidth=0.1)
    #centroids.plot(ax=ax, color='indianred', markersize=0.2, zorder=3)

    length_lines = shadows_gdf['line_geometries']
    lengths = shadows_gdf['lengths']
    heights = shadows_gdf['heights']
    centroids = shadows_gdf['centroids']
    for i in range(len(centroids)): 
        p = centroids[i]
        ax.annotate(f'l={round(lengths[i], 1)}\nh={round(heights[i],1)}', xy=(1.000001*p.x, 1.000001*p.y), size='x-small')

    if show_lines: 
        shadows_gdf['line_geometries'].plot(ax=ax, color='black')

    ax.set(xlabel='E', ylabel='N')

    save_fig(save_path, dpi)

def plot_annotations_gallery(shadows_gdf, background_tif:str, polygon_alpha:float=0.5, save_path:str=None, dpi=dpi): 
    r'''
    _Plot one or multiple annotations with shadows_
    
    TO DO 
    -----
    - need to make this more efficient? right now it's replotting the raster for every single axis, and then just clipping the axis' bounsds.
    '''

    import numpy as np
    import rasterio 
    from rasterio.mask import mask 
    from fiona.crs import from_epsg
    import geopandas as gpd 
    import rasterio.plot

    dim = int(np.trunc(np.sqrt(len(shadows_gdf.index)))+1)

    raster = rasterio.open(background_tif)

    epsg = shadows_gdf.crs.to_epsg()
    length = len(shadows_gdf.index)
    fig, axs = plt.subplots(dim, dim, figsize=(6,6))
    for i in range(dim): 
        for j in range(dim): 
            ax = axs[i,j] 
            #ax.set_axis_off()
            ax.set(xticks=[], yticks=[])

            if (length<4 and (i+j)<(length-i)) or (length>=4 and (i+j)<(length-(2*i))): 
                ax = axs[i,j]
                row = shadows_gdf.iloc[i+j]
                bbox = row['bounds_geometry'].bounds

                #inset_background, _ = mask(dataset=raster, shapes=[row['bounds_geometry']], crop=True, filled=False)
                rasterio.plot.show(raster, ax=ax, cmap='Greys_r') # change raster dim based on 
                gpd.GeoSeries(row['shadow_geometry']).plot(ax=ax, alpha=polygon_alpha)
                ax.set(xlim=(0.9999999*bbox[0], 1.0000001*bbox[2]), ylim=(0.9999999*bbox[1], 1.0000001*bbox[3]))
                ax.set_aspect('equal')
                

    plt.tight_layout()

    save_fig(save_path, dpi)

# Post Processing Related

def plot_heights_distribution(shadows_gdf, save_path:str=None, dpi:int=dpi): 
    import geopandas as gpd 

    fig, ax = plt.subplots()

    ax.hist(shadows_gdf['heights'])
    ax.set(xlabel='Tree Height (m)', ylabel='Frequency')

    plt.tight_layout()

    save_fig(save_path, dpi)

# ML Related

def plot_training_diagnostics(loss_history, save_path:str=None):
    r'''
    _Plot diagnostic plots from training process._
    
    Parameters
    ----------      

    loss_history : `dict`
        By default, Ankit's original train_model functionality returned a one-item list with a Keras history object. My version now returns the dictionary from that one item, and that is what should be provided to this function. 

    save_path : `str`
        If defined, plots will be saved at this directory together. 

    Returns
    -------

    figures : `list`
        List of figures 

    '''
    import matplotlib.pyplot as plt 
    import seaborn as sns
    import numpy as np
    import os

    train_keys = ['loss', 'dice_coef', 'dice_loss', 'specificity', 'sensitivity', 'accuracy']

    x = np.arange(0, len(loss_history['loss']))

    figures = []

    for train_key in train_keys:
        fig, ax = plt.subplots()

        val_key = f'val_{train_key}'
        ax.plot(x, loss_history[train_key], label=train_key)
        ax.plot(x, loss_history[val_key], label=val_key)
        ax.legend(loc='best')

        ax.set(xlabel='Training Epoch', ylabel=train_key.title().replace('_', ' '))

        #$plt.tight_layout()

        figures.append(fig)
        
        if save_path is not None: 
            plt.savefig(os.path.join(save_path, f'{train_key}.png'), dpi=200)
            plt.close()
    
    return figures

       