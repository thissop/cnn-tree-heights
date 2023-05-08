def get_heights(annotations_gdf, d:float=7, cutlines_shp_file:str=None, cutline_info:dict=None):
    r'''
    
    either cutlines_shp_file needs to be a defined file path, or cutline info needs to be a dictionary that would have been collected from the cutlines shp file.
    cutline_info needs to have the following keys if it is used over the cutlines file: ['SUN_ELEV', 'SUN_AZ']

    '''
    
    from cnnheights.utilities import height_from_shadow
    from cnnheights.preprocessing import get_cutline_data
    import numpy as np
    from shapely.geometry import LineString, box
    import geopandas as gpd
    from cnnheights.utilities import height_from_shadow
    import pandas as pd
    from shapely.validation import make_valid

    if cutlines_shp_file is None and cutline_info is None: 
        raise Exception('')

    cutline_info = get_cutline_data(predictions=annotations_gdf, cutlines_shp=cutlines_shp_file)
    dy = np.abs(d/np.tan(np.radians(cutline_info['SUN_AZ'])))
    zenith_angle = cutline_info['SUN_ELEV']

    annotation_centroids = annotations_gdf.centroid
    annotation_lines = [LineString([(x-d, y+dy), (x+d, y-dy)]) for x, y in zip(annotation_centroids.x, annotation_centroids.y)]
    annotation_lines_gdf = gpd.GeoDataFrame({'geometry':annotation_lines}, geometry='geometry', crs=f'EPSG:{annotations_gdf.crs.to_epsg()}')
         
    annotations_gdf['geometry'] = [make_valid(i) for i in annotations_gdf['geometry']]
    #annotations_gdf['geometry'] = annotations_gdf.buffer(0) # DON'T DO THIS!
    
    annotation_shadow_lines = annotation_lines_gdf.intersection(annotations_gdf, align=False)
    annotation_shadow_lengths = annotation_shadow_lines.length
    annotations_shadow_heights = height_from_shadow(annotation_shadow_lengths, zenith_angle=zenith_angle)
    
    if type(annotations_shadow_heights) is pd.Series: 
        annotations_shadow_heights = annotations_shadow_heights.to_numpy()

    return annotations_shadow_heights

def height_from_shadow(shadow_length:float, zenith_angle:float):
    r'''
    _get height from shadow length and zenith angle_


    TO DO 
    -----

    - need to give user information about what way zenith angle should be oriented?
    
    '''
    import numpy as np
    
    height = shadow_length*np.tan(np.radians(zenith_angle)) # does this need to get corrected for time zone? 
     # H = L tan (x), where x is solar elevation angle from ground? 
    return height 

def zenith_from_location(time:str, lat:float, lon:float): 
    r'''
    _get zenith angle for calculations based on time and location_  

    Parameters
    ----------      

    time : `str`
        Time in `"yyyy-mm-dd hh:mm:ss"` format

    lat : `float`

    lon : `float`

    Returns
    -------

    zenith : `float`
        Angle of sun's zenith.


    TO DO 
    -----
    - make it get azimuth to!
     
    '''
    
    
    from pvlib.location import Location
    import numpy as np

    site = Location(latitude=lat, longitude=lon)#, 'Etc/GMT+1') # latitude, longitude, time_zone, altitude, name
    zenith = 180-float(site.get_solarposition(time)['zenith']) # correct? 

    return zenith
