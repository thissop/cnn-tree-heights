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

def shadows_from_annotations(annotations_file, cutlines_shp:str, north:float, east:float, epsg:str, save_path:str=None, d:float=3):
    r'''
    _get shadow lengths and heights from annotations gpkg file, coordinate, and cutfile_

    Arguments 
    ---------

    annotations_file : `str`
        path to annotations file...ideally geoparquet

    NOTES 
    -----

    - get lat/long from center ish of the 

    - uses lat/long to get container in cutlines file to reference to get azimuth angle from. 
    - work with feather files (my preference) for saving these (e.g. save_path='./data.feather')...I love flexibility of .to_file(...)

    TO DO 
    -----
    - fix lat long stuff?
    - d is in meters because it's in the UTM projection? something to mention in paper how we update it for different regions (because inaccurate outside itself)

    ''' 

    import geopandas as gpd 
    import numpy as np
    from shapely.geometry import LineString, box
    from cnnheights.utilities import longest_side
    from cnnheights.preprocessing import get_cutline_data
    import os 

    
    if 'parquet' in annotations_file: 
        annotations_gdf = gpd.read_parquet(annotations_file)

    else: 
        annotations_gdf = gpd.read_file(annotations_file)

    annotations_gdf['geometry'] = annotations_gdf.buffer(0)
    annotations_gdf = annotations_gdf[annotations_gdf.geom_type == 'Polygon']
    annotations_gdf = annotations_gdf.set_crs(f'EPSG:{epsg}', allow_override=True)
   
    centroids = annotations_gdf.centroid    

    cutline_info = get_cutline_data(north=north, east=east, epsg=epsg, cutlines_shp=cutlines_shp)

    dy = np.abs(d/np.tan(np.radians(cutline_info['SUN_AZ'])))
    lines = [LineString([(x-d, y+dy), (x+d, y-dy)]) for x, y in zip(centroids.x, centroids.y)]
    lines_gdf = gpd.GeoDataFrame({'geometry':lines}, geometry='geometry', crs=annotations_gdf.crs)

    shadow_lines = lines_gdf.intersection(annotations_gdf, align=False)
    shadow_lengths = shadow_lines.length

    heights = height_from_shadow(shadow_lengths, zenith_angle=cutline_info['SUN_ELEV'])
    #print(cutline_info['SUN_ELEV'])

    bounds = annotations_gdf.bounds
    dx = np.abs(np.abs(bounds['maxx'])-np.abs(bounds['minx']))
    dy = np.abs(np.abs(bounds['maxy'])-np.abs(bounds['miny']))
    dxy = np.max(np.array([dx,dy]).T, axis=1)
    square_bounds = np.array([[minx, miny, minx+diff, miny+diff] for minx, miny, diff in zip(bounds['minx'], bounds['miny'], dxy)])    

    d = {'geometry':annotations_gdf['geometry'], 
         'centroids':centroids,
         'bounds_geometry':gpd.GeoSeries([box(*i) for i in square_bounds]),
         'heights':heights, 
         'line_geometries':shadow_lines, 
         'lengths':shadow_lengths, 
         'areas':annotations_gdf['geometry'].area, 
         'perimeters':annotations_gdf['geometry'].length,
         'diameters':gpd.GeoSeries(longest_side(annotations_gdf['geometry'])),
        }
    
    shadows_gdf = gpd.GeoDataFrame(d, crs=f'EPSG:{epsg}', index=list(range(len(shadow_lengths))))
    shadows_gdf = gpd.GeoDataFrame(shadows_gdf[shadows_gdf['geometry'] != None])

    #print(shadows_gdf)

    if save_path is not None: 
        save_path_list = save_path.split('/')
        save_dir = '/'.join(save_path_list[:-1])
        file_name = save_path_list[-1].split('.')[0]+'.geoparquet'
        shadows_gdf.to_parquet(os.path.join(save_dir, file_name))

    return shadows_gdf

def longest_side(polygons:list):
    '''
    Notes
    -----
        - This code is so inefficient...but will work for now. it calculates longest side length \
        - polygons needs to be list of polygon objects 

    '''
    import numpy as np

    longest_lengths = []
    for polygon in polygons: 
        coords = list([list(i) for i in polygon.exterior.coords])
        coords = coords+[coords[0]]
        lengths = []
        for i in range(len(coords)-1): 
            left = coords[i]
            right = coords[i+1]
            dist = np.sqrt((left[0]-right[0])**2+(left[1]-right[1])**2)
            lengths.append(dist)
        
        longest_length = np.max(lengths)
        longest_lengths.append(longest_length)

    return longest_lengths

def invert_mask(X): 
    
    r'''
    Notes
    -----

    X' = 1/X
    X' = X' renormalized to original min max of X
    returns X'
    
    '''

    import numpy as np
    from sklearn.preprocessing import normalize

    y = 1/X
    return (y - np.min(y)) / (np.max(y) - np.min(y))
