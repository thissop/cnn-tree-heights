def get_heights(annotations_gdf, cutlines_shp_file:str=None, cutline_info:dict=None, d:float=7, exact_mode:bool=True):
    r'''
    
    either cutlines_shp_file needs to be a defined file path, or cutline info needs to be a dictionary that would have been collected from the cutlines shp file.
    cutline_info needs to have the following keys if it is used over the cutlines file: ['SUN_ELEV', 'SUN_AZ']

    NOTES
    -----

    now has slower but exact way of constructing each dividing line such that each line is always exact length and will never be too short (it uses this approach if exact_mode=True)

    '''
    
    from cnnheights.utilities import height_from_shadow
    from cnnheights.preprocessing import get_cutline_data
    import numpy as np
    from shapely.geometry import LineString, box, Point
    import geopandas as gpd
    from cnnheights.utilities import height_from_shadow
    import pandas as pd
    import math

    if cutlines_shp_file is None and cutline_info is None: 
        raise Exception('')

    cutline_info = get_cutline_data(predictions=annotations_gdf, cutlines_shp=cutlines_shp_file)
    sun_az = cutline_info['SUN_AZ']
    zenith_angle = cutline_info['SUN_ELEV']
    if exact_mode: 
        lines = []
        for polygon in annotations_gdf['geometry']:
            centroid = polygon.centroid
            line_length = max(polygon.bounds[2] - polygon.bounds[0], polygon.bounds[3] - polygon.bounds[1]) * 2
            angle = math.radians(sun_az)
            endpoints = [(centroid.x - math.cos(angle) * line_length / 2, centroid.y - math.sin(angle) * line_length / 2),
                        (centroid.x + math.cos(angle) * line_length / 2, centroid.y + math.sin(angle) * line_length / 2)]
            line = LineString(endpoints)
            closest_intersection_point = min([line.intersection(LineString([polygon.exterior.coords[i], polygon.exterior.coords[i+1]]))
                                            for i in range(len(polygon.exterior.coords)-1) if line.intersects(LineString([polygon.exterior.coords[i], polygon.exterior.coords[i+1]]))],
                                            key=centroid.distance)
            line = LineString([closest_intersection_point, Point(2*centroid.x-closest_intersection_point.x, 2*centroid.y-closest_intersection_point.y)])
            lines.append(line) # @THADDAEUS: MAKE SURE IN FUTURE THAT IT DOESNT'T NOT APPEND ANYTHING!

        annotation_lines = gpd.GeoDataFrame({'geometry':lines}, crs=f'EPSG:{annotations_gdf.crs.to_epsg()}')

    else: 
        dy = np.abs(d/np.tan(np.radians(sun_az)))
        annotation_centroids = annotations_gdf.centroid
        annotation_lines = [LineString([(x-d, y+dy), (x+d, y-dy)]) for x, y in zip(annotation_centroids.x, annotation_centroids.y)]
    
    annotation_lines_gdf = gpd.GeoDataFrame({'geometry':annotation_lines}, crs=f'EPSG:{annotations_gdf.crs.to_epsg()}')

    #annotations_gdf['geometry'] = [make_valid(i) for i in annotations_gdf['geometry']]
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
