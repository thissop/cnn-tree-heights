def height_from_shadow(shadow:float, time:str, lat:float, lon:float): 
    r'''
    _Calculate height of tree from shadow length, time, and location_  

    Parameters
    ----------      

    shadow : `float`
        Shadow length in meters

    time : `str`
        Time in `"yyyy-mm-dd hh:mm:ss"` format

    lat : `float`

    lon : `float`

    Returns
    -------

    height : `float`
        The tree's height in meters. 
    '''
    
    
    from pvlib.location import Location
    import numpy as np

    site = Location(latitude=lat, longitude=lon)#, 'Etc/GMT+1') # latitude, longitude, time_zone, altitude, name
    zenith = 180-float(site.get_solarposition(time)['zenith']) # correct? 

    height = shadow/np.tan(np.radians(zenith)) # does this need to get corrected for time zone? 

    return height