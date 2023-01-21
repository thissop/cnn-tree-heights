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

    # meta data in mosaic file is half-true, half-ommission (accounts for only one of the cut lines)
    # the geometry file will tell you meta data per satilite image that was incorporated into the satilte tile
    # hence, I should probably 
    # technically can be arbitrary number of satilite images that overlap 
    # zip archive of cutline data, so I can incorporate this into my pipeline for getting lat/longs
    # will be interesting algo optimization 
    # good for parallelizing 
    # slow part could be assosc between cutline and satilite geometry (two trees next to eachother, but diff sun satilite geos)

    # past approach: used a regression to map poly to a geo that represesented the shadow ... 
    # polynomial was really brittle...benefit of using CNN
    # construct good training set....geo is more or less solved?
    # shadow length: take length of shadow in length of sun. 

    # weekend: finish transitioning old library 
        # he'll send me geo file, I'll work with it (he calls it a cut line file)
        # he'll send resources about previous approach (doesn't seem like good use of time to look through rough code)