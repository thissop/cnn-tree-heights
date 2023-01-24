def get_height(shadow:float, time:str, lat:float, lon:float):
    from pvlib.location import Location
    import numpy as np

    site = Location(latitude=lat, longitude=lon)#, 'Etc/GMT+1') # latitude, longitude, time_zone, altitude, name
    zenith = 180-float(site.get_solarposition(time)['zenith']) # correct? 

    height = shadow/np.tan(np.radians(zenith))
    print(zenith)
    print(np.tan(np.radians(zenith)))
    print(height)

get_height(3, '2018-01-01 00:00:00', lat=51.5, lon=0.34)