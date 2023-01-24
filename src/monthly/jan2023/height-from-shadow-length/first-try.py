import astropy.coordinates as coord
from astropy.time import Time
import astropy.units as u

import time as t
from pvlib.location import Location

loc = coord.EarthLocation(lon=0.1 * u.deg,
                          lat=51.5 * u.deg)
now = Time.now()
time = '2018-01-01 00:00:00'

s1 = t.time()

altaz = coord.AltAz(location=loc, obstime=time)
sun = coord.get_sun(now)

astropy_zenith = sun.transform_to(altaz).zen.value
d1 = t.time()-s1

print(astropy_zenith)
### alternative method
s2 = t.time()
site = Location(51.5, 0.1)#, 'Etc/GMT+1') # latitude, longitude, time_zone, altitude, name
zenith = float(site.get_solarposition(time)['zenith'])
d2 = t.time()-s2 
print(zenith)
# there are four time zones in africa!! 

print(d1/d2, d2/d1)
print(d1, d2) # astropy method takes 0.3 seconds, other takes ~0.02 seconds. other method is ~15.6x faster. 