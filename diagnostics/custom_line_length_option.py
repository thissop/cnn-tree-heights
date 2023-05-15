from shapely.geometry import Polygon, LineString, Point
import math
import geopandas as gpd
import matplotlib.pyplot as plt

lines = [] 
gdf = gpd.read_file('temp_annotations_gdf_debug_invalid.gpkg')

sun_angle = 299 

for polygon in gdf['geometry']:
    centroid = polygon.centroid
    line_length = max(polygon.bounds[2] - polygon.bounds[0], polygon.bounds[3] - polygon.bounds[1]) * 2
    angle = math.radians(sun_angle)
    endpoints = [(centroid.x - math.cos(angle) * line_length / 2, centroid.y - math.sin(angle) * line_length / 2),
                (centroid.x + math.cos(angle) * line_length / 2, centroid.y + math.sin(angle) * line_length / 2)]
    line = LineString(endpoints)
    closest_intersection_point = min([line.intersection(LineString([polygon.exterior.coords[i], polygon.exterior.coords[i+1]]))
                                    for i in range(len(polygon.exterior.coords)-1) if line.intersects(LineString([polygon.exterior.coords[i], polygon.exterior.coords[i+1]]))],
                                    key=centroid.distance)
    line = LineString([closest_intersection_point, Point(2*centroid.x-closest_intersection_point.x, 2*centroid.y-closest_intersection_point.y)])

    lines.append(line)

lines_gdf = gpd.GeoDataFrame({'geometry':lines}, gdf.crs)

fig, ax = plt.subplots()

gdf.plot(ax=ax)
lines_gdf.plot(ax=ax, lw=0.25, color='black')

plt.savefig('custom_lines.pdf')

