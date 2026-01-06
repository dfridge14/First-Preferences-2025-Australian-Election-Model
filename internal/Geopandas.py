import geopandas as gpd
import os

os.chdir('C:\\Dania\\2024\\Australian Election')

# Load the shapefile
shapefile_path = 'SA1_2021_AUST_GDA2020.shp'
gdf = gpd.read_file(shapefile_path)

print(gdf)

gdf['centroid'] = gdf.geometry.centroid

# Extract latitude and longitude
gdf['latitude'] = gdf.centroid.y
gdf['longitude'] = gdf.centroid.x

print(gdf.centroid.y)
print(gdf.centroid.x)
print(gdf.columns)


# Select relevant columns
#sa1_lat_lon = gdf[['SA1_CODE_2020', 'latitude', 'longitude']]

# Display the first few rows of the DataFrame
#print(sa1_lat_lon.head())