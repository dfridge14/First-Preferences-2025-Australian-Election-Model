import pandas as pd
import geopandas as gpd
import geopy.distance
import numpy as np
import os
from pathlib import Path


base_dir = Path('C:\\Dania\\2024\\Australian Election') if os.name == "nt" else Path.home() / "Australian Election"
os.chdir(base_dir)


census_year = '2011'

# Voting data required 2016->2021 Correspondence/Concordance : shapefile_path = 'SA1_2021_AUST_GDA2020.shp'
if census_year == '2021':
    shapefile_path = f'SA1_{census_year}_AUST_GDA2020.shp'
elif census_year == '2016':
    shapefile_path = f'SA1_{census_year}_AUST_GDA2020.shp'
elif census_year == '2011':
    shapefile_path = f'SA1_{census_year}_AUST.shp'

gdf_full = gpd.read_file(shapefile_path)

# 
# print(gdf_full.columns)

gdf = gdf_full[gdf_full.is_valid]
gdf['centroid'] = gdf.geometry.centroid
gdf['Lat'] = gdf.centroid.y
gdf['Long'] = gdf.centroid.x

# These are good enough! Maximum deviation 1.8 meters!!!!!

def compare_geometry_to_Australian_Albers():
    # Compute centroids in WGS84 (geographic)

    gdf['Long_WGS84'] = gdf.centroid.x
    gdf['Lat_WGS84'] = gdf.centroid.y

    gdf = gdf.to_crs(epsg=3577)  # MGA55, adjust for your region
    gdf['Long_Projected'] = gdf.centroid.x
    gdf['Lat_Projected'] = gdf.centroid.y

    # Convert back to WGS84
    gdf = gdf.to_crs(epsg=4326)
    gdf['Long_Corrected'] = gdf.centroid.x
    gdf['Lat_Corrected'] = gdf.centroid.y

    # Compare shifts
    gdf['Centroid_Shift_m'] = gdf.apply(
        lambda row: geopy.distance.distance(
            (row['Lat_WGS84'], row['Long_WGS84']),
            (row['Lat_Corrected'], row['Long_Corrected'])
        ).meters,
        axis=1
    )

    print(gdf[['Centroid_Shift_m']].describe())

    return 1

import pdb;pdb.set_trace()

if census_year == '2011':
    SA1_centroids_7dig = gdf.loc[:,[f'SA1_7DIG{census_year[-2:]}','Lat','Long']]
else:
    SA1_centroids_11dig = gdf.loc[:,[f'SA1_CODE{census_year[-2:]}','Lat','Long']] 

    SA1_centroids_7dig = SA1_centroids_11dig
    SA1_centroids_7dig.loc[:,f'SA1_CODE{census_year[-2:]}'] = SA1_centroids_7dig.loc[:,f'SA1_CODE{census_year[-2:]}'].astype(str).str[:1] + SA1_centroids_7dig.loc[:,f'SA1_CODE{census_year[-2:]}'].astype(str).str[5:]
    SA1_centroids_7dig.loc[:,f'SA1_CODE{census_year[-2:]}'] = SA1_centroids_7dig.loc[:,f'SA1_CODE{census_year[-2:]}'].astype(int)
    #print(SA1_centroids_7dig.loc[SA1_centroids_7dig["SA1_CODE21"] == 2117417,])

import pdb;pdb.set_trace()


# writes to csv file
SA1_centroids_7dig.to_csv(f'SA1_centroids_{census_year}.csv', index=False)




















# create df with 7-digit SA1s and their lat and long values
#SA1_centroids_7dig = pd.concat([gdf['SA1_7DIG16'],gdf['Lat'],gdf['Long']], axis=1)
#SA1_centroids_7dig['SA1_7DIG16'] = SA1_centroids_7dig['SA1_7DIG16'].astype(int)
#print(SA1_centroids_7dig.loc[SA1_centroids_7dig["SA1_7DIG16"] == 2117417,])

def get_SA1_centroids_2016():

    # Currently using 2016 version. Voting data required 2016->2021 Correspondence/Concordance : shapefile_path = 'SA1_2021_AUST_GDA2020.shp'
    shapefile_path = 'SA1_2016_AUST.shp'
    gdf_full = gpd.read_file(shapefile_path)
    
    print(gdf_full.columns)

    gdf = gdf_full.drop(gdf_full.index[-1])
    gdf['centroid'] = gdf.geometry.centroid
    gdf['Lat'] = gdf.centroid.y
    gdf['Long'] = gdf.centroid.x

    print(gdf)
    
    SA1_centroids_7dig = pd.concat([gdf['SA1_7DIG16'],gdf['Lat'],gdf['Long']], axis=1)
    SA1_centroids_7dig['SA1_7DIG16'] = SA1_centroids_7dig['SA1_7DIG16'].astype(int)

    return SA1_centroids_7dig

#SA1_centroids_2016 = get_SA1_centroids_2016()
#SA1_centroids_2016.to_csv('SA1_centroids_2016.csv', index=False)
#print(SA1_centroids_2016)
