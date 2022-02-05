import numpy as np
import pandas as pd
from math import cos, asin, sqrt, pi # calc_distance

# transforms age in form xx-xx to integer
def range_to_num(age):
    
    max_range = 10
    a_list = age.split('-')

    if (a_list[0] == ''):
        return round(float(a_list[1]))
    elif ((len(a_list)) == 2 and (a_list[1] == '')):
        return round(float(a_list[0]))

    map_ints = map(float, a_list)
    map_list = list(map_ints)

    if (len(map_list) == 1):
        return round(map_list[0])  

    if ((map_list[1] - map_list[0]) > max_range):
        return 'remove'
    else:
        return round((map_list[1] + map_list[0]) / 2)


# fixes inconsistent country labels
def fix_country_label(country):
    if country == 'Korea, South':
        return 'South Korea'
    if country == 'US':
        return 'United States'
    return country


# calculates distance in kilometres between two points given by latitude and longitude
# function slightly adjusted from:
# https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula/21623206
def calc_distance(lat1, lon1, lat2, lon2):
    earth_radius = 6371 # kilometres    
    p = pi/180
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1-cos((lon2-lon1)*p))/2
    return 2 * earth_radius * asin(sqrt(a)) #return in km

# Replace missing countries in the cases data with the closest country in the locations data.
def get_country(cases_location, lat, lon):
    countries = cases_location[['Country_Region', 'Lat', 'Long_']]
    countries = countries.groupby(['Country_Region']).mean().dropna().reset_index()
    countries['dist'] = countries.apply(lambda x: calc_distance(x.Lat, x.Long_, lat, lon), axis=1)
    return countries.loc[countries['dist'].idxmin(axis=0)]['Country_Region']


# Replace provinces that are missing or don't match anything in the locations dataset with the closest province from the locations dataset
def get_province(cases_location, country, lat, lon):
    provinces = cases_location[['Country_Region', 'Province_State', 'Lat', 'Long_']]
    provinces = provinces.groupby(['Country_Region', 'Province_State']).mean().dropna().reset_index()
    provinces = provinces[provinces['Country_Region']==country]
    
    if provinces.empty:
        return np.nan
    
    provinces['dist'] = provinces.apply(lambda x: calc_distance(x.Lat, x.Long_, lat, lon), axis=1)
    return provinces.loc[provinces['dist'].idxmin(axis=0)]['Province_State']
