# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Mapping municipality and effective complaints

# +
# Import Scripts
# %reload_ext autoreload
# %autoreload 2
import sys
sys.path.insert(0, '../src/utils')
from IPython.display import display, HTML
import utils 
import secret

# Packages
import pandas as pd
import geopandas
import geoplot
import matplotlib.pyplot as plt
from shapely.geometry import Point
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
pd.options.display.max_columns = 999
# -

con = utils.connect_to_database()

tab_path = 'data_output'
user_name = 'wen'

# ## Tenders

#Get municipality level data
query = """select convocante,sum(bool_of_effective_complaints) effective_complaints, sum(bool_of_amendments) amendments
from raw_labeled.proceso
where tipo_entidad = 'Municipalidades'
group by convocante;
"""

tenders = pd.read_sql_query(query, con)

#Clean strings
tenders['convocante'] = tenders['convocante'].str.replace(r'^Municipalidad de ','')
tenders['convocante'] = tenders['convocante'] + ', Paraguay'
tenders.head()

# ## Geocoding addresses to longitude and latitude

cities = tenders.convocante.to_list()


#Define a function that geocodes address parsed
def do_geocode(address):
    geolocator = Nominatim(user_agent=secret.user, timeout=5)
    try:
        return geolocator.geocode(address)
    except GeocoderTimedOut:
        return do_geocode(address)
        logging.info('do_geocode time out')


municipality_code = dict()
for i in cities:
    info = do_geocode(i)
    if info is None:
        print(f'This address is not parsed:{i}')
        pass
    else:
        municipality_code.update({i:{'longitude': info.longitude,
                                 'latitude': info.latitude}})

#Number of municipalities that are uncoded and their impact
print(len(cities) - len(municipality_code))
uncoded = set(cities) - municipality_code.keys()
uncoded_tenders = tenders[tenders['convocante'].isin(uncoded)]
uncoded_tenders

longlat = pd.DataFrame.from_dict(municipality_code, orient='index')
longlat['municipalities'] = longlat.index
longlat.reset_index(inplace=True, drop=True)
longlat.head()

#Merging tender info with longtitude and latitude information
muni_longlat = pd.merge(tenders, longlat, how='inner', left_on='convocante', right_on='municipalities')
muni_longlat.head()

# ### Converting longitude and latitude into geometry points for spatial joints

geometry = [Point(xy) for xy in zip(longlat['longitude'], longlat['latitude'])]

crs = {'init': 'epsg:4326'}
muni_geo = geopandas.GeoDataFrame(muni_longlat, crs=crs, geometry=geometry)

# ### Plotting

#Basemap sourced from url: diva-gis.org/gdata
basemap = geopandas.read_file("/data/shared_data/data/raw/map2/")

#Check that the basemap projection is the same 
basemap.crs

#Plot to see if there's any outliers in the geocoding
ax = basemap.plot(color='white',edgecolor='black', figsize=(20,5))
muni_geo.plot(ax=ax, color='red')
plt.show()

#Use spatial join to filter out only correctly parsed addresses that  are within the country boundaries
muni_with_region =geopandas.sjoin(basemap, muni_geo, how='left', op='contains')
muni_with_region

geoplot.choropleth(muni_with_region, hue='effective_complaints', cmap='Blues', figsize=(10,8), legend=True)


