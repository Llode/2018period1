#%%
import numpy as np
import pandas as pd
import geopandas as gpd
import os
import imageio

print(os.getcwd())
filepath = os.getcwd() + "\\IntroDS\\week3\\"

world = gpd.read_file(filepath+'world\\world_m.shp')
cities = gpd.read_file(filepath+'cities\\cities.shp')
cities = cities.to_crs(world.crs)
base = world.plot(color='white', edgecolor='black')
cities.plot(ax=base, marker='o', color='red', markersize=5)

hasy_full = pd.read_csv(filepath+'HASYv2\\hasy-data-labels.csv')
hasy = hasy_full.query('70<=symbol_id<=80')

hasy.head(10)

pics = np.array([])

# for pic in hasy.symbol_id:
#    np.append(pics, imageio.imread(filepath+'HASYv2\\hasy-data\\' +pic))

# print(len(pics))