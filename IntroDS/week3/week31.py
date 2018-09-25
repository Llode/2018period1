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