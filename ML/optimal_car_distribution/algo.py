import pandas 
from sklearn.cluster import KMeans,DBSCAN

df_zones = pandas.read_json('whole_MOSCOW.json')

df_zones['all_day'] = df_zones.iloc[:,:-2].mean(axis=1)
df_zones['all_day'] = (df_zones['all_day']-df_zones['all_day'].mean())/(df_zones['all_day'].std())

X = df_zones.loc[:,['lat_c','lon_c','all_day']].values

kmeans = KMeans(n_clusters=20, random_state=0).fit(X)
a = kmeans.cluster_centers_

df_car = pandas.DataFrame({'lat_c':a[:,0], 'lon_c': a[:,1]})
df_car.to_json('car.json',orient="records")
