import random
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from tqdm.notebook import tqdm
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering

from typing import Union
from fastapi import FastAPI

app = FastAPI()

# a finction to simulate clicks using random
def create_data():
	n = 500
	x = [float('{:.2f}'.format(random.random())) for _ in range(n)]
	y = [float('{:.2f}'.format(random.random())) for _ in range(n)]
	df = pd.DataFrame({'x':x,'y':y})
	return df

# storing the data to the database
def store_Data(dataframe,databaseName,collectionName,predicted):
    from pymongo import MongoClient
    client = MongoClient('localhost', 27017)  # creating connection to database
    db = client[databaseName]   # connecting to database 
    
    if not predicted:
	    # deleting previous stored records
	    db[collectionName].delete_many({})
    
    # inserting data into below table of above collection (database name)
    db[collectionName].insert_many(dataframe.apply(lambda x: x.to_dict(), axis=1).to_list())

# retrieving the data from the database
def retrieve_database(databaseName,collectionName,x_coord,y_coord):   
    from pymongo import MongoClient
    client = MongoClient('localhost', 27017)  # creating connection to database
    db = client[databaseName]   # connecting to collection (it doesn't matter it exits or not)
    if x_coord and y_coord:
        df = db[collectionName].find({'x':x_coord,'y':y_coord})
    else:
        df = db[collectionName].find()
    retrieved_df = pd.DataFrame(df)
    return(retrieved_df)

# a function to check if the user given coordinates are already in the database
def check_if_coord_exists(df,x_coord,y_coord):
	status = False
	x_df = df[df['x']==x_coord]
	if len(x_df):
	    y_df = x_df[x_df['y']==y_coord]
	    if len(y_df):
	        status = True
	return status


# at homepage, the data will be generated, clustered and saved in the database
@app.get("/")
def root():
	df = create_data()

	# storing the generated coordinates to the database
	store_Data(dataframe=df,databaseName='userClick',collectionName='coordinates',predicted=False)

	# fitting the data to DBSCAN
	clf = DBSCAN(eps=0.06, min_samples=5).fit(df)
	dbscan_df = pd.DataFrame({'x':df['x'],'y':df['y'],'labels':clf.labels_})

	# storing the labelled data to the database
	store_Data(dataframe=dbscan_df,databaseName='userClick',collectionName='dbscan_cluster',predicted=False)


# a) predict the cluster of the input coordinates
@app.get("/save_prediction")
def save_prediction(x_coord,y_coord):
	dbscan_df = retrieve_database('userClick','dbscan_cluster',None,None)
	status = check_if_coord_exists(dbscan_df,x_coord,y_coord)
	if status:
		pass
	else:
		neigh = KNeighborsClassifier(n_neighbors=5).fit(dbscan_df.iloc[:,1:-1].values, dbscan_df['labels'].values)
		cluster_id = neigh.predict(np.array([x_coord,y_coord]).reshape(1, -1))
		result_df = pd.DataFrame({ "x": [x_coord], "y": [y_coord], "labels": [cluster_id[0]]})

		# calling the function to save the result of the label prediction
		store_Data(dataframe=result_df,databaseName='userClick',collectionName='dbscan_cluster',predicted=True)

# b) Accepts input coordinates and returns the cluster-id it belongs to
@app.get("/find_clusterID")
def find_cluster(x_coord,y_coord):
	save_prediction(x_coord,y_coord)
	result = retrieve_database('userClick','dbscan_cluster',x_coord,y_coord)
	cluster_id = int(result['labels'][0])
	return cluster_id

# c) Returns all the clusters and coordinates
@app.get("/find_data_clusters")
def find_data_clusters():
	all_data = retrieve_database('userClick','dbscan_cluster',None,None).iloc[:,1:]
	return all_data

	



