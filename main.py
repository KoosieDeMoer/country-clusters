import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.preprocessing as preprocessing
import sklearn.cluster as cluster
import sklearn.decomposition as decomposition
import sklearn.impute  as impute 


# to avoid utf-8 codec error open and save the file with notepad as utf-8 
data = pd.read_csv("significant_data.csv", encoding='cp1252') 

data = data.drop(0, axis=0)

target = data['Continent Code']

labels = data['Country Name']

data = data.drop("Country Code", axis=1).drop("Country Name", axis=1).drop("Continent Code", axis=1).drop("BN.CAB.XOKA.GD.ZS", axis=1)

print(data)

imp_mean = impute.SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit(data)

data = imp_mean.transform(data)

scaler = preprocessing.StandardScaler()

scaler.fit(data)

data_scaled = scaler.transform(data)


# turn
dbscan = cluster.DBSCAN()
cluster_labels = dbscan.fit_predict(data_scaled)

pca = decomposition.PCA(n_components=2)

data_scaled_pca = pca.fit_transform(data_scaled)

print(pca.components_)

# Preview the first 5 lines of the loaded data 
#print(data_scaled)

print(target)


le1 = preprocessing.LabelEncoder()
continent_codes = ["AF", "AN", "AS", "EU", "NO", "OC", "SA", "ZZ"]
le1.fit(continent_codes)
target_numbers = le1.fit_transform(target)





# plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=target_numbers, linewidths=0, s=30)
plt.scatter(data_scaled_pca[:, 0], data_scaled_pca[:, 1], c=cluster_labels, linewidths=0, s=30)

# plt.axis([-2.5, 0, -1, 1])

#for i, txt in enumerate(labels):
#    plt.annotate(txt, (data_scaled_pca[i, 0], data_scaled_pca[i, 1]))
plt.show()

