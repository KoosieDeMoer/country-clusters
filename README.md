# country-clusters

A clustering study to establish whether meaningful clusters could be extracted for the following data.

* EG.ELC.ACCS.ZS - Access to electricity (% of population)
* IC.FRM.BRIB.ZS - Bribery incidence (% of firms experiencing at least one bribe payment request)
* IT.NET.USER.ZS - Individuals using the Internet (% of population)

The follwing pipeline was used:
1. SimpleImputer(missing_values=np.nan, strategy='mean')
1. StandardScaler
1. DBSCAN
1. PCA(n_components=2)

There really are no clusters. But there are some interesting results (possibly artefacts)

![](all_countries_unlabled.png?raw=true "All Countries - unlabled")
![](the_artefact.png?raw=true "The Artefact?")
![](all_countries_labled.png?raw=true "All Countries - labled")
