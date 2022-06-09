# -*- coding: utf-8 -*-
"""
Created on Thu May  5 13:15:13 2022

@author: Sai pranay
"""
#-------------------------importing_the_data_set-------------------------------

import pandas as pd

wine = pd.read_csv("E:\DATA_SCIENCE_ASS\PCA\\wine.csv")
wine
wine.shape
list(wine)
wine.corr()        #---checking_correlation
wine.describe()
wine.info()


x1 = pd.DataFrame(wine)    #-----converting into dataframe---------------------

x1.shape
x1.head()

#--------------------------droping---------------------------------------------
x2 = wine.drop(['Type'],axis = 1)
x2

#-------------------------------standardizing----------------------------------

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scale = scaler.fit_transform(x2)
X_scale

#------------------converting into dataframe-----------------------------------

X_scaler = pd.DataFrame(X_scale)
X_scaler



#------------ load decomposition to do PCA analysis with sklearn---------------

from sklearn.decomposition import PCA
PCA()
pca = PCA(svd_solver='full')

pc = pca.fit_transform(X_scaler)
pca.explained_variance_ratio_
sum(pca.explained_variance_ratio_)

pc.shape
pd.DataFrame(pc).head()
type(pc)



pc_win = pd.DataFrame(data = pc , columns = ['pc1','pc2','pc3','pc4','pc5','pc6','pc7','pc8','pc9','pc10','pc11','pc12','pc13'])
pc_win.head()
pc_win.shape
type(pc_win)

pc_win.to_csv("G:\\'PC13'.csv")



import seaborn as sns
win_ = pd.DataFrame({'var':pca.explained_variance_ratio_,'PC':['pc1','pc2','pc3','pc4','pc5','pc6','pc7','pc8','pc9','pc10','pc11','pc12','pc13']})
sns.barplot(x='PC',y="var", data=win_, color="c");


pca = PCA(n_components=3)
pc = pca.fit_transform(X_scale)
pc.shape

pc_winer = pd.DataFrame(data = pc , columns = ['PC1', 'PC2','PC3'])
pc_winer.head()

pca.explained_variance_ratio_



wi = pd.DataFrame({'var':pca.explained_variance_ratio_,'PC':['PC1','PC2','PC3']})
sns.barplot(x='PC',y="var", data=wi, color="c");

#------------------------------------------------------------------------------

print(pc_winer)


X = pc_winer.values 
X.shape
print(X)

x_sai = pd.DataFrame(pc_winer)
x_sai

%matplotlib qt
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x_sai.iloc[:, 0], x_sai.iloc[:, 1], x_sai.iloc[:, 2])
plt.show()

#===================== Initializing KMeans=====================================

from sklearn.cluster import KMeans
KMeans()
kmeans = KMeans(n_clusters=3,random_state=4)

# Fitting with inputs

kmeans = kmeans.fit(x_sai)

# Predicting the clusters

labels = kmeans.predict(x_sai)
type(labels)

# Getting the cluster centers

C = kmeans.cluster_centers_
C

# Total with in centroid sum of squares 

kmeans.inertia_


%matplotlib qt
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x_sai.iloc[:, 0], x_sai.iloc[:, 1], x_sai.iloc[:, 2])
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c='Red', s=1000) # S is star size, c= * color


Y = pd.DataFrame(labels)

df_new = pd.concat([pd.DataFrame(x_sai),Y],axis=1)
df_new

pd.crosstab(Y[0],Y[0])

Y
Y.value_counts()

y = wine.iloc[:,:1]
type(y)
y.shape
y.head()
y.value_counts()

Y_1=[]
for i in range(0,178,1):
    if y.iloc[i,0]==1:
        Y_1.append(0)
    elif y.iloc[i,0]==2:
        Y_1.append(1)
    elif y.iloc[i,0]==3:
        Y_1.append(2)
Y_1



from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(Y_1,Y)
acc = accuracy_score(Y_1,Y).round(2)

print(" accuracy score:" , acc)

#------------------------------------------------------------------------------
#-------------------------------HIERARCHICALCLUSTERING-------------------------

xc=x_sai.iloc[:,:3]
xc

#-------------------------------PLOTTING---------------------------------------
import scipy.cluster.hierarchy as shc

# construction of Dendogram
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 7))  
plt.title(" Dendograms")  
dend = shc.dendrogram(shc.linkage(xc, method='complete')) 

"""
Now we know the number of clusters for our dataset, 
the next step is to group the data points into these five clusters. 
To do so we will again use the AgglomerativeClustering
"""
## Forming a group using clusters
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='complete')
O = cluster.fit_predict(xc)
O

plt.figure(figsize=(10, 7))  
plt.scatter(xc.iloc[:,0], xc.iloc[:,1],xc.iloc[:,2], c=cluster.labels_, cmap='rainbow')  

Y_clust = pd.DataFrame(O)
Y_clust[0].value_counts()
Y_clust.shape

y = wine.iloc[:,:1]
type(y)
y.shape
y.head()
y.value_counts()


from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(y,Y_clust)
acc = accuracy_score(y,Y_clust).round(2)

print(" accuracy score:" , acc)

