
############ ERWTHMA 1

# PREPROCESSING
import time as time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings

df  = pd.read_csv('/home/user/Downloads/ergasia_dm/bank-additional/bank-additional-full.csv',sep=";")

### BASIKA STOIXEIA

df.head(10)
df.shape
type(df)
df.columns

print(df.info())
print(df.describe())
basic_stats = pd.DataFrame(df.describe())
# ---- PROBLHMATA AKRAIWN TIMWN ---- #
# duration, campaign , pdays,
#----------------------------------- #


sns.distplot(df['pdays'])
plt.show()

nums = df.select_dtypes(include=[np.number])
tetm = [0,1,2,3,4]*2
teta  = np.repeat(np.arange(0,2),5)

# DENSITY-PLOT
fig, axs = plt.subplots(ncols=5,nrows=2)
for i in [0,1,2,3,4,5,6,7,8,9]:
  sns.distplot(nums.iloc[:,i],ax=axs[teta[i],tetm[i]])

# BOXPLOT
fig, axs = plt.subplots(ncols=5,nrows=2)
for i in [0,1,2,3,4,5,6,7,8,9]:
  sns.boxplot(nums.iloc[:,i],ax=axs[teta[i],tetm[i]])
# NaN

df.isnull().sum().sum() #oxi Nulls
df['default']

nums_new = nums.copy()
for column in nums_new.columns :
  nums_new.loc[:,column] = sorted(nums_new.loc[:,column])

print(nums_new.head(3))

# CORR-PLOT
f, ax = plt.subplots()
corr = df.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),annot=True,
          fmt='.1f',  square=True, ax=ax)

#  PERI TOY PDAYS
dfc = df.copy()
set(df['pdays'])
df.loc[df['pdays']<999,'pdays'].sum()
df_freq = pd.crosstab(index=df['pdays'],columns='count')

# DIAXWRISMOS NUM KAI CATEGORICAL DATA
df_num =df.select_dtypes(include=['number','float64'])
df_cat = df.select_dtypes(include=[object])



dfc = df.copy()
dfc_num =dfc.select_dtypes(include=['number','float64'])
dfc_cat = dfc.select_dtypes(include=[object])



# OUTLIERS basei krithriou : <200 kai >40988 apo #diatetagm. times
dfc_num.shape[0]*1/100
sorted(dfc_num['age'])[1030]

place = []
for col in dfc_num.columns :
      k = sorted(dfc_num[col])[200]
      a = sorted(dfc_num[col])[40988]
      matrix = list(np.where((dfc_num[col]<k)|(dfc_num[col]>a))[0])
      place.extend(matrix)
print(place[0:9],len(place)/dfc_num.shape[0])
print('Percentage of Values reduced = ',len(place)/dfc_num.shape[0])
dfc_cut = df_num.drop(labels=list(set(place)),axis=0)
dfc_cat = df_cat.drop(labels=list(set(place)),axis=0)
data_prim = dfc.drop(labels=list(set(place)),axis=0)
y = dfc_cat['y']
y.reset_index(drop=True,inplace=True)
dfc_cat = dfc_cat.drop(columns='y')
dfc_cat_tr = dfc_cat.copy()

data_num = dfc_cut.copy()
data_cat = dfc_cat.copy()

fig, axs = plt.subplots(ncols=2,nrows=2)
sns.distplot(data_num.pdays,ax=axs[0,0])
sns.distplot(data_num.previous,ax=axs[0,1])
sns.distplot(data_num.campaign,ax=axs[1,0])
sns.distplot(data_num['nr.employed'],ax=axs[1,1])
plt.show()


# KWDIKOPOIHSH PDAYS
pdays_coded = pd.DataFrame(np.where(data_num['pdays']<100,0,1))
data_cat.reset_index(drop=True,inplace=True)
data_num.reset_index(drop=True,inplace=True)
data_cat = pd.concat([data_cat,pdays_coded],axis=1)
data_cat.rename(columns={0:'pdays_coded'},inplace=True)
data_num = data_num.drop(columns='pdays',axis=1)



# KWDIKOPOIHSH PREVIOUS
dfc_cut['previous'].astype('category')
type(dfc_cut['previous'])
previous = dfc_cut['previous']
previous.reset_index(drop=True,inplace=True)
data_cat = pd.concat([data_cat,previous],axis=1)
data_cat.rename(columns={0:'previous_coded'},inplace=True)
data_num = data_num.drop(columns='previous',axis=1)
data_num.columns

# KWDIKOPOIHSH CAMPAIGN
plt.hist(dfc_cut['campaign'])
set(dfc_cut['campaign'].values)
dfc_cut['campaign'].describe()
campaign_coded = np.where(dfc_cut['campaign']<=3,0,1)
campaign_coded = pd.DataFrame(campaign_coded)
data_cat = pd.concat([data_cat,campaign_coded],axis=1)
data_cat.rename(columns={0:'campaign_coded'},inplace=True)

data_num = data_num.drop(columns='campaign',axis=1)
data_num.columns


# KWDIKOPOIHSH NR.EMPLOYED
nrempl_coded =  pd.cut(data_num['nr.employed'],bins=[0,5050,5150,6000],labels=[0,1,2])
data_cat = pd.concat([data_cat,nrempl_coded],axis=1)
data_cat.rename(columns={0:'nrempl_coded'},inplace=True)
data_num = data_num.drop(columns='nr.employed',axis=1)

# KANONIKOPOIHSH
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
cl = data_num.columns
data_num = std.fit_transform(data_num)
data_num = pd.DataFrame(data_num)
data_num.columns = cl
data_num.head()

# ELEGXOS DATA_FRAME
print(
data_cat.isnull().sum().sum(),
data_num.isnull().sum().sum()
)

###################################################

fig, axs = plt.subplots(nrows=2,ncols=3)
sns.histplot(data_num['age'],ax =axs[0,0])
sns.histplot(data_num['duration'],ax=axs[0,1])
sns.histplot(data_num['emp.var.rate'],ax=axs[0,2])
sns.histplot(data_num['cons.price.idx'],ax=axs[1,0])
sns.histplot(data_num['cons.conf.idx'],ax=axs[1,1])
sns.histplot(data_num['euribor3m'],ax=axs[1,2])
plt.show()

# LABEL_ENCODER
from sklearn.preprocessing import LabelEncoder
l_e = LabelEncoder()
data_cat1 = data_cat.copy()

for col in data_cat1.columns :
    data_cat1[col] = l_e.fit_transform(data_cat1[col])

data_cat1.reset_index(drop=True,inplace=True)


# PRINCIPAL COMP ANALYSIS
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

X = pd.DataFrame(data_num).to_numpy()

pca = decomposition.PCA(n_components='mle',svd_solver='full')
pca.fit(X)
X = pca.transform(X)

data_pca = X.copy()

################################################################
                   # FEATURE_SELECTION #
################################################################

# FEATURE SELECTION TREES
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


y = y.str.get_dummies().iloc[:,1]
y = np.array(y)
X = np.array(data_cat1)
classf = ExtraTreesClassifier()
classf = classf.fit(X,y)
importances = classf.feature_importances_
fs_modeltr = SelectFromModel(classf, prefit=True).transform(X)

print(fs_modeltr[0])
print(X[0])
print(df.head(1))
print(fs_modeltr.shape)
# job, education, month, day_of_week, nr.employed


# FEATURE SELECTION K-BEST
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)


fs_modelkb = SelectKBest().fit_transform(X,y)

print(fs_modelkb[0])
print(X[0])
print(df.head(1))

print(fs_modelkb.shape)

print('\n','Dimensions of Kbest-Feature Selection','\n',
'-------------------------------------',
'\n','Kbest-Trees = ',
fs_modeltr.shape,'\n','Kbest = ',fs_modelkb.shape,
'\n','Full_Data = ',data_pca.shape)

##########################################################
                    # CLUSTERING #
##########################################################
# DBSCAN

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

xclust = data_pca

tdb1 = time.time()
db = DBSCAN(eps=1.1, min_samples=5, algorithm='kd_tree', n_jobs=8).fit(xclust)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
tdb2 = time.time()

core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_


n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Time Cost of DBscan = ',tdb2-tdb1)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(xclust, labels))

# KMEANS
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

xclust = data_pca

tkm1 = time.time()
db = KMeans(n_clusters=3, max_iter=1000).fit(xclust)
tkm2 = time.time()

centroids = db.cluster_centers_
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Time Cost of Kmeans = ',tkm2-tkm1)
print('Number of iterations: %d' % db.n_iter_)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(xclust, labels))
print("Mean Squared Error: %0.3f" % db.inertia_)

#################### noise #######################
import random
noise = np.reshape(np.random.normal(0,1,xclust.shape[0]*xclust.shape[1]),[xclust.shape[0],xclust.shape[1]])

tkm_noise1 = time.time()
db = KMeans(n_clusters=3, max_iter=1000).fit(noise+xclust)
tkm_noise2 = time.time()

centroids = db.cluster_centers_
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Time Cost of Kmeans = ',tkm_noise2-tkm_noise1)
print('Number of iterations: %d' % db.n_iter_)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(noise+xclust, labels))

###
#noisedf = pd.DataFrame(noise+xclust)
#pair_noise = sns.pairplot(noisedf,vars=[0,1,2,3,4])
#pair_noise.fig.set_size_inches(5,5)
#plt.show()
###
lab = np.reshape(db.labels_,[39687,1])
np.concatenate([xclust+noise,lab],axis=1) 
dat =  np.concatenate([xclust+noise,lab],axis=1) 
dat = pd.DataFrame(dat)   
dat.columns = ['0','1','2','3','4','5']
sns.set(style='ticks',color_codes=True)
pair = sns.pairplot(dat,vars=['0','1','2','3','4'],hue='5',palette='gist_rainbow')
pair.fig.set_size_inches(6,6)
plt.show()

####### PRIN TO DIAGRAMMA, TREXW GIA ALLAGH TWN LABELS ######
lab = np.reshape(db.labels_,[39687,1])
np.concatenate([xclust,lab],axis=1) 
dat =  np.concatenate([xclust,lab],axis=1) 
dat = pd.DataFrame(dat)   
dat.columns = ['0','1','2','3','4','5']

####### DIAGRAMMATA ANA 2 ME LABELS TWN CLUSTERS ###### 
sns.set(style='ticks',color_codes=True)
pair = sns.pairplot(dat,vars=['0','1','2','3','4'],hue='5',palette='gist_rainbow')
pair.fig.set_size_inches(6,6)
plt.show()

#####################################################################
                       # CLASSIFICATION #
#####################################################################

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from numpy import asarray

X = fs_modeltr.copy()
X = pd.concat([data_num,data_cat1],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

#clf = GaussianNB()
clf = LogisticRegression(max_iter=4000)
#clf = LinearSVC(C=1.0, max_iter=4000)
#clf = RandomForestClassifier()
#clf = DecisionTreeClassifier(random_state=1234)


clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)

print('Accuracy = ' + str(clf.score(X_test,y_test)))


import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats

X2 = sm.add_constant(X)
est = sm.Logit(y, X2)
est2 = est.fit()
print(est2.summary())

#########################################################
                  #ASSOCIATION_RULES#
#########################################################
import numpy as np
import pandas as pd

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder

ydf = pd.DataFrame(y)
repl = {0:'Not_Open_Account',1:'Open_Account'}
ydf = ydf.replace(repl)

dfc_cat_tr['default']=dfc_cat_tr['default'].replace({'no':'no\_default','unknown':'unknown\_default','yes':'yes\_default'})
dfc_cat_tr['housing']=dfc_cat_tr['housing'].replace({'no':'no\_housing','unknown':'unknown\_housing','yes':'yes\_housing'})
dfc_cat_tr['loan']=dfc_cat_tr['loan'].replace({'no':'no\_loan','unknown':'unknown\_loan','yes':'yes\_loan'})


dfc_cat_tr.reset_index(drop=True,inplace=True)
X = pd.concat([dfc_cat_tr,ydf],axis=1)
X.reset_index(drop=True,inplace=True)
X.rename(columns={0:'y'},inplace=True)

X = X.values.tolist()
te = TransactionEncoder()
te_ary = te.fit(X).transform(X)

dfar = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets = apriori(dfar, min_support=0.5, use_colnames=True)
print(frequent_itemsets)


rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

#print(rules[['antecedents','consequents','confidence']])


rules["antecedents"] = rules["antecedents"].apply(lambda x: list(x)[0]).astype("unicode")
rules["consequents"] = rules["consequents"].apply(lambda x: list(x)[0]).astype("unicode")



#sns.heatmap(pd.DataFrame(cm), annot=True,fmt='g')

