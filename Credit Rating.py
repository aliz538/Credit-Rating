#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install opendatasets -q')


# In[7]:


import opendatasets as od
od.download("https://www.kaggle.com/agewerc/corporate-credit-rating")


# In[57]:


import pandas as pd
import numpy as np
from numpy import loadtxt
from numpy import sort
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.ticker import PercentFormatter
import matplotlib.ticker as mtick
from random import sample
import seaborn as sns
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.utils import resample
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold ,cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel


# In[29]:


# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


# In[10]:


Dataset=pd.read_csv("corporate-credit-rating\corporate_rating.csv")


# In[11]:


Dataset.head()


# In[12]:


Dataset.info()


# In[14]:


Dataset.shape


# In[16]:


Dataset['Date'] = pd.to_datetime(Dataset['Date'])
Dataset['Year'] = Dataset['Date'].dt.year


# In[18]:


Dataset.isnull().sum()


# In[19]:


Dataset.describe()


# In[31]:


Dataset["Rating"].value_counts()


# In[43]:


Dataset['Rating'].value_counts().plot(kind='bar',
                                             figsize=(8,4),
                                             title="Count of Rating by Type",
                                             grid=True)


# In[22]:


Dataset.plot(kind="scatter", x="Rating", y="freeCashFlowPerShare")


# In[26]:


Max_index=Dataset[Dataset["daysOfSalesOutstanding"]==Dataset["daysOfSalesOutstanding"].max()].index


# In[34]:


Dataset_final=Dataset.drop(["Name","Symbol","Date"], axis=1)


# In[ ]:


# Drawing Boxplots

figure, axes = plt.subplots(nrows=8, ncols=3, figsize=(20,44))

i = 0 
j = 0

for c in Dataset_final.columns[6:30]:
    
    sns.boxplot(x=Dataset_final.Rating, y=Dataset_final[c], palette="Set3", ax=axes[i, j])
    
    if j == 2:
        j=0
        i+=1
    else:
        j+=1   


# In[42]:


X=Dataset_final.drop("Rating", axis=1)
y=Dataset_final["Rating"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f' Training {X_train.shape}, Testing {X_test.shape} ')


# In[46]:


Dataset_final.columns


# In[47]:


num_attr=['currentRatio', 'quickRatio',
       'cashRatio', 'daysOfSalesOutstanding', 'netProfitMargin',
       'pretaxProfitMargin', 'grossProfitMargin', 'operatingProfitMargin',
       'returnOnAssets', 'returnOnCapitalEmployed', 'returnOnEquity',
       'assetTurnover', 'fixedAssetTurnover', 'debtEquityRatio', 'debtRatio',
       'effectiveTaxRate', 'freeCashFlowOperatingCashFlowRatio',
       'freeCashFlowPerShare', 'cashPerShare', 'companyEquityMultiplier',
       'ebitPerRevenue', 'enterpriseValueMultiple',
       'operatingCashFlowPerShare', 'operatingCashFlowSalesRatio',
       'payablesTurnover']

cat_attr=['Rating Agency Name', 'Sector','Year']


# In[49]:


full_pipe=ColumnTransformer([
    ('num', StandardScaler(), num_attr),
    ('cat', OneHotEncoder(sparse=False, categories='auto'), cat_attr)],
        remainder='passthrough'
)

X_train_prepared=full_pipe.fit_transform(X_train)


# In[50]:


pd.DataFrame(X_train_prepared).head()


# In[53]:


ada_clf=AdaBoostClassifier(
DecisionTreeClassifier(max_depth=1), n_estimators=200, algorithm="SAMME.R", learning_rate=0.5
)


# In[55]:


models = {
    'NB' : GaussianNB(),
    'xgboost' : XGBClassifier(),
    'AdaBoost' : ada_clf,
    'gradient boosting' : GradientBoostingClassifier(),
    'Logistic regression' : LogisticRegression(),
    'random forest' : RandomForestClassifier(),
    'Decision Tree' : DecisionTreeClassifier(),
    'SVM': LinearSVC(),
    'knn' : KNeighborsClassifier(n_neighbors = 4)
}


# In[58]:


results = {}
for name, model in models.items():
    model.fit(X_train_prepared, y_train)
    scores = cross_val_score(model, X_train_prepared, y_train, scoring = 'accuracy', cv=5)
    results[name] = round(np.mean(scores),3)

    print(f'{name} trained')
    print("Data shows that the mean is %0.2f and the standard deviation is %0.2f" % (scores.mean(), scores.std()))


# In[59]:


results


# In[60]:


results_df = pd.DataFrame(results, index=range(0,1)).T.rename(columns={0: 'Accuracy'}).sort_values('Accuracy', ascending=False)
results_df


# In[61]:


plt.figure(figsize = (20, 6))
sns.barplot(x= results_df.index, y = results_df['Accuracy'], palette = 'winter')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy of different models');


# In[62]:


RSME_Test={}

for name , model in models.items():
    Res=accuracy_score(y_test, model.predict(full_pipe.transform(X_test)))
    RSME_Test[name]=round(Res,3)


# In[63]:


RSME_Test


# In[66]:


test_df = pd.DataFrame(RSME_Test, index=range(0,1)).T.rename(columns={0: 'Accuracy'}).sort_values('Accuracy', ascending=False)
test_df


# In[65]:


plt.figure(figsize = (20, 6))
sns.barplot(x= test_df.index, y = test_df['Accuracy'], palette = 'winter')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy of different test models');


# In[71]:


from sklearn.model_selection import GridSearchCV
forest_clf = RandomForestClassifier()

param_grid = [
 {'n_estimators': [ 10, 30, 40, 50, 70], 'max_features': [2, 4, 6, 8]},
 {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
 ]

grid_search = GridSearchCV(forest_clf, param_grid, cv=5)
grid_search.fit(X_train_prepared, y_train)


# In[72]:


grid_search.best_estimator_


# In[79]:


forest_clf_opt= RandomForestClassifier()
forest_clf_opt.fit(X_train_prepared, y_train)
for name, score in zip(list(X_train), forest_clf_opt.feature_importances_):
    print(name, round(score,2))


# In[ ]:


y_pred = forest_clf_opt.predict(full_pipe.transform(X_test))
accuracy_score(y_test, y_pred)


# In[76]:


confusion_matrix(y_test, y_pred)


# In[ ]:




