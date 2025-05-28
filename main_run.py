
import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from lazypredict.Supervised import  LazyClassifier
from ydata_profiling import ProfileReport

data=pd.read_csv('CYBER.csv')
print(data.columns)
print(data.isna().sum())
print(data.info())
print(data.describe())
print('-----------------------')

print(data.Label.value_counts())


lab=LabelEncoder()
for i in data.select_dtypes(include='object').columns.values:
    if len(data[i].value_counts().values)<20:
        print(data[i].value_counts())
        data[i]=lab.fit_transform(data[i])

print(data.Label.value_counts())


X=[]
for i in data.select_dtypes(include='number').columns.values:
    data['z-scores']=(data[i]-data[i].mean())/data[i].std()
    outliers=np.abs(data['z-scores']>3).sum()
    if outliers > 0:
        X.append(i)

print(len(data))
thresh=3
for i in X[:25]:
    upper=data[i].mean()+thresh*data[i].std()
    lower=data[i].mean()-thresh*data[i].std()
    data=data[(data[i]>lower)&(data[i]<upper)]

print(len(data))

print(data['Label'].value_counts())


corr=data.corr()['Label']
corr=corr.drop(['z-scores','Label'])
x=[i for i in corr.index if corr[i]>0.2]
for i in corr.index:
    print(i," : ",corr[i])

y=data['Label']
x=data[['fwd_pkt_len_min','bwd_pkt_len_min','flow_iat_min','pkt_len_min']]

print(x.columns.values)
print(data['Label'].value_counts())

smote=SMOTE()
x,y=smote.fit_resample(x,y)
x_train,x_test,y_train,y_test=train_test_split(x,y)

light = LGBMClassifier()
light.fit(x_train, y_train)
joblib.dump(light,'the_light.pkl')



print("The light GBM ", light.score(x_test, y_test))
print("The feature importance of the LightGBM classifier")
print(x.columns, " : ", light.feature_importances_)
print("The feature names ", light.feature_name_)
print('---------prediction------------------')
print(light.predict([x_test.values[0]]))
print(y_test.values[0])
print('------------------------')
print('-------------------------------------------------------------')
print(confusion_matrix(y_test,(light.predict(x_test))))
print('-------------------------------------------------------')

ext = ExtraTreesClassifier()
ext.fit(x_train, y_train)
joblib.dump(ext,'the_ext.pkl')
print('The extra tree classifier ', ext.score(x_test, y_test))
print("The feature importance of ExtratreeClassifier")
print(x.columns, " : ", ext.feature_importances_)
print("The max features ", ext.max_features)
print('-------------------------------------------------------------')
print(confusion_matrix(y_test,(ext.predict(x_test))))
print('-------------------------------------------------------')

rf = RandomForestClassifier()
rf.fit(x_train, y_train)
joblib.dump(rf,'the_rf.pkl')
print("The Random forest ", rf.score(x_test, y_test))
print("The feature importance of the LightGBM classifier")
print(x.columns, " : ", rf.feature_importances_)
print('-------------------------------------------------------------')
print(confusion_matrix(y_test,(rf.predict(x_test))))
print('-------------------------------------------------------')


dtree=DecisionTreeClassifier()
dtree.fit(x_train,y_train)
print("The decision Tree ",dtree.score(x_test,y_test))
print('-------------------------------------------------------------')
print(confusion_matrix(y_test,(dtree.predict(x_test))))
print('-------------------------------------------------------')


lr=LogisticRegression(max_iter=350)
lr.fit(x_train,y_train)
print("The logistic regression  ",lr.score(x_test,y_test))
print('-------------------------------------------------------------')
print(confusion_matrix(y_test,(lr.predict(x_test))))
print('-------------------------------------------------------')

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from lazypredict.Supervised import  LazyClassifier
from ydata_profiling import ProfileReport

data=pd.read_csv('CYBER.csv')
print(data.columns)
print(data.isna().sum())
print(data.info())
print(data.describe())
print('-----------------------')

print(data.Label.value_counts())


lab=LabelEncoder()
for i in data.select_dtypes(include='object').columns.values:
    if len(data[i].value_counts().values)<20:
        print(data[i].value_counts())
        data[i]=lab.fit_transform(data[i])

print(data.Label.value_counts())


X=[]
for i in data.select_dtypes(include='number').columns.values:
    data['z-scores']=(data[i]-data[i].mean())/data[i].std()
    outliers=np.abs(data['z-scores']>3).sum()
    if outliers > 0:
        X.append(i)

print(len(data))
thresh=3
for i in X[:25]:
    upper=data[i].mean()+thresh*data[i].std()
    lower=data[i].mean()-thresh*data[i].std()
    data=data[(data[i]>lower)&(data[i]<upper)]

print(len(data))

print(data['Label'].value_counts())


corr=data.corr()['Label']
corr=corr.drop(['z-scores','Label'])
x=[i for i in corr.index if corr[i]>0.2]
for i in corr.index:
    print(i," : ",corr[i])

y=data['Label']
x=data[['fwd_pkt_len_min','bwd_pkt_len_min','flow_iat_min','pkt_len_min']]

print(x.columns.values)
print(data['Label'].value_counts())

smote=SMOTE()
x,y=smote.fit_resample(x,y)
x_train,x_test,y_train,y_test=train_test_split(x,y)

light = LGBMClassifier()
light.fit(x_train, y_train)
joblib.dump(light,'the_light.pkl')



print("The light GBM ", light.score(x_test, y_test))
print("The feature importance of the LightGBM classifier")
print(x.columns, " : ", light.feature_importances_)
print("The feature names ", light.feature_name_)
print('---------prediction------------------')
print(light.predict([x_test.values[0]]))
print(y_test.values[0])
print('------------------------')
print('-------------------------------------------------------------')
print(confusion_matrix(y_test,(light.predict(x_test))))
print('-------------------------------------------------------')

ext = ExtraTreesClassifier()
ext.fit(x_train, y_train)
joblib.dump(ext,'the_ext.pkl')
print('The extra tree classifier ', ext.score(x_test, y_test))
print("The feature importance of ExtratreeClassifier")
print(x.columns, " : ", ext.feature_importances_)
print("The max features ", ext.max_features)
print('-------------------------------------------------------------')
print(confusion_matrix(y_test,(ext.predict(x_test))))
print('-------------------------------------------------------')

rf = RandomForestClassifier()
rf.fit(x_train, y_train)
joblib.dump(rf,'the_rf.pkl')
print("The Random forest ", rf.score(x_test, y_test))
print("The feature importance of the LightGBM classifier")
print(x.columns, " : ", rf.feature_importances_)
print('-------------------------------------------------------------')
print(confusion_matrix(y_test,(rf.predict(x_test))))
print('-------------------------------------------------------')


dtree=DecisionTreeClassifier()
dtree.fit(x_train,y_train)
print("The decision Tree ",dtree.score(x_test,y_test))
print('-------------------------------------------------------------')
print(confusion_matrix(y_test,(dtree.predict(x_test))))
print('-------------------------------------------------------')


lr=LogisticRegression(max_iter=350)
lr.fit(x_train,y_train)
print("The logistic regression  ",lr.score(x_test,y_test))
print('-------------------------------------------------------------')
print(confusion_matrix(y_test,(lr.predict(x_test))))
print('-------------------------------------------------------')
