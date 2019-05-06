import util.NaiveBayes as NB
import util.NB as NB2

import pandas as pd
import pymysql
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import preprocessing
import numpy as np

# conn = pymysql.connect(host='localhost', user='root', password='12345678',
#                        db='yelp')
# QC = "'QC'"
# ON1 = "'ON'"
# sql = 'select city, state, latitude, longitude, review_count, stars AS Y from clean_business inner join clean_business_categories using (business_id) where review_count>10 and is_open = 1 and state!=%s and state!=%s'%(QC, ON1)
#
# df = pd.read_sql(sql, con=conn)
# df.to_csv("business_mark.csv")
#
# print(df.head())
# conn.close()
le = preprocessing.LabelEncoder()
data = pd.read_csv('business_mark.csv')
y = data['Y']
y.loc[data['Y'] == 1.0] = 0
y.loc[data['Y'] == 1.5] = 0
y.loc[data['Y'] == 2.0] = 0
y.loc[data['Y'] == 2.5] = 0
y.loc[data['Y'] == 3.0] = 1
y.loc[data['Y'] == 3.5] = 1
y.loc[data['Y'] == 4.0] = 1
y.loc[data['Y'] == 4.5] = 1
y.loc[data['Y'] == 5.0] = 1
# y.loc[data['Y'] == 5.0] = 1
X = data[['state', 'latitude', 'longitude', 'review_count']]

# le.fit(X['city'])
# X['city'] = le.transform(X['city'])
le.fit(X['state'])
np.save('classes.npy', le.classes_)
X['state'] = le.transform(X['state'])
X = preprocessing.minmax_scale(X)
le.fit(y)
y = le.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
clf = NB.NaiveBayes()
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
precision = precision_score(y_test, y_pred)
print("Precision", precision)
recall = recall_score(y_test, y_pred)
print("recall", recall)
joblib.dump(clf, 'bussiness_model_US.m')
# print(X.head())

