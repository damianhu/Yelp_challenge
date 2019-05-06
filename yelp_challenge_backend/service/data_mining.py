import util.NaiveBayes as NB

import pandas as pd
import pymysql
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib
import numpy as np

# conn = pymysql.connect(host='localhost', user='root', password='12345678',
#                        db='yelp')
# sql = 'select stars AS Y, length(text) AS textlength, review_count, average_stars, funny_vote_count, useful_vote_count, cool_vote_count, fans from review inner join user_friends on review.user_id = user_friends.friend_id inner join user on user_friends.user_id = user.user_id where review_count>10'
#
# df = pd.read_sql(sql, con=conn)
# df.to_csv("review_mark.csv")
#
# print(df.head())
# conn.close()
data = pd.read_csv('review_mark.csv')
y = data['Y']
y.loc[data['Y'] == 1.0] = 0
y.loc[data['Y'] == 2.0] = 0
y.loc[data['Y'] == 3.0] = 1
y.loc[data['Y'] == 4.0] = 1
y.loc[data['Y'] == 5.0] = 1

X = data[['textlength','review_count','average_stars','funny_vote_count','useful_vote_count','cool_vote_count','fans']]
X = preprocessing.minmax_scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=10)
clf = NB.NaiveBayes()
# X_train = np.array(X_train)
# y_train = np.array(y_train)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
precision = precision_score(y_test, y_pred)
print("Precision", precision)
recall = recall_score(y_test, y_pred)
print("recall", recall)

# joblib.dump(clf, 'review_model_clean.m')
# gnb = GaussianNB()
# gnb.fit(X_train, y_train)
# y_pred = gnb.predict(X_test)
# accuracy2 = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy2)

# print(X)