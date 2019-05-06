from flask import Flask
from flask import request
import pymysql
import pandas as pd
import numpy as np
import json
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from flask import Response
import matplotlib.pyplot as plt
from flask import render_template
app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/review/review_predict', methods=["GET", "POST"])
def review_predict():
    review_id = request.args.get("reviewid")
    sql = 'select length(text) AS textlength, review_count, average_stars, funny_vote_count, useful_vote_count, cool_vote_count, fans from review inner join user_friends on review.user_id = user_friends.friend_id inner join user on user_friends.user_id = user.user_id where review_id="%s" limit 1'%(review_id)
    conn = pymysql.connect(host='localhost', user='root', password='12345678', db='yelp')
    df = pd.read_sql(sql, con=conn)
    df = preprocessing.minmax_scale(df)
    df = np.array(df)
    clf = joblib.load('service/review_model_clean.m')
    y_pred = clf.predict(df)
    consequence = {}
    if y_pred[0] == 0:
        consequence['con'] = '0-2'
    else:
        consequence['con'] = '3-5'
    conn.close()
    return json.dumps(consequence)


@app.route('/business/business_predict_US', methods=["GET", "POST"])
def business_predict():
    business_id = request.args.get("businessid")
    sql = 'select state, latitude, longitude, review_count AS Y from business inner join business_categories using (business_id) where business_id = "%s" limit 1'%(business_id)
    conn = pymysql.connect(host='localhost', user='root', password='12345678', db='yelp')
    df = pd.read_sql(sql, con=conn)
    encoder = preprocessing.LabelEncoder()
    encoder.classes_ = np.load('service/classes.npy')
    df['state'] = encoder.transform(df['state'])
    df = preprocessing.minmax_scale(df)
    df = np.array(df)
    clf = joblib.load('service/bussiness_model_US.m')
    y_pred = clf.predict(df)
    consequence = {}
    if y_pred[0] == 0:
        consequence['con'] = '0-2'
    else:
        consequence['con'] = '3-5'
    conn.close()
    return json.dumps(consequence)


@app.route('/review/all_review_predict', methods=["GET", "POST"])
def review_predict_all():
    review_id = request.args.get("reviewid")
    sql = 'select length(text) AS textlength, review_count, average_stars, funny_vote_count, useful_vote_count, cool_vote_count, fans from review inner join user_friends on review.user_id = user_friends.friend_id inner join user on user_friends.user_id = user.user_id where review_id="%s" limit 1'%(review_id)
    conn = pymysql.connect(host='localhost', user='root', password='12345678', db='yelp')
    df = pd.read_sql(sql, con=conn)
    df = preprocessing.minmax_scale(df)
    df = np.array(df)
    clf = joblib.load('service/review_model_unclean.m')
    y_pred = clf.predict(df)
    consequence = {}
    if y_pred[0] == 0:
        consequence['con'] = '0-2'
    else:
        consequence['con'] = '3-5'
    conn.close()
    return json.dumps(consequence)


@app.route('/business/all_business_predict', methods=["GET", "POST"])
def business_predict_all():
    business_id = request.args.get("businessid")
    sql = 'select state, latitude, longitude, review_count AS Y from business inner join business_categories using (business_id) where business_id = "%s" limit 1'%(business_id)
    conn = pymysql.connect(host='localhost', user='root', password='12345678', db='yelp')
    df = pd.read_sql(sql, con=conn)
    encoder = preprocessing.LabelEncoder()
    encoder.classes_ = np.load('service/classes.npy')
    df['state'] = encoder.transform(df['state'])
    df = preprocessing.minmax_scale(df)
    df = np.array(df)
    clf = joblib.load('service/bussiness_model_all.m')
    y_pred = clf.predict(df)
    consequence = {}
    if y_pred[0] == 0:
        consequence['con'] = '0-2'
    else:
        consequence['con'] = '3-5'
    conn.close()
    return json.dumps(consequence)


@app.route('/validation/review_model', methods=["GET", "POST"])
def model_validation():
    clf1 = joblib.load('service/review_model_clean.m')
    data = pd.read_csv('service/review_mark.csv')
    y = data['Y']
    y.loc[data['Y'] == 1.0] = 0
    y.loc[data['Y'] == 2.0] = 0
    y.loc[data['Y'] == 3.0] = 1
    y.loc[data['Y'] == 4.0] = 1
    y.loc[data['Y'] == 5.0] = 1

    X = data[['textlength', 'review_count', 'average_stars', 'funny_vote_count', 'useful_vote_count', 'cool_vote_count',
              'fans']]
    X = preprocessing.minmax_scale(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=10)
    clf1.fit(X_train, y_train)
    y_pred = clf1.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    consequence = {}
    consequence['acc'] = accuracy
    consequence['precision'] = precision
    consequence['recall'] = recall
    return json.dumps(consequence)


@app.route('/validation/allreview_model', methods=["GET", "POST"])
def model_validation2():
    clf1 = joblib.load('service/review_model_unclean.m')
    data = pd.read_csv('service/review_mark_unclean.csv')
    y = data['Y']
    y.loc[data['Y'] == 1.0] = 0
    y.loc[data['Y'] == 2.0] = 0
    y.loc[data['Y'] == 3.0] = 1
    y.loc[data['Y'] == 4.0] = 1
    y.loc[data['Y'] == 5.0] = 1

    X = data[['textlength', 'review_count', 'average_stars', 'funny_vote_count', 'useful_vote_count', 'cool_vote_count',
              'fans']]
    X = preprocessing.minmax_scale(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=20)
    clf1.fit(X_train, y_train)
    y_pred = clf1.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    consequence = {}
    consequence['acc'] = accuracy
    consequence['precision'] = precision
    consequence['recall'] = recall
    return json.dumps(consequence)


@app.route('/validation/business_model_USA', methods=["GET", "POST"])
def model_validation3():
    le = preprocessing.LabelEncoder()
    data = pd.read_csv('service/business_mark.csv')
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
    # np.save('classes.npy', le.classes_)
    X['state'] = le.transform(X['state'])
    X = preprocessing.minmax_scale(X)
    le.fit(y)
    y = le.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    clf = joblib.load('service/bussiness_model_US.m')
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    # print("Accuracy:", accuracy)
    precision = precision_score(y_test, y_pred)
    # print("Precision", precision)
    recall = recall_score(y_test, y_pred)
    # print("recall", recall)
    consequence = {}
    consequence['acc'] = accuracy
    consequence['precision'] = precision
    consequence['recall'] = recall
    return json.dumps(consequence)


@app.route('/validation/business_model_all', methods=["GET", "POST"])
def model_validation4():
    le = preprocessing.LabelEncoder()
    data = pd.read_csv('service/business_mark_unclean.csv')
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
    # np.save('classes.npy', le.classes_)
    X['state'] = le.transform(X['state'])
    X = preprocessing.minmax_scale(X)
    le.fit(y)
    y = le.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    clf = joblib.load('service/bussiness_model_all.m')
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    # print("Accuracy:", accuracy)
    precision = precision_score(y_test, y_pred)
    # print("Precision", precision)
    recall = recall_score(y_test, y_pred)
    # print("recall", recall)
    consequence = {}
    consequence['acc'] = accuracy
    consequence['precision'] = precision
    consequence['recall'] = recall
    return json.dumps(consequence)

if __name__ == '__main__':
    app.run()


# reviewid=---z1dGqzrcq_MD5mE-dqA
# businessid=_aYYRuwTXItHvrc19cNqZg