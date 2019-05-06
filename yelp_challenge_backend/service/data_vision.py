import numpy as np
import matplotlib.pyplot as plt
import pymysql
import pandas as pd
conn = pymysql.connect(host='localhost', user='root', password='12345678',
                       db='yelp')
sql = 'select stars, date from review where business_id = "%s" order by date'%('3DHwRjWomOtnxM1v6-bPEQ')
df = pd.read_sql(sql, con=conn)
x = df['date']
y = df['stars']
plt.plot(x,y)
plt.show()
# data = pd.read_csv('review_mark.csv')
# data1 = data[['textlength', 'review_count']][(data.Y==1)]
# data2 = data[['textlength', 'review_count']][(data.Y==2)]
# data3 = data[['textlength', 'review_count']][(data.Y==3)]
# data4 = data[['textlength', 'review_count']][(data.Y==4)]
# data5 = data[['textlength', 'review_count']][(data.Y==5)]
# # print(data1)
# fig = plt.figure()
# ax = plt.subplot()
# ax.scatter(data1['textlength'], data1['review_count'], c='blue', s=100, label = 'rate=1')
# ax.scatter(data2['textlength'], data2['review_count'], c='green',  s=15,label ='rate=2')
# ax.scatter(data3['textlength'], data3['review_count'], c='violet', s=10, label ='rate=3')
# ax.scatter(data4['textlength'], data4['review_count'], c='red', s=15, label ='rate=4')
# ax.scatter(data5['textlength'], data5['review_count'], c='grey', s=15, label ='rate=5')
# plt.xlabel('textlength')
# plt.ylabel('review_count')
# plt.legend()
# plt.savefig('rate.svg')
# plt.show()


# print(x1)
