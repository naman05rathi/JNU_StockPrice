import csv
import numpy
import pandas
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets.mldata import fetch_mldata
from sklearn.ensemble import RandomForestRegressor


stock = pandas.read_csv("Oracle1.csv") 
print(stock.columns)


kmeans_model = KMeans(n_clusters=5, random_state=1)
good_columns = stock._get_numeric_data()
kmeans_model.fit(good_columns)
labels = kmeans_model.labels_
pca_2 = PCA(2)


columns = stock.columns.tolist()
columns = [c for c in columns if c not in ["Close", "Date"]]
target = "Close"


train=stock[:-100]
test=stock[-100:]
arr=[]
seed=7
for value in test[target]:
	arr.append(value)

arr1=[]

for value in test['Date']:
	arr1.append(value)

actual=arr
model = LinearRegression()
model.fit(train[columns], train[target])
predictions = model.predict(test[columns])
root=mean_squared_error(predictions, test[target])


print root
print arr1
print actual
print predictions


plt.plot(predictions, 'r')
plt.plot(actual, 'g')
plt.xlabel('Day in Future', fontsize=30)
plt.ylabel('Price', fontsize=30)
red_patch = mpatches.Patch(color='red', label='Predicted Price')
green_patch = mpatches.Patch(color='green', label='Actual Price')
plt.legend(handles=[red_patch])
plt.legend(handles=[red_patch, green_patch])
plt.show()


model1 = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)
model1.fit(train[columns], train[target])
predictions1 = model1.predict(test[columns])
mean_squared_error(predictions, test[target])