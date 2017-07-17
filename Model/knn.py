import csv
import random
import math
import operator
import numpy
import pandas
from scipy.spatial import distance
from numpy.random import permutation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.neighbors import KNeighborsRegressor



with open("Oracle1.csv", 'r') as csvfile:
	stock = pandas.read_csv(csvfile)
	for row in reversed(list(csv.reader(csvfile))):
		print ', '.join(row)


selected_date = stock.iloc[0]
distance_columns = ['Close', 'DailyReturn']


def euclidean_distance(row):
	inner_value = 0
	for k in distance_columns:
		inner_value += (row[k] - selected_date[k])** 2
	return math.sqrt(inner_value)


date_distance = stock.apply(euclidean_distance, axis = 1)
stock_numeric = stock[distance_columns]
stock_normalized = (stock_numeric - stock_numeric.mean()) / stock_numeric.std()
stock_normalized.fillna(0, inplace = True)
date_normalized = stock_normalized[stock["Date"] == "2016-06-29"]
euclidean_distances = stock_normalized.apply(lambda row: distance.euclidean(row, date_normalized), axis = 1)
distance_frame = pandas.DataFrame(data = {"dist": euclidean_distances, "idx": euclidean_distances.index})
distance_frame.sort("dist", inplace=True) 
second_smallest = distance_frame.iloc[1]["idx"]
most_similar_to_date = stock.loc[int(second_smallest)]["Date"]

test_cutoff = math.floor(len(stock)/3)
test = stock.loc[stock.index[1:test_cutoff]]
train = stock.loc[stock.index[test_cutoff:]]
x_column = ['Close', 'DailyReturn']
y_column = ['CloseNext']


knn = KNeighborsRegressor (n_neighbors = 2)
knn.fit(train[x_column], train[y_column])
predictions = knn.predict(test[x_column])
print predictions

actual = test[y_column]


plt.plot(predictions, 'r')
plt.plot(actual, 'g')
plt.xlabel('Day in Future', fontsize=30)
plt.ylabel('Price', fontsize=30)
red_patch = mpatches.Patch(color='red', label='Predicted Price')
green_patch = mpatches.Patch(color='green', label='Actual Price')
plt.legend(handles=[red_patch])
plt.legend(handles=[red_patch, green_patch])
plt.show()


mse = (((predictions - actual) ** 2).sum())/len(predictions)
print mse