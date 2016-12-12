import csv
from sklearn import tree
import numpy as np
from sklearn.cross_validation import train_test_split
import decision_tree_train as my_tree
import pandas as pd
import tensorflow as tf


DATA_SIZE = 1000000

print('Training in progress')
clf = my_tree.train_dec_tree()

data = [[0 for x in range(10)] for y in range(DATA_SIZE)]
COLUMNS = ["suit1","card1","suit2","card2","suit3","card3","suit4","card4","suit5","card5"]
print('Reading test file')
df_test = pd.read_csv(tf.gfile.Open('test_no_serial.csv'),names=COLUMNS,skipinitialspace=False,engine="python",nrows=DATA_SIZE)
for i in range(0,DATA_SIZE):
	if(i % 100000 == 0):
		print(i)
	for j in range(0,10):
		data[i][j] = df_test.loc[i].loc[COLUMNS[j]]

pred_list = []
print('Predicting results')
for i in range(DATA_SIZE):
	if(i % 10000 == 0):
		print(i)
	#print(data[:][i])
	data[i] = my_tree.extract_features(data[:][i])
	#print(data[:][i])
	pred_list.append(clf.predict(np.asarray(data[:][i]).reshape(1,-1)))

print('Writing results')
i = 1
with open('res.csv', 'w') as csvfile:
	writer = csv.writer(csvfile)
	for pred in pred_list:
		writer.writerow(str(i))
		writer.writerow(str(pred[0]))
		i = i + 1

