import csv
from sklearn import tree
import numpy as np
from sklearn.cross_validation import train_test_split
import decision_tree_train as my_tree


DATA_SIZE = 1000000

data = [[0 for x in range(10)] for y in range(DATA_SIZE)]
i = 0
with open('test.csv', 'r') as csvfile:
	print('Reading test CSV file')
	reader = csv.reader(csvfile, delimiter=',')
	for row in reader:
		for j in range(1, 11):
			data[:][i][j-1] = int(row[j])
		if(i % 100 == 0):
			print('line')
			print(i)
		i = i + 1
		if(i >= DATA_SIZE):
			break
print('Training in progress')
clf = my_tree.train_dec_tree()
pred_list = []
print('Predicting results')
for i in range(DATA_SIZE):
	#print(data[:][i])
	data[i] = my_tree.extract_features(data[:][i])
	#print(data[:][i])
	pred_list.append(clf.predict(np.asarray(data[:][i]).reshape(1,-1)))

print('Writing results')
with open('res.csv', 'w') as csvfile:
	writer = csv.writer(csvfile)
	for pred in pred_list:
		writer.writerow(str(pred[0]))

