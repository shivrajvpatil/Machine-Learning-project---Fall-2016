import csv
from sklearn import tree
import numpy as np
from sklearn.cross_validation import train_test_split


DATA_SIZE = 25010
TRAIN_DATA = 20000

def perform_training(max_depth,train_X,train_Y):
	clf = tree.DecisionTreeClassifier(max_depth = max_depth, random_state = 40)
	clf = clf.fit(np.asarray(train_X[:][:TRAIN_DATA]), np.asarray(train_Y[:TRAIN_DATA]).reshape(-1,1))
	return clf

def calc_Score(clf,train_X,train_Y):
	countPos = 0
	count = 0
	for i in range(len(train_Y)):
		pred = clf.predict(np.asarray(train_X[:][i]).reshape(1,-1))
		expected = train_Y[i]
		if(pred == expected):
			countPos = countPos + 1
		count = count + 1
	score = countPos/count
	return score


def write_res_CSV(clf,train_X,train_Y):
	res_dict = {}
	for i in range(len(train_Y)):
		res_dict[i] = []
		res_dict[i].append(clf.predict(np.asarray(train_X[:][i]).reshape(1,-1)))
		res_dict[i].append(train_Y[i])

	with open('res.csv', 'w') as csvfile:
		writer = csv.writer(csvfile)
		for key, value in res_dict.items():
			writer.writerow([value[1], value[0]])	

def cards_sort(cards):
	card_dict = {}
	card_dict[1] = []
	card_dict[2] = []
	card_dict[3] = []
	card_dict[4] = []
	sorted_cards = []
	for i in range(0,9,2):
		suit = cards[i]
		card_no = cards[i+1]
		#print(suit)
		card_dict[suit].append(card_no)
	k = 0
	for suit in range(1,4):
		card_dict[suit] = sorted(card_dict[suit])
		for j in range(len(card_dict[suit])):
			cards[k] = suit
			cards[k+1] = card_dict[suit][j]
			k = k + 2


train_data = [[0 for x in range(11)] for y in range(DATA_SIZE)]
i = 0
with open('train.csv', 'r') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	for row in reader:
		for j in range(0, 11):
			train_data[:][i][j] = int(row[j])
		i = i + 1
		if(i >= DATA_SIZE):
			break

train_X = [[0 for x in range(10)] for y in range(DATA_SIZE)]
for i in range(DATA_SIZE):
	train_X[i] = train_data[:][i][:-1]
	cards_sort(train_X[i])

train_Y = [0 for x in range(DATA_SIZE)]
for i in range(DATA_SIZE):
	train_Y[i] = train_data[:][i][10]

i = 0
test_data = [[0 for x in range(11)] for y in range(19)]
with open('test1.csv', 'r') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	for row in reader:
		for j in range(1, 11):
			test_data[:][i][j] = int(row[j])
		i = i + 1
		if(i >= DATA_SIZE):
			break


test_X = [[0 for x in range(11)] for y in range(19)]
for i in range(19):
	#print(test_data[:][i][:-1])
	test_X[i] = test_data[:][i][1:11]
	cards_sort(test_X[i])
	#print(test_X[i])

f_train = open('train_error.csv','w')
f_test = open('test_error.csv','w')
for max_depth in range(1,100,1):
	rand_train_X, rand_test_X = train_test_split(train_X, test_size = 0.2)
	rand_train_Y, rand_test_Y = train_test_split(train_Y, test_size = 0.2)
	clf = perform_training(max_depth,rand_train_X,rand_train_Y)
	#write_res_CSV(clf,train_X,train_Y)
	print('max_depth : '+str(max_depth)+', testing score : '+str(calc_Score(clf,rand_test_X,rand_test_Y)))
	print('max_depth : '+str(max_depth)+', training score : '+str(calc_Score(clf,rand_train_X,rand_train_Y)))
	f_train.write(str(max_depth)+','+str(1-calc_Score(clf,rand_train_X,rand_train_Y))+'\n')
	f_test.write(str(max_depth)+','+str(1-calc_Score(clf,rand_test_X,rand_test_Y))+'\n')
f_train.close()
f_test.close()

