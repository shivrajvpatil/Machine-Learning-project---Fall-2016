import csv
from sklearn import tree
import numpy as np
from sklearn.cross_validation import train_test_split

TRAIN_SIZE = 5010
DATA_SIZE = 20000# #1000#
EN_FEATURE_EXTRACT = False

def perform_training(max_depth,train_X,train_Y):
	for i in range(20):
		# print('$$$$$$$$$$$')
		# print(train_X[:][i])
		# print(train_Y[i])
		# print('$$$$$$$$$$$')
		clf = tree.DecisionTreeClassifier(max_depth = max_depth, random_state = 40)
		#clf = clf.fit(np.asarray(train_X[:][:TRAIN_DATA]), np.asarray(train_Y[:TRAIN_DATA]).reshape(-1,1))
		clf = clf.fit(np.asarray(train_X[:][:len(train_Y)]), np.asarray(train_Y[:len(train_Y)]).reshape(-1,1))
	return clf

def calc_Score(clf,data):
	X = [[0 for x in range(10)] for y in range(len(data[:]))]
	Y = [0 for x in range(len(data[:]))]
	for i in range(len(data[:])):
		X[i] = data[:][i][:-1]
		# print(X[:][i])
		Y[i] = data[:][i][10]
		X[i] = extract_features(X[:][i])
		# del X[:][i][9]
		# print(X[:][i])
	countPos = 0
	count = 0
	for i in range(len(Y)):
		pred = clf.predict(np.asarray(X[:][i]).reshape(1,-1))
		expected = Y[i]
		if(pred == expected):
			countPos = countPos + 1
		# else:
		# 	print('Wrongly predicted')
		# 	print(data[:][i])
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

def extract_features(cards):
	if not EN_FEATURE_EXTRACT:
		return cards
	features = []
	suits = []
	card_nos = []
	for i in range(5):
		suits.append(cards[i*2])
		card_nos.append(cards[i*2+1])
	#print(cards)
	#print(suits)
	suits = sorted(suits)
	card_nos = sorted(card_nos)
	for i in range(4):
		features.append(card_nos[i+1] - card_nos[i])
	features.append(card_nos[4])
	for i in range(4):
		features.append(suits[i+1] - suits[i])
	return features



def train_dec_tree():
	data = [[0 for x in range(11)] for y in range(DATA_SIZE)]
	test_data = [[0 for x in range(11)] for y in range(TRAIN_SIZE)]
	i = 0
	with open('train.csv', 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		for row in reader:
			if(i >= DATA_SIZE):
				for j in range(0, 11):
					test_data[:][i-DATA_SIZE][j] = int(row[j])
			else:
				for j in range(0, 11):
					data[:][i][j] = int(row[j])
			i = i + 1

	train_data, val_data = train_test_split(data, test_size = 0.2, random_state=0)

	train_X = [[0 for x in range(10)] for y in range(len(train_data[:]))]
	for i in range(len(train_data[:])):
		train_X[i] = train_data[:][i][:-1]
		# print('************')
		# print(train_X[:][i])
		features = extract_features(train_X[:][i])
		train_X[i] = features
		# #del train_X[:][i][9]
		# print(train_X[i])
		# print(features)
		# print('************')

	train_Y = [0 for x in range(len(train_data[:]))]
	for i in range(len(train_data[:])):
		train_Y[i] = train_data[:][i][10]


	val_X = [[0 for x in range(10)] for y in range(len(val_data[:]))]
	for i in range(len(val_data[:])):
		val_X[i] = val_data[:][i][:-1]
		val_X[:][i] = extract_features(val_X[:][i])
		#del val_X[:][i][9]

	val_Y = [0 for x in range(len(val_data[:]))]
	for i in range(len(val_data[:])):
		val_Y[i] = val_data[:][i][10]


	test_X = [[0 for x in range(10)] for y in range(len(test_data[:]))]
	for i in range(len(test_data[:])):
		test_X[i] = test_data[:][i][:-1]
		test_X[:][i] = extract_features(test_X[:][i])
		#del val_X[:][i][9]

	test_Y = [0 for x in range(len(test_data[:]))]
	for i in range(len(test_data[:])):
		test_Y[i] = test_data[:][i][10]



	f_train = open('train_error.csv','w')
	f_val = open('val_error.csv','w')
	f_test = open('test_error.csv','w')
	for max_depth in range(1,50,1):
		clf = perform_training(max_depth,train_X,train_Y)
		print('max_depth : '+str(max_depth)+', validation score : '+str(calc_Score(clf,val_data)))
		print('max_depth : '+str(max_depth)+', training score : '+str(calc_Score(clf,train_data)))
		print('max_depth : '+str(max_depth)+', testing score : '+str(calc_Score(clf,test_data)))
		f_train.write(str(max_depth)+','+str(1-calc_Score(clf,train_data))+'\n')
		f_val.write(str(max_depth)+','+str(1-calc_Score(clf,val_data))+'\n')
		f_test.write(str(max_depth)+','+str(1-calc_Score(clf,test_data))+'\n')
	f_train.close()
	f_val.close()
	f_test.close()

	return clf