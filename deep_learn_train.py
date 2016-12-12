from keras.models import Sequential
from keras.layers import Dense
import numpy
import os

DATA_SIZE = 25010
TRAIN_SIZE = 20000
EN_FEATURE_EXTRACT = True

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

def get_neuro_arr(y):
	Y = [[0 for x in range(10)] for y in range(y.size)]
	for i in range(0,y.size):
		Y[i][y[i]] = 1
	return Y

seed = 7
numpy.random.seed(seed)
dataset = numpy.loadtxt("train.csv",dtype = int, delimiter=",")

train_X = [[0 for x in range(10)] for y in range(TRAIN_SIZE)]
test_X = [[0 for x in range(10)] for y in range(DATA_SIZE - TRAIN_SIZE)]

for i in range(TRAIN_SIZE):
	train_X[i] = extract_features(dataset[i,0:10])
k = 0
for i in range(TRAIN_SIZE,DATA_SIZE):
	test_X[k] = extract_features(dataset[i,0:10])
	k = k + 1

train_Y = get_neuro_arr(dataset[:TRAIN_SIZE,10])
test_Y = get_neuro_arr(dataset[TRAIN_SIZE:DATA_SIZE,10])

print(train_X[3])
print(train_Y[3])

model = Sequential()
model.add(Dense(9, input_dim=9, init='uniform', activation='relu'))
model.add(Dense(10, init='uniform', activation='relu'))
model.add(Dense(10, init='uniform', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_X, train_Y, nb_epoch=200, batch_size=100)

train_score = model.evaluate(train_X, train_Y)
print('Train score')
print("%s: %.2f%%" % (model.metrics_names[1], train_score[1]*100))

test_score = model.evaluate(test_X, test_Y)
print('Test score')
print("%s: %.2f%%" % (model.metrics_names[1], test_score[1]*100))

# serialize model to JSON
model_json = model.to_json()
with open("model1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")



