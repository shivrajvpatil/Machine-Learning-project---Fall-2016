from keras.models import model_from_json
import numpy
import pandas as pd
import tensorflow as tf
import csv

DATA_SIZE = 1000000
EN_FEATURE_EXTRACT = True

def extract_features(cards):
	# print(cards)
	if not EN_FEATURE_EXTRACT:
		return cards
	features = []
	suits = []
	card_nos = []
	for i in range(5):
		# print(i)
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


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


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
X = [0 for x in range(9)]
print('Predicting results')
for i in range(DATA_SIZE):
	if(i % 10000 == 0):
		print(i)
	data[i] = extract_features(data[:][i])
	for j in range(9):
		X[j] = data[:][i][j].tolist()
	print(X)
	pred_list.append(loaded_model.predict_classes(numpy.array(X).reshape(1,-1), batch_size=2))

print('Writing results')
i = 1
with open('res_deeplearn.csv', 'w') as csvfile:
	writer = csv.writer(csvfile)
	for pred in pred_list:
		writer.writerow(str(i))
		writer.writerow(str(pred[0]))
		i = i + 1