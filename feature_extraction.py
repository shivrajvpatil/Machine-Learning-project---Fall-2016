def extract_features(cards):
	features = [0 for x in range(9)]
	suits = []
	card_nos = []
	for i in range(5):
		suits.append(cards[i*2])
		card_nos.append(cards[i*2+1])
	suits = sorted(suits)
	card_nos = sorted(card_nos)
	print(suits)
	print(card_nos)
	for i in range(4):
		features[i] = card_nos[i+1] - card_nos[i]
	features[4] = card_nos[4]
	for i in range(4):
		features[5+i] = suits[i+1] - suits[i]
	return features

cards = []
cards.append(4)
cards.append(9)
cards.append(2)
cards.append(1)
cards.append(2)
cards.append(2)
cards.append(4)
cards.append(7)
cards.append(2)
cards.append(8)
features  = extract_features(cards)
print(features)
