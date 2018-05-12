import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random

def k_NN(data, predict, k=3):
	if len(data) >= k:
		warnings.warn("K is set to a value less than total voting groups !")

	distances = []
	[[ distances.append([np.linalg.norm(np.array(features) - np.array(predict)), group]) for features in data[group]] for group in data]

	votes = [i[1] for i in sorted(distances)[:k]]
	vote_result = Counter(votes).most_common(1)[0][0]
	confidence = Counter(votes).most_common(1)[0][1] / k

	return vote_result, confidence

df = pd.read_csv("breast-cancer-wisconsin.data.txt")
df.replace('?', -99999, inplace=True)
df.drop(df.columns[[0]], 1, inplace=True)
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

test_size = 0.2
train_set = {2: [], 4: []}
test_set = {2: [], 4: []}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

[ train_set[i[-1]].append(i[:-1]) for i in train_data ]
[ test_set[i[-1]].append(i[:-1]) for i in test_data ]

correct = 0
total = 0
confidences = []

for group in test_set:
	for data in test_set[group]:
		vote, confidence = k_NN(train_set, data, k=3)
		if group == vote:
			correct += 1
		total += 1
		confidences.append(confidence)

print('Accuracy:', correct/total, 'Average confidence', (sum(confidences)/len(confidences)))