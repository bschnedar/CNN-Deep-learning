import pickle
from keras.engine.saving import load_model
from numpy import mean
from sklearn.model_selection import KFold
import sys

model = sys.argv[1]

#model = load_model('model.h5')

with open('train.pickle', 'rb') as f:
    trainX, trainY = pickle.load(f)

dataX = trainX
dataY = trainY

scores, histories = list(), list()
runs = 2 # the number of times the networks runs/learns must be bigger than 2

kfold = KFold(runs, shuffle=True, random_state=1)
for train_ix, test_ix in kfold.split(dataX):
	trainX = dataX[train_ix]
	trainY = dataY[train_ix]
	testX = dataX[test_ix]
	testY =  dataY[test_ix]
	#Learning from history
	history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
	_, acc = model.evaluate(testX, testY, verbose=0)

	print('> %.3f' % (acc * 100.0))
	# stores scores
	scores.append(acc)
	#histories.append(history)

print('Accuracy: %.3f ' % (mean(scores)*100))







