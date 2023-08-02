#Brandon Schnedar
#25821724
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
import pickle
training_size = 20000






# load dataset
print("Loading MNIST dataset")
(trainX, trainY), (testX, testY) = mnist.load_data()
trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))

trainY = to_categorical(trainY)
testY = to_categorical(testY)
#slice data set
trainX = trainX[:training_size,:]
testX = testX[:training_size,:]
trainY = trainY[:training_size,:]
testY = testY[:training_size,:]

print("Converting MNIST dataset")
#convert to readable images
train_norm = trainX.astype('float32')
test_norm = testX.astype('float32')
# rgb color scheme conversion
trainX = train_norm / 255.0
testX = test_norm / 255.0
print("Generating Model")
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))
opt = SGD(lr=0.01, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
#Save model/usable Json fuke
model.save('model.h5')
#model_json = model.to_json()
#with open("model.json", "w") as json_file:
#	json_file.write(model_json)
print("Model Creation Completed, model.h5 file ready for evaluation")


with open('train.pickle', 'wb') as f:
    pickle.dump([trainX, trainY], f)

#model.h5 is the output, it is hardcoded