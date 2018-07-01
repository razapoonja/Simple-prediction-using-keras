import itertools
from random import randint
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.models import load_model


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




# Preprocessing data

train_labels = []
train_samples = []

# Generating fake random data

for i in range(50):
	# 5% younger individuals who experience side effects 
	random_younger = randint(13, 64)
	train_samples.append(random_younger)
	train_labels.append(1)

	# 5% older individuals who did not experience side effects 
	random_older = randint(65, 100)
	train_samples.append(random_older)
	train_labels.append(0)


for i in range(1000):
	# 95% younger individuals who did not experience side effects 
	random_younger = randint(13, 64)
	train_samples.append(random_younger)
	train_labels.append(0)

	# 95% older individuals who experience side effects 
	random_older = randint(65, 100)
	train_samples.append(random_older)
	train_labels.append(1)

# Making our fake data as numpy array

train_labels = np.array(train_labels)
train_samples = np.array(train_samples)

# Scaling our train_sample data from 0 to 1

scalar = MinMaxScaler(feature_range=(0, 1))
scaled_train_samples = scalar.fit_transform((train_samples).reshape(-1, 1))





# Creating neural network

# Using sequential model because every layer has input and output. 
model = Sequential()
# First Layer (number of neurons, dimention of array, activation_function)
model.add(Dense(16, input_shape=(1,), activation='relu'))
# Hidden Layer
model.add(Dense(32, activation='relu'))
# Output Layer (number of neurons, activation_function)
# we have 2 output option 0 or 1 thats why two neurons
model.add(Dense(2, activation='softmax'))





# Taining neural network

# (optimizing function(learning rate), loss function, metrics)
model.compile(Adam(lr=.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Validation 10% to avoid overfitting
model.fit(scaled_train_samples, train_labels, validation_split=0.1, batch_size=10, epochs=3, shuffle=True, verbose=2)





# Prdiction

# Generating test data to predict
test_labels = []
test_samples = []

for i in range(10):
	random_younger = randint(13, 64)
	test_samples.append(random_younger)
	test_labels.append(1)

	random_older = randint(65, 100)
	test_samples.append(random_older)
	test_labels.append(0)


for i in range(200):
	random_younger = randint(13, 64)
	test_samples.append(random_younger)
	test_labels.append(0)

	random_older = randint(65, 100)
	test_samples.append(random_older)
	test_labels.append(1)

test_labels = np.array(test_labels)
test_samples = np.array(test_samples)

scalar = MinMaxScaler(feature_range=(0, 1))
scaled_test_samples = scalar.fit_transform((test_samples).reshape(-1, 1))

# Predicting
# From this we'll get [0, 1] probabilities
# predictions = model.predict(scaled_test_samples, batch_size=10, verbose=0)

# From this we'll turn probabilities in rounded form from which we'll get 0 or 1
rounded_predictions = model.predict_classes(scaled_test_samples, batch_size=10, verbose=0)





# Confusion matrix


cm = confusion_matrix(test_labels, rounded_predictions)
cm_plot_labels = ['no_side_effects', 'had_side_effects']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')

plt.show()





# Deploying

model.save('medical_dummy_model.h5')

new_model = load_model('medical_dummy_model.h5')
new_model.summary()
new_model.get_weights()
new_model.optimizer
