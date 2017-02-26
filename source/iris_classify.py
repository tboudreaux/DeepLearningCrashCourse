import keras
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from pandas.tools.plotting import scatter_matrix
from sklearn.cross_validation import train_test_split
from keras.utils import np_utils

def one_hot_encode_object_array(arr):
    uniques, ids = np.unique(arr, return_inverse=True)
    return np_utils.to_categorical(ids, len(uniques))

df = pd.read_csv('iris.data', names=['SL', 'SW', 'PL', 'PW', 'Class'])

# uncomment if you want to see the iris dataset uncomment this
# plt.figure(figsize=(10, 7))
# scatter_matrix(iris)
# plt.show()

data_set = df.values
X = data_set[:, 0:4].astype(float)
y = data_set[:, 4]

train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.5, random_state=2)

train_y_ohe = one_hot_encode_object_array(train_y)
test_y_ohe = one_hot_encode_object_array(test_y)

model = Sequential()
model.add(Dense(16, input_shape=(4,)))
model.add(Activation('sigmoid'))
model.add(Dense(3))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(train_X, train_y_ohe, verbose=0, batch_size=1)
predict = model.predict(test_X)

score = model.evaluate(test_X, test_y_ohe, verbose=0)
print ('Test Score: {}'.format(score))

# Uncomment if you want to see the predictions
# c = [x.argmax() for x in predict]
# data = {'SL': test_X.T[0], 'SW': test_X.T[1], 'PL': test_X.T[2], 'PW': test_X.T[2], 'CLASS': c}
# df = pd.DataFrame(data=data)
# sns.pairplot(df, hue='CLASS')
# plt.show()

