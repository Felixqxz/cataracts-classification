from keras.datasets import mnist
from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt
from keras.optimizers import sgd, RMSprop, Adam
from keras.utils import to_categorical
from keras.layers import Conv2D, Flatten, MaxPool2D
import matplotlib.image as mpimg

# %% load the dataset
(x_train, y_train),(x_test, y_test) = mnist.load_data()
type(x_train)

print(x_train.shape, y_train.shape)
x_train_0 = x_train[0]
y_train_0 = y_train[0]


plt.imshow(x_train_0)
plt.title('Training Label ()'.format(y_train_0))
plt.show()

y_train.min()
y_train.max()

plt.hist(y_train, bins = 10)
plt.show()

nb_imgs = 1000
x_train_small = x_train[:nb_imgs]
y_test_prob = to_categorical(y_test)
y_train_small = y_train[:nb_imgs]

x_train_small.shape


nb_neurons = 64

# Define Multi-layer Perceptron
inputs = Input(shape=(784,))
h1 = Dense(nb_neurons, activation='relu')(inputs)
h2 = Dense(nb_neurons, activation='relu')(h1)
probabilities = Dense(10, activation='softmax')(h2)


model = Model(inputs=inputs, outputs=probabilities)
model.compile(optimizer=RMSprop(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


y_train_small_prob = to_categorical(y_train_small)
y_train_small_prob.shape
print(y_train_small_prob)
y_test_prob = to_categorical(y_test)

nb_epochs = 10
model.fit(x_train_small.reshape(nb_imges,784),
          y_train_small_prob,
          epochs=nb_epochs,
          validation_data=(x_test.reshape(x_test.shape[0],784), y_test_prob))

nb_filters = 8
kernel_size = 3
# Define Convolutional Neural Work
inputs = Input(shape=(28, 28,1))

conv1 = Conv2D(nb_filters, kernel_size, activation='relu')(inputs)
conv1 = Conv2D(nb_filters, kernel_size, activation='relu')(conv1)
max1 = MaxPool2D()(conv1)

conv2 = Conv2D(2*nb_filters, kernel_size, activation='relu')(max1)
conv2 = Conv2D(2*nb_filters, kernel_size, activation='relu')(conv2)
max2 = MaxPool2D()(conv2)

flat = Flatten()(max2)

dense1 = Dense(2*nb_filters, activation='relu')(flat)
dense2 = Dense(2*nb_filters, activation='relu')(dense1)

probabilities = Dense(2*nb_filters, activation='softmax')(dense2)

model = Model(inputs=inputs, outputs=probabilities)
model.compile(optimizer=RMSprop(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
nb_epochs = 10
model.fit(x_train_small.reshape((*x_train_small.shape,1)), y_train_small_prob, epochs=nb_epochs,
validation_data = (x_test.reshape(*x_test.shape,1), y_test_prob))
