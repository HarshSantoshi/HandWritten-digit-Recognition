import numpy as np 
import keras
from keras.datasets import mnist

#BY HARSH SANTOSHI (2k21/SE/82) AND HARSH(2K21/SE/77)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train =x_train.astype(np.float32)/255
x_test=x_test.astype(np.float32)/255

x_train = np.expand_dims(x_train,-1)
x_test = np.expand_dims(x_test,-1)

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

MY_model = keras.models.load_model("MyModel.h5")

my_accuracy = MY_model.evaluate(x_test,y_test)

print(f"The accuracy is {my_accuracy[1]*100}")