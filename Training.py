import numpy as np 
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense , Conv2D , MaxPool2D , Flatten , Dropout
#ploting the digit images by a function
def print_image(i):
    plt.imshow(x_train[i] ,cmap='binary')
    plt.axis("off")
    plt.title(y_train[i])
    plt.show()
#Printing first 5 images using the above function
for i in range(5):
    print_image(i) 
#Collecting the data to preprocess it
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train)
print(x_train,y_train)
print("X train "  , x_train.shape )
print("Y train ", y_train.shape)
print("X test "  , x_test.shape )
print("Y test ", y_test.shape)
#preprocess the image
#converting it to float 32 from string and normalizing it
x_train =x_train.astype(np.float32)/255
x_test=x_test.astype(np.float32)/255
x_train = np.expand_dims(x_train,-1)
x_test = np.expand_dims(x_test,-1)
print("New X train shape " , x_train.shape)
print("New X test shape " , x_test.shape)
#Preprocess the output variables
# changing to vector 
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
print("Y train " , y_train)
print("Y test " , y_test)
#preparing the model
model = Sequential()
model.add(Conv2D(filters= 32,kernel_size= (3,3) , input_shape= (28,28,1),activation='relu'))
model.add(MaxPool2D((2,2)))
#Adding one more convolutional layer
model.add(Conv2D(64,(3,3) ,activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(Flatten())
#For overfitting
model.add(Dropout(0.25))
#For classification
model.add(Dense(10,activation='softmax'))
model.compile(optimizer='adam' , loss=keras.losses.categorical_crossentropy , metrics=['accuracy'])
Es = EarlyStopping(monitor='val_acc', min_delta=0.01 , patience = 4 , verbose=1)
#Model Check Point 
Mc = ModelCheckpoint("./MyModel.h5" , monitor="val_accuracy" , verbose=1, save_best_only=True)
cb = [Es,Mc]
his = model.fit(x_train,y_train , epochs=20 , validation_split= 0.3 ,callbacks=cb)



