import numpy as np 
import tensorflow as tf 
import keras 
import sklearn 
import matplotlib.pyplot as plt 
import itertools

def main1(inputNumber): 
    nb_classes = 10
    nb_epoch = 30

    from tensorflow.keras.models import Model 
    from tensorflow.keras.layers import Input, Conv2D, Add, MaxPooling2D, BatchNormalization, Activation, Dropout, Flatten, Dense

    def resblock(input_layer, filters, kernel_size): 
        input_layer = Conv2D(filters, kernel_size, padding='same')(input_layer)
        input_layer = BatchNormalization()(input_layer) 
        input_layer = Activation('relu')(input_layer) 

        layer = Conv2D(filters, kernel_size, padding ='same')(input_layer) 
        layer = BatchNormalization()(layer) 
        layer = Activation('relu')(layer) 

        out = Add()([input_layer, layer])
        out = Activation('relu')(out) 
        out = MaxPooling2D(pool_size=(2,2))(out) 
        out = BatchNormalization()(out) 
        out = Dropout(0.3)(out) 
        return out 
        
    # residual blocks 
    input_layer = Input(shape=(28, 28, 1))  # change to (129, 189, 1)
    block1 = resblock(input_layer, 32, (3,3)) 
    block2 = resblock(block1, 64, (3, 3)) 
    block3 = resblock(block2, 32, (3, 3)) 

    cov = Conv2D(32, (3, 3))(block3)

    # cov = MaxPooling2D(pool_size=(2,2))(block2) # added
    # cov = MaxPooling2D(pool_size=(2,2))(cov) # added

    cov = Flatten()(cov) 
    cov = Dense(32, activation='relu')(cov)
    cov = Dropout(0.3)(cov) 

    c0 = Dense(nb_classes, activation='softmax')(cov) 

    model = Model(inputs= input_layer, outputs = c0)
    model.compile(loss='categorical_crossentropy', optimizer ='adam', metrics = ['accuracy'])

    from tensorflow.keras.datasets import mnist 
    (x_train, y_train), (x_test, y_test) = mnist.load_data() 

    def remove_num(num, x, y): 
        yid = (y != num).nonzero() 
        return x[yid], y[yid] 

    x_train, y_train = remove_num(inputNumber, x_train, y_train) 
    x_train = np.expand_dims(x_train, axis = -1) 

    # one-hot encode y_train 
    from tensorflow.keras.utils import to_categorical 
    y_train = to_categorical(y_train, num_classes = nb_classes) 

    cp_path = 'models-30\\categorical\\{}\\{}'.format(inputNumber, inputNumber) 
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = cp_path, save_weights_only = True, verbose= 1)

    model.fit(
        x_train,
        y_train,
        epochs = nb_epoch, 
        validation_split = 0.25, 
        verbose = 1, 
        callbacks = [cp_callback] 
    )

    x_test = np.expand_dims(x_test, axis = -1)

    # predictions = model.predict(
    #     x_test, 
    #     steps = len(x_test) 
    # )