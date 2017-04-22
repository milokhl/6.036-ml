import numpy as np
np.random.seed(12321)  # for reproducibility
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D 
from keras.layers.pooling import MaxPooling2D 
from keras.optimizers import Adagrad
import h5py
from keras import backend as K
import utils_multiMNIST as U
import time
path_to_data_dir = '../Datasets/'
use_mini_dataset = True

nb_classes = 10
num_classes = 10
img_rows, img_cols = 42, 28
nb_epoch = 3
batch_size = 64
K.set_image_dim_ordering('th')

def main():
    X_train, y_train, X_test, y_test = U.get_data(path_to_data_dir, use_mini_dataset)

    # input layer
    inputs = Input(shape=(1, 42, 28), name='image_input')

    # intermediate layers
    conv_layer1 = Conv2D(8, (3,3), activation='relu', input_shape=(1, X_train.shape[2], X_train.shape[3]))(inputs)
    maxpooling1 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(conv_layer1)
    conv_layer2 = Conv2D(16, (3,3), activation='relu', input_shape=(1, X_train.shape[2], X_train.shape[3]))(maxpooling1)
    maxpooling2 = MaxPooling2D(pool_size=(2,2), strides=(1,1))(conv_layer1)
    flatten = Flatten()(maxpooling2)
    hidden1 = Dense(64, activation='relu')(flatten)
    dropout = Dropout(0.5)(hidden1)

    # create 2 outputs!
    prediction1 = Dense(10, activation='softmax', name="first_prediction")(dropout)
    prediction2 = Dense(10, activation='softmax', name="second_prediction")(dropout)

    model = Model(inputs=inputs, outputs=[prediction1, prediction2])
    model.compile(loss='categorical_crossentropy', optimizer=Adagrad(lr=0.01, epsilon=1e-08, decay=0.0),  \
                  metrics=['accuracy'], loss_weights=[0.5, 0.5])
    #==================== Fetch Data and Fit ===================#
    start = time.time()
    model.fit(X_train, [y_train[0], y_train[1]], nb_epoch=nb_epoch, batch_size=batch_size, verbose=1)
    end = time.time()
    print("Training time:", end-start)

    objective_score = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1, sample_weight=None)
    print('Evaluation on test set:', dict(zip(model.metrics_names, objective_score)))
    
    #Uncomment the following line if you would like to save your trained model
    #model.save('./current_model_conv.h5')
    if K.backend()== 'tensorflow':
        K.clear_session()

if __name__ == '__main__':
    main()
