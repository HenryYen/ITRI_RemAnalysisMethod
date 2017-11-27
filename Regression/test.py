import numpy as np
import pandas as pd
import pickle
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


path = './data'
fn_train_data = path + '/MDT.csv'
#fn_test_data = path + '/test.in'
#fn_output = 'out.csv'
fn_model = 'model'



def load_train_data():
    dataset = pd.read_csv(fn_train_data).values
    X_train = dataset[:, 1:3]   
    y_train = dataset[:, 4]    
    return (X_train, y_train)


if __name__ == '__main__':
    model = load_model(fn_model)
    """
    (X_train, y_train) = load_train_data()
    y_predict = model.predict(X_train)
    r2 = r2_score( y_train, y_predict )
    rmse = mean_squared_error( y_train, y_predict )
    print "[Accuracy] R2 : %f" % r2 
    print "[Accuracy] RMSE : %f" % rmse
    """

    
    while True:
        x, y = input("X,Y : ")
        print 'Predicted Rx power =', model.predict(np.array([[x, y]]))
        print
    

