import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation , Dropout
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.preprocessing import Normalizer
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
from Utility import draw_heatmap, draw_bitmap, draw_importance_forest, cross_validation
from sklearn.cross_validation import cross_val_score

    
path = './data'
fn_train_data = path + '/MDT.csv'
fn_model = 'model1'

train_data_size = 300
test_data_size = 3000
nb_epoch = 50
nb_feature = 11      # feature : x, y


def load_train_data():
    dataset = pd.read_csv(fn_train_data).values
    X_train = dataset[:train_data_size, 1:1+nb_feature]     
    y_train = dataset[:train_data_size, -2]   
    X_test = dataset[train_data_size:train_data_size+test_data_size, 1:1+nb_feature]   
    y_test = dataset[train_data_size:train_data_size+test_data_size, -2]  
    return (X_train, y_train, X_test, y_test)
    


def build_dNN_model(X_train, y_train):     
    print "+++Deep NN"
    model = Sequential()    
    model.add(Dense(128, activation='relu', input_dim = nb_feature)) 
    model.add(Dense(128, activation='relu'))    
    model.add(Dense(128, activation='relu'))    
    model.add(Dense(128, activation='relu'))    
    model.add(Dense(1))     
    model.compile(loss='mean_squared_error', optimizer='adam')    
    model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=16,  shuffle=True)
    return model
def build_sNN_model(X_train, y_train):   #shallow NN, "non-deep" NN    
    print "+++Shallow NN"
    model = Sequential()    # first input layer
    model.add(Dense(128, activation='relu', input_dim = nb_feature))  #2nd hidden layer with x neurons
    model.add(Dense(128, activation='relu'))    # 3rd hidden layer
    model.add(Dense(1))     # output layer
    model.compile(loss='mean_squared_error', optimizer='adam')    
    model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=16,  shuffle=True)
    return model


def build_DTReg_model(X_train, y_train):
    print "+++Decision Tree"
    model = DecisionTreeRegressor(max_depth=30)
    model.fit(X_train, y_train)   
    return model
def build_RFReg_model(X_train, y_train):
    n_estimators = 100
    print "+++Random Forest consisting of " + str(n_estimators) + " trees"
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=30, random_state=2)
    model.fit(X_train, y_train)   
    return model


def build_LinReg_model_sk(X_train, y_train, X_test): 
    print "+++Linear regression"
    X_train = add_constant(X_train)   
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)   
    return model
def build_LinReg_model_sm(X_train, y_train, X_test): 
    X_train = add_constant(X_train)   
    model = sm.OLS(y_train, X_train).fit()   
    print model.summary()
    return model
def build_LinReg_model_numpy(X_train, y_train, X_test): 
    X_train = add_constant(X_train)
    model = np.linalg.lstsq(X_train,y_train)[0]
    return model    
def add_constant(X):
    constant = np.array([[1]] * len(X))
    X = np.append(X, constant, axis=1)
    return X
    
    
    
if __name__ == '__main__':
    try:
        print '***load data...'
        (X_train, y_train, X_test, y_test) = load_train_data()        
        print '***begin to train...'
        
        #model = build_dNN_model(X_train, y_train)  
        #model = build_sNN_model(X_train, y_train)       
        #model.save(fn_model)             
        #model = build_DTReg_model(X_train, y_train)
        model = build_RFReg_model(X_train, y_train)
        #model = build_LinReg_model_sk(X_train, y_train, X_test)         
        #model = build_LinReg_model_sm(X_train, y_train, X_test)         
        #model = build_LinReg_model_numpy(X_train, y_train, X_test)  
        #X_test = add_constant(X_test)   
        #y_pred = np.dot(X_test, model)      # only for numpy linear reg
        
        
        y_pred = model.predict(X_test)
        print "---Model 1--- %s features" %nb_feature
        print("[MSE]: %.3f" % mean_squared_error(y_test, y_pred))
        print('[R2]: %.3f' % r2_score(y_test, y_pred))         
        #print('[ExplainVariance]: %.3f' % explained_variance_score(y_test, y_pred))
        #draw_importance_forest(model, nb_feature)
        #draw_heatmap(model, nb_feature)
        #draw_bitmap(X_train)        
        #cross_validation(X_train, y_train)   
    
    except KeyboardInterrupt:           
        model.save(fn_model)




