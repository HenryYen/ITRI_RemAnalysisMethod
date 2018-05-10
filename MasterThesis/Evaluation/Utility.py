import numpy as np
import pandas as pd
import random as rd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.model_selection import cross_validate
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense, Activation 
from keras.optimizers import Adam



path = './data'
fn_heatmap_pixel = path + '/heatmap_pixel_1m.csv'
map_size = [104, 26]
nb_fold = 10


def load_pixel_data(nb_feature):
    dataset = pd.read_csv(fn_heatmap_pixel).values
    pixel = dataset[:, :nb_feature]     
    return pixel


def draw_heatmap(model, nb_feature):     # specialized drawing function for ITRI building 51
    plt.figure(rd.randint(0, 100000))
    x_resolution = map_size[0]
    y_resolution = map_size[1]
    xaxis = np.linspace(0., x_resolution, x_resolution+1)   # why +1? there are 105 point between 0~104
    yaxis = np.linspace(0., y_resolution, y_resolution+1) 
    x, y = np.meshgrid(xaxis, yaxis)
    pixel_pos = load_pixel_data(nb_feature)
    z = model.predict(pixel_pos)
    z = np.reshape(z, (y_resolution+1, x_resolution+1)) 
    """ 
    pos = np.column_stack((np.reshape(x, (x_resolution+1)*(y_resolution+1)), np.reshape(y, (x_resolution+1)*(y_resolution+1))))  # [[x1, y1], [x2, y2]...]   
    print pos[:10]
    z = model.predict(pos)
    z = np.reshape(z, ((y_resolution+1), (x_resolution+1)))   
    """
    plt.contourf(x, y, z, 500, cmap='jet')                             
    plt.colorbar() 
    plt.savefig('heapmap', dpi=200)
    plt.show()


def draw_bitmap(X_train):          # draw the location distribution of user reports
    """   make background grid
    fig, ax = plt.subplots()
    plt.axis([0, map_size[0], 0, map_size[1]])
    minorLocator = MultipleLocator(1)   #spacing of grid : 1 unit
    ax.yaxis.set_minor_locator(minorLocator)
    ax.xaxis.set_minor_locator(minorLocator)
    """
    plt.figure(1)
    for i in range(len(X_train)):        
        plt.plot(round(X_train[i][0])+0.5, round(X_train[i][1])+0.5, color='g', marker='2')  #shift x, y by 0.5 unit
    #plt.grid(which = 'minor', color='black', linestyle='-', linewidth=0.5)
    img = plt.imread("./pic/51_5F.jpg")
    plt.imshow(img, zorder=0, extent=[0, map_size[0], 0, map_size[1]])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Location of user report')
    plt.savefig('bitmap.png', dpi=200)
    plt.show()
    

def draw_importance_forest(model, feature_no):   #draw feature importance of forest
    print model.feature_importances_ 
    importances = model.feature_importances_   
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(feature_no), importances[indices], color="r", yerr=std[indices], align="center")
    plt.xticks(range(feature_no), indices)
    plt.xlim([-1, feature_no])
    plt.savefig('importance.png', dpi=200)
    plt.show()

def get_10cv_score(model, X_train, y_train):
    scoring = {'mse':'neg_mean_squared_error', 'r2':'r2'}
    scores = cross_validate(model, X_train, y_train, cv=nb_fold, scoring=scoring)
    print('[MSE]: %.3f' % -scores['test_mse'].mean())
    print('[R2] : %.3f' % scores['test_r2'].mean()) 

def cross_val_RF(X_train, y_train):
    n_estimators = 100
    print "+++Random Forest consisting of " + str(n_estimators) + " trees"
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=30, random_state=2)
    get_10cv_score(model, X_train, y_train)

def cross_val_Lin(X_train, y_train):
    print "+++Linear regression"
    model = linear_model.LinearRegression()    
    get_10cv_score(model, X_train, y_train)

def cross_val_NN(X_train, y_train, nb_epoch):
    print "+++Shallow NN"
    def create_net():
        model = Sequential()  
        model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))  
        model.add(Dense(128, activation='relu'))    
        model.add(Dense(1))     
        model.compile(loss='mean_squared_error', optimizer='adam')    
        return model
    model = KerasRegressor(build_fn=create_net, epochs=nb_epoch, batch_size=16, shuffle=True, verbose=0)
    get_10cv_score(model, X_train, y_train)







