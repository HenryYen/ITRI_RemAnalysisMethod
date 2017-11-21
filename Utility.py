import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


map_size = [104, 26]

def draw_heatmap(model):     # specialized drawing function for ITRI building 51
    x_resolution = map_size[0]
    y_resolution = map_size[1]
    xaxis = np.linspace(0., x_resolution, x_resolution)
    yaxis = np.linspace(0., y_resolution, y_resolution)  
    x, y = np.meshgrid(xaxis, yaxis)
    pos = np.column_stack((np.reshape(x, x_resolution*y_resolution), np.reshape(y, x_resolution*y_resolution)))  # [[x1, y1], [x2, y2]...]
    z = model.predict(pos)
    z = np.reshape(z, (y_resolution, x_resolution))    
    plt.contourf(x, y, z, 500, cmap='jet')                             
    plt.colorbar() 
    plt.savefig('heapmap', dpi=500)
    plt.show()


def draw_bitmap(X_train):          # draw the location distribution of user reports
    fig, ax = plt.subplots()
    plt.axis([0, map_size[0], 0, map_size[1]])
    minorLocator = MultipleLocator(1)   #spacing of grid : 1 unit
    ax.yaxis.set_minor_locator(minorLocator)
    ax.xaxis.set_minor_locator(minorLocator)

    for i in range(len(X_train)):        
        plt.plot(round(X_train[i][0])+0.5, round(X_train[i][1])+0.5, color='b', marker='d')  #shift x, y by 0.5 unit
    plt.grid(which = 'minor', color='black', linestyle='-', linewidth=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Location of user report')
    plt.savefig('bitmap.png', dpi=500)
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
    plt.savefig('importance.png', dpi=300)
    plt.show()
