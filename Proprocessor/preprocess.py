import numpy as np
import pandas as pd


path = './'
fn_data = path + '/MDT_AUE_10percent.csv'
map_size = [104, 26]
mask_size = 3 


def load_data():
    df = pd.read_csv(fn_data).values  
    return df


def pos_to_index(x, y):
    i, j = int(x/mask_size),  int(y/mask_size)
    return (i, j)




if __name__ == '__main__':
    M_size = map(lambda x:x+1, pos_to_index(map_size[0], map_size[1]))
    M = [[[] for _ in range(M_size[1])] for _ in range(M_size[0])]      
    """
    Take left-bottom as base point. 
    Whenever going right, i increases.
    And whenever going up, j increase.
    """
    
    dataset = load_data()
    for report in dataset:
        x, y = report[1], report[2]
        (i, j) = pos_to_index(x, y)
        M[i][j].append(report)
    
    acc = [0]*11    
    for col in M:
        for row in col:
            received = [report[-2] for report in row]
            mean, std = np.mean(received), np.std(received)
            AUE = [report[0] for report in row if abs(report[-2]-mean)>std*2]
            print AUE
            
            acc[10] += len(row)
            for u in AUE:
                acc[int(u)] += 1                
    print acc
            
    
    


