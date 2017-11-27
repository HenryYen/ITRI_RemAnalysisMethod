import Parameter as pr
from Cell import *
from User import *
from Utility import get_dist, generate_scenario, getID, get_pathloss, write_csv, draw_system
import pickle as pk
import random as rd
    
    
def init_cell():
    mylist = []
    fn_opened = pr.fn_pos_fixed if pr.fix_cell_pos else pr.fn_pos_random
    with open (fn_opened, 'rb') as f:
        pos = pk.load(f)   # include [cm, um]
        pos_cm = pos[0]
    for i in range(pr.cell_no):
        x = pos_cm[i][0]
        y = pos_cm[i][1]
        power = pos_cm[i][2]
        mylist.append( Cell(getID('cell'), x, y, power) )
    return mylist
    
    
def init_user():
    mylist = []
    with open (pr.fn_pos_random, 'rb') as f:
        pos = pk.load(f)
        pos_um = pos[1]
    for i in range(pr.user_no):            
        x = pos_um[i][0]
        y = pos_um[i][1]
        mylist.append( User(getID('user'), x, y) )
    return mylist 


def init_few_fix_evenly_user():       # Scenario : User reports are very few and reports are evenly distributed in fixed-distant interval.
    mylist = []
    for x in range(0, pr.map_size[0], 5):
        for y in range(0, pr.map_size[1], 5):            
            print x, y
            mylist.append( User(getID('user'), x, y) )
    return mylist     
    

def begin():
    cm = init_cell()
    um = init_user()
    reportset = []
       
        
    for u in um:
        #↓↓ This scope finds the first N closest small cells to user u.
        closest_N_cell = sorted(cm, key=lambda c: get_dist(u.x, u.y, c.x, c.y))[:pr.closest_N_cell_no]   # index0:closest, index1:second close, index2:third close
        #print ','.join([c.__str__() for c in closest_N_cell])
        #print ','.join([str(get_dist(u.x, u.y, c.x, c.y)) for c in closest_N_cell])
                    
                
        #↓↓ This scope aims at finding the serving/neighbor Rx power of specific user u. 
        serve_id = neighbor_id = None
        serve_rx = neighbor_rx = float('-inf')
        for c in cm:
            rx = u.get_rxpower(c)
            if rx > serve_rx:
                neighbor_id = serve_id
                neighbor_rx = serve_rx
                serve_id = c.ID
                serve_rx = rx
            elif rx > neighbor_rx:
                neighbor_id = c.ID
                neighbor_rx = rx
            #PL = get_pathloss(u, c)            
            #print '/ PathLoss between User %d & Cell %d = %d' % (um.index(u), cm.index(c), PL)
            #print '\ RSRP of User %d from Cell %d = %d' % (um.index(u), cm.index(c), u.get_rxpower(c)) 
                
        per_report = [u.ID, u.x, u.y, serve_rx, neighbor_rx]
        cell_info = []
        for c in closest_N_cell:
            cell_info.extend([c.x, c.y, c.power])
        per_report[3:3] = cell_info    # insert the position and power of deployed small cells  into per MDT   
        reportset.append(per_report)     
        #print u.ID, u.x, u.y, serve_id, serve_rx, neighbor_id, neighbor_rx
        #print ' '
    write_csv(reportset)
    #draw_system(cm, um[:500])



if __name__ == '__main__':
    #generate_scenario()
    begin()
    