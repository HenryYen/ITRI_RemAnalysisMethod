import Parameter as pr
from Cell import *
from User import *
import random as rd
from math import sqrt, pi, degrees, atan2, log, pow
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pickle as pk
import numpy as np



def get_dist(x1, y1, x2, y2):
    return sqrt((x1-x2)**2 + (y1-y2)**2)


def get_pathloss(c, u):  # argument:cell, user    # (db),  path loss = a * log10(d) + b + c * log10(fc) = 16.9 * log10(d) + 32.8 + 20 * log10(fc)
        dist = get_dist(u.x, u.y, c.x, c.y)
        dist = pr.ref_dist if dist < pr.ref_dist else dist
        #fading = float(np.random.normal(loc=3.5, scale=2, size=1))
        return pr.a * log(dist, 10) + pr.b + pr.c * log(pr.fc, 10) 


def getID(type):
    if type == 'cell':
        getID.cell_id = getattr(getID, 'cell_id', -1) + 1
        return getID.cell_id            
    if type == 'user':
        getID.user_id = getattr(getID, 'user_id', -1) + 1
        return getID.user_id
        

def write_csv(data):            # parameter DATA is a collection of user reports. each report includes these string-type feature: [user id, x, y, serving id, serving rx, neighbor id, neighbor rx]
    with open(pr.fn_output, 'w'):
        with open(pr.fn_output, 'a+') as f:
            f.write(','.join(pr.header) + '\n')
            for report in data:
                f.write(','.join([str(e) for e in report]) + '\n')


def generate_scenario(MAX_CELL = 20000, MAX_USER = 100000):           # [ [(cell1_x, cell1_y, cell1_power), (cell2_x, cell2_y, cell2_power)...], [(user1_x, user1_y), (user2_x, user2_y)...] ]
    cm_pos = []
    um_pos = []
    for _ in range(MAX_CELL):
        x = rd.uniform(0, pr.map_size[0])
        y = rd.uniform(0, pr.map_size[1])
        power = int(rd.uniform(pr.Pmin, pr.Pmax))
        cm_pos.append((x, y, power))
    for _ in range(MAX_USER):
        x = rd.uniform(0, pr.map_size[0])
        y = rd.uniform(0, pr.map_size[1])
        um_pos.append((x, y))
    with open (pr.fn_pos_random, 'wb') as f:
        pk.dump([cm_pos, um_pos], f)
    print "### rd_pos.pkl is succefully created!"
    
    
def draw_system(cm, um):
    color = ['b', 'g', 'r', 'c', 'k', 'y', 'm' ,'w', '#8172B2', '#56B4E9', '#D0BBFF', '#D65F5F', '#017517', '#e5ae38', '#001C7F', '#6ACC65', '#8EBA42', '#EAEAF2', '#7600A1', '#E8000B']
    for c in cm:
        color_no = 0
        plt.text(c.x, c.y, c.ID)
        plt.plot(c.x, c.y, color = color[color_no], marker='^')
    
    for u in um:
        color_no = 1
        #if min([get_dist(u.x, u.y, c.x, c.y) for c in cm]) > 10:
        plt.plot(u.x, u.y, color = color[color_no], marker='.')  
    
    img = plt.imread("./pic/51_5F.jpg")
    plt.imshow(img, zorder=0, extent=[0, 104, 0, 26])
    plt.axis([0, pr.map_size[0], 0, pr.map_size[1]])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('System Overview')
    plt.savefig('./pic/overview.png', dpi=200)
    plt.show()


#--------------------------------------------------------
"""


def load_pos_cell():
    pos = []
    with open(pr.fn_pos_bs, 'r') as f:
        for line in f:
            parts = line.split()
            pos.append([float(e) for e in parts])
    return pos

def load_pos_user():
    pos = []
    with open(pr.fn_pos_ue, 'r') as f:
        for line in f:
            parts = line.split(',')
            pos.append([float(e) for e in parts])
    return pos

def get_snr(cell, user, cm, isInterfere, isAddGain):        # return power ratio not db
    def dbm2mwatt(dbm):
        return pow(10, dbm/10.)         
    def ratio2db(ratio):        # SNR in db
        return 10 * log(ratio, 10)                
    dist = get_dist(cell.pos_x, cell.pos_y, user.pos_x, user.pos_y)
    gain = rp.beam_gain(cell.get_beam_pattern(), get_angle(cell, user)) if isAddGain else 0
    
    signal_power = dbm2mwatt(cell.power - get_pathloss(dist) + gain)    # in watt
    interfere_power = 0. if len(cm)!=1 else 0.01                        # in watt
    for e in cm:
        if e is cell:
            continue
        dist = get_dist(e.pos_x, e.pos_y, user.pos_x, user.pos_y)
        interfere_power += dbm2mwatt(e.power - get_pathloss(dist))
    #interfere_power += dbm2mwatt(pr.gaussian_noise)    
    interfere_power = interfere_power if isInterfere else 1.    
    return  (signal_power / interfere_power) * pr.interfereReduction   
    
    
           
def print_all_status(cm, um):
    cover = print_cover_rate(um)
    power = print_cm_power(cm)
    capacity = print_capacity(cm ,um)
    #draw_system(cm, um)  
    #print_cm_client(cm)
    #print_power_reduce(cm)
    #print_interfere_reduce(cm, um)  
    return (cover, power, capacity)
    
    

    
    
def get_cover_nb(um):
    covered = 0.
    for u in um:
        if u.master != None:
            covered += 1.
    return covered   
    
    
def print_capacity(cm, um):
    capacity = 0.
    for c in cm:
        nb_client = c.get_client_no()
        if nb_client == 0:
            continue
        Bk = pr.bandwidth / nb_client 
        capacity += sum([Bk * log(1 + get_snr(c, u, cm, pr.isInterfere, False), 2) for sec in c.client for u in sec])
    avg_capacity = capacity / get_cover_nb(um) if get_cover_nb(um) != 0 else 0
    #print('[Average user capacity] :', avg_capacity)
    return avg_capacity
    
    
def print_cover_rate(um):
    covered = get_cover_nb(um)
    #print ('[Cover rate] :%.3f%%  (%d/%d)' % (covered/pr.user_no*100, covered, pr.user_no))
    return covered/pr.user_no


def print_cm_power(cm):
    avg_power = sum([e.power for e in cm])/len(cm)
    #print('[Average cell power] :', avg_power)
    #print('[Cell power] :' , [e.power for e in cm])
    return avg_power
        
def print_cm_client(cm):
    print('[Cell client]:')
    for c in cm:
        print(' ', c.get_client_no(), '/', pr.max_client_no, [len(e) for e in c.client])

        
def print_power_reduce(cm):
    nb_sector = pr.cell_no * pr.sector_no
    opened = sum([int(len(sec) > 0) for c in cm for sec in c.client ])
    print ('[Power saving] : from %d sectors opened to %d sectors' % (nb_sector, opened))
    #print ('[Power saving] :%f  (%d/%d)' % (opened/nb_sector*100, opened, nb_sector))
    
def print_interfere_reduce(cm, um):
    covered = 0.;
    intersect = 0.
    
    tmp_p = [c.power for c in cm]
    for c in cm:
        c.power = pr.Pmax
    for u in um:
        counter = 0.
        for c in cm:
            snr = get_snr(c, u, cm, pr.isInterfere, False)
            dist = get_dist(c.pos_x, c.pos_y, u.pos_x, u.pos_y)
            if snr >= pr.snr_threshold and dist <= c.radius:  
                counter += 1            # if counter >= 2, means this user is in the intersection of two cell's coverage
        if counter >= 2:
            intersect += 1        
            if u.master != None:
                covered += 1
    for c in cm:
        c.power = tmp_p[cm.index(c)]
    rate = covered/intersect if intersect != 0 else 0
    print ('[Interference reduction] :%.3f%%  (%d/%d)' % (rate * 100, covered, intersect))
    

def scenario1():
    cm = [sc.Cell(500, 500), sc.Cell(480, 500), sc.Cell(450, 500)]
    um = [ur.User(i*5, 500) for i in range(80, 101)]
    
    cell = cm[0]
    import Simulate as si
    um[0].master = cell
    cell.client[0].append(um[0])
    print (si.objective_func(cell, cm, um))
    for i in range(-31, 20):
        cell.power = i
        print (si.objective_func(cell, cm, um))         
"""                
        
    
    
    
    
    
    
    
    
    
    
    
    
