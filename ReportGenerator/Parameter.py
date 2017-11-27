# ReportGenerator is developed in Spyderpy(2.7)

# for global system
cell_no = 6  
user_no = 20000
map_size = [104, 26]    # Logical coordinate system : bottom left is (0, 0),  and top rigth is(1000, 1000)
fix_cell_pos = True 
closest_N_cell_no = 6       #there are some BS information(x,y,power) in MDT. This parameter determine the No of BSs shown in MDT.  N means the first N closest small cells to user.


# for cell
Pmax = 19      
Pmin = -30 
max_client_no = 32
radius = 32.
ref_dist = 1.


# path loss = a * log10(d) + b + c * log10(fc) = 16.9 * log10(d) + 32.8 + 20 * log10(fc)
a = 16.9
b = 32.8
c = 20  
fc = 2300 


#external data
header = ['id','user_x','user_y'] + ['cell_x', 'cell_y', 'cell_power']*closest_N_cell_no + ['serving Rx','neighbor Rx']
path = './data'
fn_pos_random = path + '/pos_rd.pkl'    # [ [(cell1_x, cell1_y, cell1_power), (cell2_x, cell2_y, cell2_power)...], [(user1_x, user1_y), (user2_x, user2_y)...] ]
fn_pos_fixed = path + '/pos_fix.pkl'    # smallcell deployment of ITRI building 51 :  [[(24.377, 11.978, 19), (51.446, 17.966, 19), (69.175, 11.247, 19), (77.565, 5.4045, 19), (82.947, 18.112, 19), (101.31, 19.135, 19)], []]
fn_output = path + '/MDT.csv'


#--------------------------------------------------------------------------------
"""
# for global system
cell_no = 20 
user_no = 100              
map_size = [200, 200]        # Logical coordinate system : bottom left is (0, 0),  and top rigth is(1000, 1000)
alpha = 1.                  # no use
gamma = 4.                  # for path loss model
bandwidth = 20.              
snr_threshold = 0.7       
gaussian_noise = 8.        # (db)  no use
delta = 0.               # delta=1 means objective function only consider coverage factor; delta=0 means objective function only consider capacity factor
capacity4G = 100.        # standard 4G : 100Mbps
isInterfere = True       # decide whether serving cell is interfered by other cells
interfereReduction = 1.   # no use


# for cell
fix_cell_pos = False 
fix_user_pos = False  
Pmax = 19.       
Pmin = -31.      
sector_no = 5     
max_client_no = 32
radius = 32.
ref_dist = 1.
Pref = 30.                # PL(d1) = 30


#external data
path = './data'
fn_pos_random = path + '/pos.pkl'
fn_pos_bs = path + '/161130-BS.txt'
fn_pos_ue = path + '/161130-UE.txt'
fn_result_bs_no = path + '/result_bs_no.pkl'                # this path is used to store the result of vtc-fall simulation
fn_result_user_no = path + '/result_user_no.pkl'
"""


    
    
    
    
    