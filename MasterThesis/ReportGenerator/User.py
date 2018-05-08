from Utility import get_pathloss

class User:
    def __init__(self, ID, x, y):
        self.ID = ID
        self.x = x
        self.y = y
        self.snr = 0.
        self.master = None              # put the reference of master
      
    def __str__(self):
        return '[User %d:  pos(%d, %d)]' % (self.ID, self.x, self.y)
        
    def get_rxpower(self, cell):
            PL = get_pathloss(cell, self)
            return cell.power - PL
        