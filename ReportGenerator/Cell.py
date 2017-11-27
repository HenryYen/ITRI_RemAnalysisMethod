import Parameter as pr

class Cell:
    def __init__(self, ID, x, y, power=pr.Pmax):
        self.ID = ID
        self.x = x
        self.y = y
        self.power = min(max(power, pr.Pmin), pr.Pmax)                          # defaultly, open transmission power to maximum     
        self.max_client_no = pr.max_client_no
        self.radius = pr.radius
        self.client = []         
        
    def __str__(self):
        return '[Cell %d:  pos(%d, %d)  power(%d)   client(%d)]' % (self.ID, self.x, self.y, self.power, self.get_client_no())
        
    def power_up(self, n=1):
        self.power = min(max(self.power+n, pr.Pmin), pr.Pmax)
        
    def power_down(self, n=1):
        self.power = min(max(self.power-n, pr.Pmin), pr.Pmax)

    def is_power_min(self):
        return self.power == pr.Pmin
        
    def is_power_max(self):
        return self.power == pr.Pmax
        
    def add_client(self, user):
        if self.get_client_no < self.max_client_no:
            self.client.append(user)
        
    def get_client_no(self):
        return len(self.client)
        
        
if __name__ == '__main__':    
    sc1 = Cell(5, 6, 19)
    print sc1
    sc1.power_up()
    print sc1
    
    
    