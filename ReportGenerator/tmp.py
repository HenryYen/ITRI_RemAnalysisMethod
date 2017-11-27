import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


fig, ax = plt.subplots()
plt.axis([0, 104, 0, 26])
minorLocator = MultipleLocator(1)
ax.yaxis.set_minor_locator(minorLocator)
ax.xaxis.set_minor_locator(minorLocator)

plt.plot(1.5, 1.5, color='b', marker='d')  
plt.grid(which = 'minor', color='black', linestyle='-', linewidth=0.5, markersize=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Location of user report')
plt.savefig('bitmap.jpg', dpi=500)
plt.show()