import collections;
import matlibplot.pyplot as plt;


alphab = ['Normal find', 'Metastases', 'Malign lymph','Fibrosis']
frequencies = c.values()

pos = np.arange(len(alphab))
width = 1.0     # gives histogram aspect to the bar diagram

ax = plt.axes()
ax.set_xticks(pos + (width / 2))
ax.set_xticklabels(alphab)

plt.bar(pos, frequencies, width, color='r')
