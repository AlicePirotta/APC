import numpy as np
from getdist import MCSamples, plots
import matplotlib.pyplot as plt




#LOAD THE CHAIN
chain = np.load('500000_aver_dopo.npy')
print(chain.shape)




#SAMPLE
s1 = MCSamples(samples=chain, names=["Bd", "T", "Bs",  "g1", "g2", "g3", "g4", "g5", "g6", "g7", "g9", "g10", "g11", "g12", "g13", "g14", "g15", "g16", "g17", "g18", "g19", "g20", "g21", "g22" ], labels=["Bd", "T", "Bs",  "g1", "g2", "g3", "g4", "g5", "g6", "g7", "g9", "g10", "g11", "g12", "g13", "g14", "g15", "g16", "g17", "g18", "g19", "g20", "g21", "g22" ], label='21g')


#CORNER PLOT
g = plots.get_subplot_plotter()
g.triangle_plot([s1], filled=True, title_limit= True)
plt.show()

