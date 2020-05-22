import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('res.csv')
xs=np.log(df['grid_dim'])
ys=df['time']
plt.plot(xs,ys,"-o")
plt.title("Different grid size performance")
plt.xlabel("grid_size")
plt.ylabel("time")
plt.savefig("grid.png")
