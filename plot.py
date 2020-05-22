import pandas as pd
import matplotlib as plt

df = pd.read_csv('res.csv')
xs=df['grid_dim']
ys=df['time']
plt.plot(xs,ys,"-o")
