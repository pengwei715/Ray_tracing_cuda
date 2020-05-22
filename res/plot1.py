import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('res1.csv')
xs=df['size']
gys=df['GPU_time']
cys=df['CPU_time']
plt.plot(xs,gys,"-o")
plt.plot(xs,cys,"-o")
plt.title("Different problem size performance between CPU and GPU")
plt.xlabel("problem size")
plt.ylabel("time")
plt.savefig("cvsg.png")
