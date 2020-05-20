import numpy as np
import matplotlib.pyplot as plt

def plot_dat(filename, shape, dtype=np.float64):
    a = np.fromfile(filename, dtype=dtype).reshape(shape);
    print(a)
    plt.imshow(a, origin='lower')
    plt.savefig(filename[:-4] + ".png")

if __name__ == '__main__':
    plot_dat("ball.dat", (1000,1000))
