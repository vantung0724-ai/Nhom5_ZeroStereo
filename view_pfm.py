import numpy as np
import matplotlib.pyplot as plt

def read_pfm(file):
    with open(file, 'rb') as f:
        header = f.readline().decode().rstrip()
        width, height = map(int, f.readline().decode().split())
        scale = float(f.readline().decode())
        data = np.fromfile(f, '<f')
        return np.reshape(data, (height, width))

pfm_path = "dataset/disparity/a_rain_of_stones_x2/left/0000.pfm"
img = read_pfm(pfm_path)

plt.imshow(img, cmap='jet')
plt.colorbar()
plt.show()
