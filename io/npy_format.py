import yaml
import numpy as np
import os
import matplotlib.pyplot as plt

PROJECT_ROOT = ".."

with open(PROJECT_ROOT + "/config.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

exec = cfg['exec']
data_path = os.path.join(PROJECT_ROOT, exec['data_folder'], exec['data_file'])

# Load npy file containing images 28x28 in gray
npy_array = np.load(data_path)
print('npy_array shape:', npy_array.shape)

# reshape to image format
img_array = npy_array.reshape(npy_array.shape[0], 28, 28)
print('img_array shape:', img_array.shape)
# print(img_array[0])
plt.imshow(img_array[0], cmap='gray')
plt.show()

# reshape to data format
x_array = npy_array.reshape(npy_array.shape[0], 28, 28, 1)
print('x_array shape:', x_array.shape)
# print(x_array[0])

# variant 1
plt.imshow(x_array[1].reshape(28, 28), cmap='gray')
# variant 2
# plt.imshow(x_array.reshape(x_array.shape[0], 28, 28)[1], cmap='gray')
plt.show()
# sys.exit("Stopped")






