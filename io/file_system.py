import yaml
import numpy as np
import os
from glob import glob

PROJECT_ROOT = ".."

with open(os.path.join(PROJECT_ROOT, "config.yml"), "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

exec = cfg['exec']

# get all file names in folder
mypath = os.path.join(PROJECT_ROOT, exec['data_folder'])
filenames = np.array(glob(os.path.join(mypath, '*.*')))
print('folder:', mypath)
print('filenames:', filenames)