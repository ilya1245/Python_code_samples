import os

file = os.path.realpath(__file__)
print(file)

dir_path = os.path.dirname(file)
print(dir_path)

file_name = os.path.basename(file)
print(file_name)