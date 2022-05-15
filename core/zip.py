import numpy as np

numbers = [1, 2, 3]
letters = ['a', 'b', 'c']
zipped = zip(numbers, letters)

print(type(zipped))
l_zip = list(zipped)
print(l_zip)
print(*l_zip)

# need to init zipped again
zipped = zip(numbers, letters)
for (n, l) in zipped:
    print(n, l)

print('--------------------------')
l_zip_2 = list(zip(*l_zip))
print(l_zip_2)

# look through the files in zip
from zipfile import ZipFile
zipfilename = 'some.zip'
with ZipFile(zipfilename) as zip:
    for filename in zip.filelist:
        with zip.open(filename) as f:
            for line in f:
                if "[Result" in str(line):
                    total += 1