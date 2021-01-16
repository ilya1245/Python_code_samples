numbers = [1, 2, 3]
letters = ['a', 'b', 'c']
zipped = zip(numbers, letters)

print(type(zipped))
print(list(zipped))

# need to init zipped again
zipped = zip(numbers, letters)
for (n, l) in zipped:
    print(n, l)
