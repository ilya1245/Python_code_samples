# Prints today's date with help
# of datetime library
import datetime

today = datetime.datetime.today()
print(f"{today:%B %d, %Y}")

a = 123.456
b = 234.567
c = 890
print(f'a = {a}   b = {b}')
print('a = %.1f   b = %.2f' % (a, b))
print('c = %i ' % (c))

s1 = 'a = %s'
print(s1 % ('qwe'))