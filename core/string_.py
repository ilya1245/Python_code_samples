s = ""
for i in range(5):
    s += '-' + str(i)

print(s)

a_srt = 'qwe/asd'
b_str = a_srt.replace('/', '_')
print(b_str)
print('--------------------------------------------')

# split and join
c_srt= '17.53_20.0 17.58_19.77 17.63_19.55 17.7_19.33 17.79_19.11 17.89_18.89 18.0_18.68'

def get_last_n_points(data: str, number_of_points) -> str:
    data_list = data.split(' ')
    # print(data_list[-number_of_points:])
    return ' '.join(data_list[-number_of_points:])

def cut_last_n_points(data: str, number_of_points) -> str:
    data_list = data.split(' ')
    # print(data_list[-number_of_points:])
    return ' '.join(data_list[:-number_of_points])

print(get_last_n_points(c_srt, 5))
print(cut_last_n_points(c_srt, 1))
