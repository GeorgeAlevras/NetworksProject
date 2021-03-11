import sys
import time
import pickle

my_list = [1, 2, 3]
my_dict = {1:1, 2:2, 3:3}

file = open('Files/save.txt', 'wb')
for i in range(2):
    pickle.dump(my_list, file)
file.close()

data = []
file = open ('Files/save.txt', 'rb')
for i in range(2):
    data.append(pickle.load(file))

print(data)
