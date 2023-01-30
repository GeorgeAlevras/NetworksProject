import numpy as np
from logbin2020 import logbin


def combine_log_bins(data_x, data_y):
    data_x_final = np.unique(np.concatenate(data_x, 0))
    data_y_final = []
    errors = []

    for i in range(len(data_x_final)):
        sample = []
        for j in range(len(data_y)):
            if data_x_final[i] in data_x[j]:
                sample.append(data_y[j][data_x[j].index(data_x_final[i])])
        
        data_y_final.append(np.average(sample))
        errors.append(np.std(sample)/np.sqrt(len(sample)))

    return data_x_final, data_y_final, errors



# Driver program to the above graph class 
if __name__ == "__main__":
    data_1 = [1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    data_2 = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    data_3 = [1, 1, 1, 2, 3, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    x_1, y_1 = logbin(data_1, scale=1.2, zeros=False)
    x_2, y_2 = logbin(data_2, scale=1.2, zeros=False)
    x_3, y_3 = logbin(data_3, scale=1.2, zeros=False)

    # x_1 = [1, 2, 3, 4, 5, 6, 7, 8.2, 9.2, 11, 14, 19]
    # x_2 = [5, 6, 7, 8.2, 9.2, 11, 14, 19]
    # x_3 = [1, 2, 3, 4, 5, 6, 7, 8.2, 9.2, 11, 14, 19]
    
    print(x_1, y_1)
    print(x_2, y_2)
    print(x_3, y_3)

    data_x = [list(x_1), list(x_2), list(x_3)]
    data_y = [list(y_1), list(y_2), list(y_3)]

    print(combine_log_bins(data_x, data_y))
    