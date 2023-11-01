import csv
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(__file__ + "/../.."))

from exp.data_factory import NYCO, NYCD, BJO, BJD, PEMS04, PEMS07, PEMS08


def read_csv(path, num_nodes, id_path=None):
    a = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    id_dict = {}
    if id_path:
        with open(id_path, 'r') as file_:
            id_dict = {int(i): idx for idx, i in enumerate(file_.read().strip().split('\n'))}

    with open(path, 'r') as file_:
        reader = csv.reader(file_)
        for row in reader:
            if len(row) < 3: continue
            try:
                from_, to, weight = int(row[0]), int(row[1]), float(row[2])
                if id_path:
                    from_ = id_dict[from_]
                    to = id_dict[to]
                a[from_][to] = weight
            except ValueError as e:  # header row (if exists)
                print('ValueError in read_csv:', e)
    return a


def write_csv(path, a):
    with open(path, 'w') as file_:
        writer = csv.writer(file_)
        for i in range(len(a)):
            for j in range(len(a[i])):
                if a[i][j] > 0:
                    writer.writerow([i, j, a[i][j]])


def gen_grid_matrix(max_i, max_j):
    a = np.zeros((max_i * max_j, max_i * max_j), dtype=np.float32)
    ok = lambda x, y: x >= 0 and x < max_i and y >= 0 and y < max_j
    dirs = [[0, 1], [1, 0], [-1, 0], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]
    for i in range(max_i):
        for j in range(max_j):
            from_ = i * max_j + j
            for d in dirs:
                x = i + d[0]
                y = j + d[1]
                if ok(x, y):
                    to = x * max_j + y
                    a[from_][to] = 1
                    a[to][from_] = 1
    return a


def get_adj(dataset, num_nodes):
    id_path = None
    if dataset in [PEMS04, PEMS07, PEMS08]:
        path = os.path.join('data', dataset, ('PEMS0%s.csv' % dataset[-1]))
    elif dataset in [NYCD, NYCO, BJO, BJD]:
        path = os.path.join('data', dataset[:-1], 'adj_8.csv')
    else:
        raise ValueError
    return read_csv(path, num_nodes, id_path)


if __name__ == '__main__':
    import pickle

    # a = read_csv(path='../ASTGNN/data/PEMS04/PEMS04.csv', num_nodes=307)
    # a = get_adj(PEMS04, 307)
    a = gen_grid_matrix(10, 20)
    # print(a)
    write_csv('./data/NYCTAXI15/adj_8.csv', a)
    with open("./data/NYCTAXI15/adj_8.pkl", "wb") as f:
        pickle.dump(a, f)
    a = gen_grid_matrix(32, 32)
    print(a)
    write_csv('./data/BJTAXI/adj_8.csv', a)
    with open("./data/BJTAXI/adj_8.pkl", "wb") as f:
        pickle.dump(a, f)
