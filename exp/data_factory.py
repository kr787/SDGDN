import os

import torch
import numpy as np
import torch.utils.data

from exp.normalization import NScaler, MinMax01Scaler, MinMax11Scaler, StandardScaler, ColumnMinMaxScaler
from exp.model_factory import DMSTGCN, GCDE, GWN

PEMS04 = 'PEMSD4'
PEMS07 = 'PEMSD7'
PEMS08 = 'PEMSD8'
NYCO = 'NYCTAXI15O'
NYCD = 'NYCTAXI15D'
BJO = 'BJTAXIO'
BJD = 'BJTAXID'


def load_st_dataset(dataset, model):
    # output B, N, D
    if dataset in [PEMS04, PEMS07, PEMS08]:
        data_path = os.path.join('data', dataset, ('PEMS0%s.npz' % dataset[-1]))
        data = np.load(data_path)['data']
        data = data[:, :, 0]  # only the first dimension, traffic flow data

    elif dataset in [NYCO, NYCD]:
        data_path = os.path.join('data', 'NYCTAXI15', 'volume.npz')
        data = np.load(data_path)['volume']
        shape = data.shape
        data = data.reshape(shape[0], -1, shape[-1])  # e.g. [2880, 10, 20, 2] -> [2880, 200, 2]

        if dataset == NYCO:
            data = data[..., :-1]
        elif dataset == NYCD:
            data = data[..., 1:]

    elif dataset in [BJO, BJD]:
        data_path = os.path.join('data', 'BJTAXI', 'data.npz')
        data = np.load(data_path)['data']

        if dataset == BJO:
            data = data[..., :-1]
        elif dataset == BJD:
            data = data[..., 1:]

    else:
        raise ValueError

    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)

    if model == DMSTGCN:
        data = np.concatenate([data, data], axis=-1)

    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data


def save_dataset(x_train, y_train, x_val, y_val, x_test, y_test, dir):
    for name in ('train', 'val', 'test'):
        print(name)
        x = locals()["x_" + name]
        y = locals()["y_" + name]
        print(x.shape, y.shape)
        data_path = os.path.join('data', dir, ('%s.npz' % name))
        np.savez(data_path, x=x, y=y)


def add_window(data, input_window=3, output_window=1, single=False):
    '''
    :param data: shape [B, ...]
    :param input_window:
    :param output_window:
    :return: X is [B, W, ...], Y is [B, H, ...]
    '''
    length = len(data)
    end_index = length - output_window - input_window + 1
    X = []  # input
    Y = []  # output
    index = 0
    if single:
        while index < end_index:
            X.append(data[index:index + input_window])
            Y.append(data[index + input_window + output_window - 1:index + input_window + output_window])
            index = index + 1
    else:
        while index < end_index:
            X.append(data[index:index + input_window])
            Y.append(data[index + input_window:index + input_window + output_window])
            index = index + 1
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def normalize_dataset(data, normalizer, column_wise=False):
    if normalizer == 'max01':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax01Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax01 Normalization')
    elif normalizer == 'max11':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax11Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax11 Normalization')
    elif normalizer == 'std':
        if column_wise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
        else:
            mean = data.mean()
            std = data.std()
        # print(std.shape, data.std(axis=0).shape, data.std(axis=0, keepdims=True).shape), exit()
        scaler = StandardScaler(mean, std)
        data = scaler.transform(data)
        print('Normalize the dataset by Standard Normalization')
    elif normalizer == 'None':
        scaler = NScaler()
        data = scaler.transform(data)
        print('Does not normalize the dataset')
    elif normalizer == 'cmax':
        # column min max, to be depressed
        # note: axis must be the spatial dimension, please check !
        scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
        data = scaler.transform(data)
        print('Normalize the dataset by Column Min-Max Normalization')
    else:
        raise ValueError
    return data, scaler


def split_data_by_days(data, val_days, test_days, interval=60):
    '''
    :param data: [B, *]
    :param val_days:
    :param test_days:
    :param interval: interval (15, 30, 60) minutes
    :return:
    '''
    T = int((24 * 60) / interval)
    test_data = data[-T * test_days:]
    val_data = data[-T * (test_days + val_days): -T * test_days]
    train_data = data[:-T * (test_days + val_days)]
    return train_data, val_data, test_data


def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len * test_ratio):]
    val_data = data[-int(data_len * (test_ratio + val_ratio)):-int(data_len * test_ratio)]
    train_data = data[:-int(data_len * (test_ratio + val_ratio))]
    return train_data, val_data, test_data


def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    TensorFloat = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader


def data_loader_cde(X, Y, batch_size, shuffle=True, drop_last=True):
    TensorFloat = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    data = torch.utils.data.TensorDataset(*X, TensorFloat(Y))
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader


def get_dataloader(args, normalizer='std', tod=False, dow=False, weather=False, single=True, save=False):
    # load raw st dataset
    data = load_st_dataset(args.dataset, args.model)

    # normalize st data
    normal_by_train = False
    if normal_by_train:
        data_len = data.shape[0]
        val_ratio, test_ratio = args.val_ratio, args.test_ratio
        train_data = data[:-int(data_len * (test_ratio + val_ratio))]
        if args.column_wise:
            mean = train_data.mean(axis=0, keepdims=True)
            std = train_data.std(axis=0, keepdims=True)
        else:
            mean = train_data.mean()
            std = train_data.std()
        scaler = StandardScaler(mean, std)
        data = scaler.transform(data)
        print('Normalize the dataset by train data Standard Normalization')
    else:
        data, scaler = normalize_dataset(data, normalizer, args.column_wise)

    if tod:
        MAX_TOD = args.step_per_hour * 24
        tod = [(i % MAX_TOD) for i in range(data.shape[0])]
        if args.model == GWN:
            tod = [t / MAX_TOD for t in tod]
        tod = np.tile(tod, [1, data.shape[1], 1]).transpose((2, 1, 0))
        data = np.concatenate((data, tod), axis=-1)
    if dow:
        MAX_TOD = args.step_per_hour * 24
        MAX_DOW = 7
        dow = [((i // MAX_TOD) % MAX_DOW) for i in range(data.shape[0])]
        dow = np.tile(dow, [1, data.shape[1], 1]).transpose((2, 1, 0))
        data = np.concatenate((data, dow), axis=-1)

    data_train, data_val, data_test = split_data_by_ratio(data, args.val_ratio, args.test_ratio)

    # add time window
    x_tra, y_tra = add_window(data_train, args.input_window, args.output_window, single)
    x_val, y_val = add_window(data_val, args.input_window, args.output_window, single)
    x_test, y_test = add_window(data_test, args.input_window, args.output_window, single)
    print('Train: ', x_tra.shape, y_tra.shape)
    print('Val: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)

    if save:
        save_dataset(x_tra, y_tra, x_val, y_val, x_test, y_test, args.dataset)

    if args.model == GCDE:
        print('Customising dataset for STG-NCDE.')
        times = torch.linspace(0, 11, 12)

        augmented_X_tra = []
        augmented_X_tra.append(
            times.unsqueeze(0).unsqueeze(0).repeat(x_tra.shape[0], x_tra.shape[2], 1).unsqueeze(-1).transpose(1, 2))
        augmented_X_tra.append(torch.Tensor(x_tra[..., :]))
        x_tra = torch.cat(augmented_X_tra, dim=3)
        augmented_X_val = []
        augmented_X_val.append(
            times.unsqueeze(0).unsqueeze(0).repeat(x_val.shape[0], x_val.shape[2], 1).unsqueeze(-1).transpose(1, 2))
        augmented_X_val.append(torch.Tensor(x_val[..., :]))
        x_val = torch.cat(augmented_X_val, dim=3)
        augmented_X_test = []
        augmented_X_test.append(
            times.unsqueeze(0).unsqueeze(0).repeat(x_test.shape[0], x_test.shape[2], 1).unsqueeze(-1).transpose(1, 2))
        augmented_X_test.append(torch.Tensor(x_test[..., :]))
        x_test = torch.cat(augmented_X_test, dim=3)

        import controldiffeq
        train_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, x_tra.transpose(1, 2))
        valid_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, x_val.transpose(1, 2))
        test_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, x_test.transpose(1, 2))

        train_dataloader = data_loader_cde(train_coeffs, y_tra, args.batch_size, shuffle=True, drop_last=True)
        val_dataloader = data_loader_cde(valid_coeffs, y_val, args.batch_size, shuffle=False, drop_last=True)
        test_dataloader = data_loader_cde(test_coeffs, y_test, args.batch_size, shuffle=False, drop_last=False)
    else:
        train_dataloader = data_loader(x_tra, y_tra, args.batch_size, shuffle=True, drop_last=True)
        val_dataloader = data_loader(x_val, y_val, args.batch_size, shuffle=False, drop_last=True)
        test_dataloader = data_loader(x_test, y_test, args.batch_size, shuffle=False, drop_last=False)

    return train_dataloader, val_dataloader, test_dataloader, scaler
