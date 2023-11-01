import csv

import torch


def get_model_parameters(model, only_num=True, show=True):
    if show:
        print('*****************Model Parameter*****************')
    if not only_num:
        for name, param in model.named_parameters():
            print(name, param.shape, param.requires_grad)
    total_num = sum([param.nelement() for param in model.parameters()])
    if show:
        print('Total params num: {}'.format(total_num))
        print('*****************Finish Parameter****************')
    return total_num


def get_memory_usage(device, show=False):
    allocated_memory = torch.cuda.memory_allocated(device) / (1024 * 1024.)
    cached_memory = torch.cuda.memory_reserved(device) / (1024 * 1024.)
    if show:
        print('Allocated Memory: {:.2f} MB, Cached Memory: {:.2f} MB'.format(allocated_memory, cached_memory))
    return allocated_memory, cached_memory


def write_cost(model, data, attr_name, attr, path='./exp/costs.csv'):
    with open(path, 'a') as file_:
        writer = csv.writer(file_)
        writer.writerow((model, data, attr_name, attr))
