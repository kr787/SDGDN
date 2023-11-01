import argparse
import configparser
import torch

from exp.data_factory import NYCO, NYCD, BJD, BJO, PEMS04, PEMS07, PEMS08
from exp.model_factory import AGCRN, ASTGNN, DCRNN, DGCRN, DMSTGCN, GCDE, GWN, HA, MTGNN, SCINET, SDGDN


def default_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='arguments')
    parser.add_argument('--dataset', default='PEMSD8', type=str)
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--device', default='cuda:0', type=str, help='indices of GPUs')
    parser.add_argument('--debug', default='True', type=eval)
    parser.add_argument('--model', default='', type=str)
    parser.add_argument('--cuda', default=True, type=bool)

    parser.add_argument('--trained', default='', type=str, help='model path for test')
    parser.add_argument('--trained_dict', default='', type=str, help='model state_dict path for test')

    parser.add_argument('--tod', default=False, type=eval, help='time of day')
    parser.add_argument('--dow', default=False, type=eval, help='day of week')

    # ablation
    parser.add_argument('--woA', default=False, action='store_true',
                        help='without Additional input of time-of-day or day-of-week')
    parser.add_argument('--woO', default=False, action='store_true',
                        help='without One-hot encoding vector for time-of-day and day-of-week')
    parser.add_argument('--woT', default=False, action='store_true',
                        help='without Transition matrix for generating diffusion matrix')
    parser.add_argument('--woV', default=False, action='store_true',
                        help='without Varing graph structures of different time spans')
    parser.add_argument('--woS', default=False, action='store_true',
                        help='without Specific weight parameters for each node')

    return parser


def get_cfg(config_file) -> configparser.ConfigParser:
    print('Read configuration file: %s' % (config_file))
    config = configparser.ConfigParser()
    config.read(config_file)
    return config


def add_common_cfg(parser, dataset) -> argparse.ArgumentParser:
    config = get_cfg('cfg/common/{}.conf'.format(dataset))

    if dataset in [NYCO, NYCD]:
        parser.add_argument('--index_of_saturday', default=2)  # 2015.01.01 = Thursday = 0

    # data
    parser.add_argument('--val_ratio', default=config['data']['val_ratio'], type=float)
    parser.add_argument('--test_ratio', default=config['data']['test_ratio'], type=float)
    parser.add_argument('--input_window', default=config['data']['input_window'], type=int)
    parser.add_argument('--output_window', default=config['data']['output_window'], type=int)
    parser.add_argument('--num_nodes', default=config['data']['num_nodes'], type=int)
    parser.add_argument('--normalizer', default=config['data']['normalizer'], type=str)
    parser.add_argument('--column_wise', default=config['data']['column_wise'], type=eval)
    parser.add_argument('--step_per_hour', default=config['data']['step_per_hour'], type=int)

    # model
    parser.add_argument('--input_dim', default=1, type=int)
    parser.add_argument('--output_dim', default=1, type=int)

    # train
    parser.add_argument('--epochs', default=512, type=int)
    parser.add_argument('--lr_init', default=0.001, help='learning rate')
    parser.add_argument('--loss_func', default='mae')

    # metric
    parser.add_argument('--mae_thresh', default=config['metric']['mae_thresh'], type=eval)
    parser.add_argument('--mape_thresh', default=config['metric']['mape_thresh'], type=float)
    parser.add_argument('--real_value_train', default=True,
                        help='inverse_transform(label) for loss calculation during training')
    parser.add_argument('--real_value_test', default=True,
                        help='inverse_transform(label) for loss calculation during testing and validating')
    parser.add_argument('--real_value_output_train', default=False,
                        help='inverse_transform(output) for loss calculation during training')
    parser.add_argument('--real_value_output_test', default=False,
                        help='inverse_transform(output) for loss calculation during testing and validating')

    # log
    parser.add_argument('--log_dir', default='./', type=str)
    parser.add_argument('--log_step', default=config['log']['log_step'], type=int)
    parser.add_argument('--plot', default=config['log']['plot'], type=eval)
    return parser


def get_agcrn_args(parser, dataset) -> argparse.Namespace:
    if dataset in [PEMS04, BJO, BJD]:
        config = get_cfg('cfg/agcrn/{}_AGCRN.conf'.format(PEMS04))
    else:
        config = get_cfg('cfg/agcrn/{}_AGCRN.conf'.format(PEMS08))

    # data
    # parser.add_argument('--tod', default=config['data']['tod'], type=eval)
    # parser.add_argument('--normalizer', default=config['data']['normalizer'], type=str)
    # parser.add_argument('--column_wise', default=config['data']['column_wise'], type=eval)
    parser.add_argument('--lag', default=parser.parse_args().input_window)
    parser.add_argument('--horizon', default=parser.parse_args().output_window)
    parser.add_argument('--default_graph', default=config['data']['default_graph'], type=eval)
    # model
    # parser.add_argument('--input_dim', default=config['model']['input_dim'], type=int)
    # parser.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
    parser.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int)
    parser.add_argument('--rnn_units', default=config['model']['rnn_units'], type=int)
    parser.add_argument('--num_layers', default=config['model']['num_layers'], type=int)
    parser.add_argument('--cheb_k', default=config['model']['cheb_order'], type=int)
    # train
    # parser.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
    # parser.add_argument('--seed', default=config['train']['seed'], type=int)
    parser.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
    # parser.add_argument('--epochs', default=config['train']['epochs'], type=int)
    # parser.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
    parser.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
    parser.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
    parser.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
    # parser.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
    # parser.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
    parser.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
    parser.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
    # parser.add_argument('--teacher_forcing', default=False, type=bool)
    # parser.add_argument('--tf_decay_steps', default=2000, type=int, help='teacher forcing decay steps')
    # parser.add_argument('--real_value', default=config['train']['real_value'], type=eval, help = 'use real value for loss calculation')
    args = parser.parse_args()

    args.lr_init = float(config['train']['lr_init'])
    if dataset in [NYCO, NYCD, BJO, BJD]:
        args.lr_init = 0.01
    args.epochs = int(config['train']['epochs'])
    return args


def get_astgnn_args(parser, dataset) -> argparse.Namespace:
    config = get_cfg('cfg/astgnn/{}.conf'.format(dataset))

    parser.add_argument('--astgnn_config', default=config)

    training_config = config['Training']
    learning_rate = float(training_config['learning_rate'])
    epochs = int(training_config['epochs'])
    fine_tune_epochs = int(training_config['fine_tune_epochs'])
    batch_size = int(training_config['batch_size'])

    parser.add_argument('--batch_size', default=batch_size)
    parser.add_argument('--fine_tune_epochs', default=fine_tune_epochs, type=int)

    args = parser.parse_args()

    args.lr_init = learning_rate
    args.epochs = epochs
    return args


def get_dcrnn_args(parser, dataset) -> argparse.Namespace:
    # parser.add_argument('--device', default='cuda:5', type=str, help='indices of GPUs')
    # parser.add_argument('--debug', default=False, type=bool)
    # Model details
    parser.add_argument('--enc_input_dim', default=2, type=int)
    # parser.add_argument('--input_dim', default=2, type=int)
    parser.add_argument('--dec_input_dim', default=1, type=int)
    # parser.add_argument('--output_dim', default=1, type=int)
    parser.add_argument('--diffusion_step', default=2, type=int)
    # parser.add_argument('--num_nodes', default=207, type=int)
    parser.add_argument('--num_rnn_layers', default=2, type=int)
    parser.add_argument('--rnn_units', default=64, type=int)
    parser.add_argument('--seq_len', default=12, type=int)
    # training
    parser.add_argument('--batch_size', default=64, type=int)
    # parser.add_argument('--epochs', default=100, type=int)
    # parser.add_argument('--lr_init', default=0.01, type=float)
    parser.add_argument('--lr_type', default='MultiStepLR', type=str)
    parser.add_argument('--lr_milestones', default='20,30,40,50', type=str)
    parser.add_argument('--lr_decay_rate', default=0.1, type=float)
    parser.add_argument('--max_grad_norm', default=5, type=int)

    parser.add_argument('--lr_decay', default=True)
    parser.add_argument('--early_stop', default=True)
    parser.add_argument('--early_stop_patience', default=10, type=int)

    # parser.add_argument('--tf_decay_steps', default=2000, type=int, help='teacher forcing decay steps')
    # log
    # parser.add_argument('--log_dir', default='./', type=str)
    # parser.add_argument('--log_step', default=20, type=int)
    # parser.add_argument('--plot', default=True, type=bool)

    args = parser.parse_args()
    args.epochs = 100

    if dataset in [NYCO, NYCD]:
        args.lr_decay = False
        args.lr_init = 0.01
    elif dataset in [BJO, BJD]:
        args.early_stop_patience = 20
        args.lr_decay = True
        args.lr_init = 0.01
        args.lr_decay_rate = 0.9
    elif dataset in [PEMS04, PEMS08]:
        args.early_stop_patience = 20
        args.lr_decay = True
        args.lr_init = 0.003
        args.lr_decay_rate = 0.3
    else:
        args.early_stop_patience = 20
        args.lr_decay = True
        args.lr_init = 0.005
        args.lr_decay_rate = 0.5
    return args


# deprecated
def get_dgcrn_args(parser, dataset) -> argparse.Namespace:
    def str_to_bool(value):
        if isinstance(value, bool):
            return value
        if value.lower() in {'false', 'f', '0', 'no', 'n'}:
            return False
        elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
            return True
        raise ValueError(f'{value} is not a valid boolean value')

    # parser.add_argument('--tolerance', type=int, default=100, help='tolerance for earlystopping')
    parser.add_argument('--OUTPUT_PREDICTION', default=False, type=str_to_bool, help='If OUTPUT_PREDICTION.')

    parser.add_argument('--cl_decay_steps', default=2000, type=float, help='cl_decay_steps.')
    parser.add_argument('--new_training_method', default=False, type=str_to_bool, help='new_training_method.')
    parser.add_argument('--rnn_size', default=64, type=int, help='rnn_size.')
    parser.add_argument('--hyperGNN_dim', default=16, type=int, help='hyperGNN_dim.')

    # parser.add_argument('--device', type=str, default='cuda:1', help='')
    # parser.add_argument('--data', type=str, default='data/METR-LA', help='data path')

    # parser.add_argument('--adj_data', type=str, default='data/sensor_graph/adj_mx.pkl', help='adj data path')
    parser.add_argument('--propalpha', type=float, default=0.05, help='prop alpha')

    parser.add_argument('--cl', type=str_to_bool, default=True, help='whether to do curriculum learning')

    parser.add_argument('--gcn_depth', type=int, default=2, help='graph convolution depth')
    # parser.add_argument('--num_nodes', type=int, default=207, help='number of nodes/variables')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--subgraph_size', type=int, default=20, help='k')
    parser.add_argument('--node_dim', type=int, default=40, help='dim of nodes')

    parser.add_argument('--in_dim', type=int, default=2, help='inputs dimension')
    parser.add_argument('--seq_in_len', type=int, default=12, help='input sequence length')
    parser.add_argument('--seq_out_len', type=int, default=12, help='output sequence length')

    parser.add_argument('--layers', type=int, default=3, help='number of layers')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--tanhalpha', type=float, default=3, help='adj alpha')

    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
    parser.add_argument('--clip', type=int, default=5, help='clip')
    parser.add_argument('--step_size1', type=int, default=2500, help='step_size')

    # parser.add_argument('--epochs', type=int, default=100, help='')
    # parser.add_argument('--print_every', type=int, default=50, help='')
    # parser.add_argument('--save', type=str, default='./save/', help='save path')

    # parser.add_argument('--expid', type=str, default='1', help='experiment id')

    args = parser.parse_args()
    args.epochs = 100
    args.early_stop_patience = 100
    args.tod = True
    args.real_value_output_train = True
    args.real_value_output_test = True

    if dataset in [NYCO, NYCD]:
        args.lr_decay = False
        args.lr_init = 0.001
        args.seq_out_len = args.output_window = 1
    else:
        args.lr_decay = True
        args.lr_init = 0.003
        args.lr_decay_rate = 0.8
    return args


def get_dmstgcn_args(parser, dataset) -> argparse.Namespace:
    # parser.add_argument('--device', type=str, default='cuda:0', help='')
    # parser.add_argument('--data', type=str, default='PEMSD4', help='data path')
    parser.add_argument('--seq_length', type=int, default=12, help='output length')
    parser.add_argument('--in_len', type=int, default=12, help='input length')
    parser.add_argument('--nhid', type=int, default=32, help='')
    parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
    # parser.add_argument('--epochs', type=int, default=200, help='')
    # parser.add_argument('--print_every', type=int, default=50, help='')
    # parser.add_argument('--runs', type=int, default=1, help='number of experiments')
    # parser.add_argument('--expid', type=int, default=1, help='experiment id')
    # parser.add_argument('--iden', type=str, default='', help='identity')
    parser.add_argument('--dims', type=int, default=32, help='dimension of embeddings for dynamic graph')
    parser.add_argument('--order', type=int, default=2, help='order of graph convolution')

    parser.add_argument('--days', type=int, default=288,
                        help='number of timeslots in a day which depends on the dataset')
    parser.add_argument('--normalization', default='batch')

    args = parser.parse_args()

    args.in_len = args.input_window
    args.seq_length = args.output_window
    args.learning_rate = args.lr_init
    args.epochs = 400
    args.days = args.step_per_hour * 24
    args.tod = True

    return args


def get_gwn_args(parser, dataset) -> argparse.Namespace:
    # parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
    # parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
    # parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
    # parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
    # parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    # parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
    # parser.add_argument('--epochs',type=int,default=100,help='')
    # parser.add_argument('--print_every',type=int,default=50,help='')
    # parser.add_argument('--seed',type=int,default=99,help='random seed')

    # parser.add_argument('--lr_init',type=float,default=0.001,help='learning rate')
    # parser.add_argument('--lr_decay', default=False, type=eval)
    # parser.add_argument('--early_stop', default=False, type=eval)
    # parser.add_argument('--early_stop_patience', default=0, type=int)
    # parser.add_argument('--loss_func', default='mask_mae', type=str)
    # parser.add_argument('--grad_norm', default=False, type=eval)

    args = parser.parse_args()

    args.epochs = 100
    args.tod = True
    args.dow = False
    args.input_dim += 1  # tod
    args.real_value_output_train = args.real_value_output_test = True
    return args


def get_scinet_args(parser, dataset) -> argparse.Namespace:
    ### -------  input/output length settings --------------
    # parser.add_argument('--window_size', type=int, default=12)
    # parser.add_argument('--horizon', type=int, default=12)
    parser.add_argument('--concat_len', type=int, default=0)
    parser.add_argument('--single_step_output_One', type=int, default=0)

    # parser.add_argument('--train_length', type=float, default=6)
    # parser.add_argument('--valid_length', type=float, default=2)
    # parser.add_argument('--test_length', type=float, default=2)

    ### -------  training settings --------------
    # parser.add_argument('--train', type=bool, default=True)
    # parser.add_argument('--resume', type=bool, default=False)
    # parser.add_argument('--evaluate', type=bool, default=False)
    # parser.add_argument('--finetune', type=bool, default=False)
    # parser.add_argument('--validate_freq', type=int, default=1)

    # parser.add_argument('--epoch', type=int, default=80)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=8)
    # parser.add_argument('--optimizer', type=str, default='N') #
    parser.add_argument('--early_stop', type=bool, default=False)
    parser.add_argument('--exponential_decay_step', type=int, default=5)
    parser.add_argument('--decay_rate', type=float, default=0.5)

    parser.add_argument('--lradj', type=int, default=1, help='adjust learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    # parser.add_argument('--model_name', type=str, default='SCINet')

    ### -------  model settings --------------
    parser.add_argument('--hidden-size', default=0.0625, type=float, help='hidden channel scale of module')
    parser.add_argument('--INN', default=1, type=int, help='use INN or basic strategy')
    parser.add_argument('--kernel', default=5, type=int, help='kernel size for the first layer')
    # parser.add_argument('--dilation', default=1, type=int, help='dilation')
    parser.add_argument('--positionalEcoding', type=bool, default=True)
    parser.add_argument('--groups', type=int, default=1)
    parser.add_argument('--levels', type=int, default=2)
    parser.add_argument('--stacks', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_decoder_layer', type=int, default=1)
    parser.add_argument('--RIN', type=bool, default=False)
    # parser.add_argument('--decompose', type=bool,default=False)

    args = parser.parse_args()

    args.real_value_train = False
    args.real_value_output_train = False
    args.real_value_test = True
    args.real_value_output_test = True

    args.epochs = 80
    args.lr_init = args.lr
    if dataset == PEMS04:
        args.hidden_size = 0.0625
        args.dropout = 0
    elif dataset == PEMS07:
        args.hidden_size = 0.03125
        args.dropout = 0.25
    elif dataset == PEMS08:
        args.hidden_size = 1
        args.dropout = 0.5
    elif dataset in [NYCO, NYCD]:
        args.hidden_size = 0.25
        args.dropout = 0.25
    elif dataset in [BJO, BJD]:
        args.hidden_size = 0.25
        args.dropout = 0.25
    else:
        raise ValueError(f'unsupport dataset: {dataset}')
    return args


def get_gcde_args(args) -> argparse.Namespace:
    config = get_cfg('cfg/gcde/PEMSD4_GCDE.conf')

    # data
    args.add_argument('--lag', default=args.parse_args().input_window)
    args.add_argument('--horizon', default=args.parse_args().output_window)
    args.add_argument('--default_graph', default=config['data']['default_graph'], type=eval)
    # model
    args.add_argument('--model_type', default=config['model']['type'], type=str)
    args.add_argument('--g_type', default=config['model']['g_type'], type=str)
    # args.add_argument('--input_dim', default=config['model']['input_dim'], type=int)
    # args.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
    args.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int)
    args.add_argument('--hid_dim', default=config['model']['hid_dim'], type=int)
    args.add_argument('--hid_hid_dim', default=config['model']['hid_hid_dim'], type=int)
    args.add_argument('--num_layers', default=config['model']['num_layers'], type=int)
    args.add_argument('--cheb_k', default=config['model']['cheb_order'], type=int)
    args.add_argument('--solver', default='rk4', type=str)

    # train
    # args.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
    # args.add_argument('--seed', default=config['train']['seed'], type=int)
    args.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
    # args.add_argument('--epochs', default=config['train']['epochs'], type=int)
    # args.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
    args.add_argument('--weight_decay', default=config['train']['weight_decay'], type=eval)
    args.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
    args.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
    args.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
    args.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
    args.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
    args.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
    args.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
    args.add_argument('--teacher_forcing', default=False, type=bool)
    # args.add_argument('--tf_decay_steps', default=2000, type=int, help='teacher forcing decay steps')
    # args.add_argument('--real_value', default=config['train']['real_value'], type=eval, help = 'use real value for loss calculation')

    args.add_argument('--missing_test', default=False, type=bool)
    args.add_argument('--missing_rate', default=0.1, type=float)

    args = args.parse_args()

    args.epochs = int(config['train']['epochs'])
    args.lr_init = float(config['train']['lr_init'])
    args.loss_func = config['train']['loss_func']
    args.input_dim = int(config['model']['input_dim'])

    return args


# deprecated
def get_mtgnn_args(parser, dataset) -> argparse.Namespace:
    def str_to_bool(value):
        if isinstance(value, bool):
            return value
        if value.lower() in {'false', 'f', '0', 'no', 'n'}:
            return False
        elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
            return True
        raise ValueError(f'{value} is not a valid boolean value')

    # parser.add_argument('--adj_data', type=str,default='data/sensor_graph/adj_mx.pkl',help='adj data path')
    parser.add_argument('--gcn_true', type=str_to_bool, default=True, help='whether to add graph convolution layer')
    parser.add_argument('--buildA_true', type=str_to_bool, default=True,
                        help='whether to construct adaptive adjacency matrix')
    parser.add_argument('--load_static_feature', type=str_to_bool, default=False, help='whether to load static feature')
    parser.add_argument('--cl', type=str_to_bool, default=True, help='whether to do curriculum learning')

    parser.add_argument('--gcn_depth', type=int, default=2, help='graph convolution depth')
    # parser.add_argument('--num_nodes',type=int,default=207,help='number of nodes/variables')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--subgraph_size', type=int, default=20, help='k')
    parser.add_argument('--node_dim', type=int, default=40, help='dim of nodes')
    parser.add_argument('--dilation_exponential', type=int, default=1, help='dilation exponential')

    parser.add_argument('--conv_channels', type=int, default=32, help='convolution channels')
    parser.add_argument('--residual_channels', type=int, default=32, help='residual channels')
    parser.add_argument('--skip_channels', type=int, default=64, help='skip channels')
    parser.add_argument('--end_channels', type=int, default=128, help='end channels')

    parser.add_argument('--in_dim', type=int, default=parser.parse_args().input_dim, help='inputs dimension')
    parser.add_argument('--seq_in_len', type=int, default=parser.parse_args().input_window,
                        help='input sequence length')
    parser.add_argument('--seq_out_len', type=int, default=parser.parse_args().output_window,
                        help='output sequence length')

    parser.add_argument('--layers', type=int, default=3, help='number of layers')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    # parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
    # parser.add_argument('--clip',type=int,default=5,help='clip')
    parser.add_argument('--max_grad_norm', default=5, type=int)
    ## parser.add_argument('--task_level',type=int,default=1,help='loss(y[:task_level])')
    ## parser.add_argument('--cl_iter',type=int,default=1,help='to compute task_level')
    parser.add_argument('--step_size1', type=int, default=2500, help='step_size for updating loss task_level')
    parser.add_argument('--step_size2', type=int, default=100, help='step_size for permuting nodes')

    # parser.add_argument('--epochs',type=int,default=100,help='')
    # parser.add_argument('--print_every',type=int,default=50,help='')
    # parser.add_argument('--seed',type=int,default=101,help='random seed')
    # parser.add_argument('--save',type=str,default='./save/',help='save path')
    # parser.add_argument('--expid',type=int,default=1,help='experiment id')

    parser.add_argument('--propalpha', type=float, default=0.05, help='prop alpha')
    parser.add_argument('--tanhalpha', type=float, default=3, help='adj alpha')

    parser.add_argument('--num_split', type=int, default=1, help='number of splits for graphs')

    # parser.add_argument('--runs',type=int,default=10,help='number of runs')

    args = parser.parse_args()

    args.epochs = 100
    args.loss_func = 'mask_mae'
    args.real_value_train = args.real_value_test = args.real_value_output_train = args.real_value_output_test = True

    return args


def get_default_args(parser, dataset) -> argparse.Namespace:
    config = get_cfg('cfg/{}.conf'.format(dataset))

    # data
    parser.add_argument('--default_graph', default=config['data']['default_graph'], type=eval)
    # model
    parser.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int)
    # parser.add_argument('--rnn_units', default=config['model']['rnn_units'], type=int)
    parser.add_argument('--num_layers', default=config['model']['num_layers'], type=int)
    parser.add_argument('--cheb_k', default=config['model']['k_order'], type=int)
    parser.add_argument('--k', default=config['model']['k_order'], type=int)
    parser.add_argument('--residual_channels', default=32, type=int)
    # parser.add_argument('--delta_k', default=-0.25, type=float)
    # parser.add_argument('--pad', default=True, type=eval)
    # train
    # parser.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
    # parser.add_argument('--seed', default=config['train']['seed'], type=int)
    parser.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
    # parser.add_argument('--epochs', default=config['train']['epochs'], type=int)
    # parser.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
    parser.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
    parser.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
    # parser.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
    parser.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
    parser.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
    parser.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
    parser.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)

    parser.add_argument('--cl_step', default=config['train']['cl_step'], type=int)

    args = parser.parse_args()

    args.epochs = int(config['train']['epochs'])
    args.lr_init = float(config['train']['lr_init'])
    args.loss_func = config['train']['loss_func']
    args.input_dim = int(config['model']['input_dim'])
    args.output_dim = int(config['model']['output_dim'])

    return args


def get_args() -> argparse.Namespace:
    parser = default_parser()

    # get configuration
    dataset = parser.parse_args().dataset
    add_common_cfg(parser, dataset)

    model = parser.parse_args().model
    if model == AGCRN:
        args = get_agcrn_args(parser, dataset)
        if args.tod:
            args.input_dim += 1
        if args.dow:
            args.input_dim += 1
    elif model == ASTGNN:
        args = get_astgnn_args(parser, dataset)
    elif model == DCRNN:
        args = get_dcrnn_args(parser, dataset)
    elif model == DGCRN:
        args = get_dgcrn_args(parser, dataset)
        if args.tod:
            args.input_dim += 1
    elif model == DMSTGCN:
        args = get_dmstgcn_args(parser, dataset)
    elif model == GCDE:
        args = get_gcde_args(parser)
    elif model == GWN:
        args = get_gwn_args(parser, dataset)
    elif model == MTGNN:
        args = get_mtgnn_args(parser, dataset)
    elif model == SCINET:
        args = get_scinet_args(parser, dataset)
    else:
        args = get_default_args(parser, dataset)
        if model == HA:
            args.normalizer = 'None'
            args.tod = args.dow = False
            args.real_value_train = args.real_value_test = args.real_value_output_train = args.real_value_output_test = False
            args.epochs = 1
            args.cl_step = 0
        if SDGDN in model:
            args.tod = args.dow = True

            args.lr_decay = True

            if args.woA:
                args.tod = args.dow = False
                args.input_dim = 1
            elif args.woO:
                if args.tod:
                    args.input_dim += 1
                if args.dow:
                    args.input_dim += 1
            elif args.tod and args.dow:
                args.input_dim *= (24 * args.step_per_hour + 7)

    print('curriculum learning step', hasattr(args, 'cl_step') and args.cl_step or 0)
    print('batch_size', args.batch_size)
    print('early_stop_patience', hasattr(args, 'early_stop_patience') and args.early_stop_patience or 0)

    print('cudnn', torch.backends.cudnn.enabled)  # True
    print('deterministic', torch.backends.cudnn.deterministic)  # False
    print('benchmark', torch.backends.cudnn.benchmark)  # False

    if 'cost' in args.mode:
        args.epochs = 1
        args.lr_decay = False
        if args.model == ASTGNN:
            args.fine_tune_epochs = 1
        if SDGDN in args.model:
            args.cl_step = 0

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.device[5]))
    else:
        args.device = 'cpu'

    return args
