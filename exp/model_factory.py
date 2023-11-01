import argparse
import configparser
import os
import sys

from importlib import import_module

import numpy as np
import torch
from torch import nn

AGCRN = 'agcrn'
ASTGNN = 'astgnn'
DCRNN = 'dcrnn'
DGCRN = 'dgcrn'
DMSTGCN = 'dmstgcn'
GCDE = 'gcde'
GWN = 'gwn'
HA = 'ha'
MTGNN = 'mtgnn'
SCINET = 'scinet'
SDGDN = 'SDGDN'


class _HA(nn.Module):
    def __init__(self, args):
        super(_HA, self).__init__()
        self.out_len = args.output_window
        self.a = nn.Parameter(torch.rand(1))

    def forward(self, x):
        in_len = x.shape[1]
        y = self.a * torch.zeros_like(x[:, :, :, 0])
        for i in range(self.out_len):
            y[:, i, :] = x[:, :, :, 0].sum(axis=1) / in_len
        return y[:, :self.out_len, :].unsqueeze(dim=3)


def get_model(args):
    global AGCRN, ASTGNN, DCRNN, DGCRN, DMSTGCN, GCDE, GWN, HA, MTGNN, SCINET, SDGDN
    print(args)
    if args.model == AGCRN:
        sys.path.append(os.path.join(sys.path[-1], 'AGCRN'))
        from AGCRN.model.AGCRN import AGCRN
        model = AGCRN(args)
    elif args.model == ASTGNN:
        sys.path.append(os.path.join(sys.path[-1], 'ASTGNN'))
        from lib.utils import get_adjacency_matrix, get_adjacency_matrix_2direction

        DEVICE = torch.device(args.device)
        config = args.astgnn_config
        data_config = config['Data']
        training_config = config['Training']
        adj_filename = data_config['adj_filename']
        # graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
        if config.has_option('Data', 'id_filename'):
            id_filename = data_config['id_filename']
        else:
            id_filename = None
        num_of_vertices = int(data_config['num_of_vertices'])
        points_per_hour = int(data_config['points_per_hour'])
        num_for_predict = int(data_config['num_for_predict'])
        # dataset_name = data_config['dataset_name']
        # model_name = training_config['model_name']
        # learning_rate = float(training_config['learning_rate'])
        # start_epoch = int(training_config['start_epoch']) 
        # epochs = int(training_config['epochs'])
        # fine_tune_epochs = int(training_config['fine_tune_epochs'])
        # print('total training epoch, fine tune epoch:', epochs, ',' , fine_tune_epochs, flush=True)
        # batch_size = int(training_config['batch_size'])
        # print('batch_size:', batch_size, flush=True)
        num_of_weeks = int(training_config['num_of_weeks'])
        num_of_days = int(training_config['num_of_days'])
        num_of_hours = int(training_config['num_of_hours'])
        direction = int(training_config['direction'])
        encoder_input_size = int(training_config['encoder_input_size'])
        decoder_input_size = int(training_config['decoder_input_size'])
        dropout = float(training_config['dropout'])
        kernel_size = int(training_config['kernel_size'])

        # filename_npz = os.path.join(dataset_name + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks)) + '.npz'
        num_layers = int(training_config['num_layers'])
        d_model = int(training_config['d_model'])
        nb_head = int(training_config['nb_head'])
        ScaledSAt = bool(int(training_config['ScaledSAt']))  # whether use spatial self attention
        SE = bool(int(training_config['SE']))  # whether use spatial embedding
        smooth_layer_num = int(training_config['smooth_layer_num'])
        aware_temporal_context = bool(int(training_config['aware_temporal_context']))
        TE = bool(int(training_config['TE']))
        use_LayerNorm = True
        residual_connection = True

        # direction = 1 means: if i connected to j, adj[i,j]=1;
        # direction = 2 means: if i connected to j, then adj[i,j]=adj[j,i]=1
        if direction == 2:
            adj_mx, distance_mx = get_adjacency_matrix_2direction(adj_filename, num_of_vertices, id_filename)
        if direction == 1:
            adj_mx, distance_mx = get_adjacency_matrix(adj_filename, num_of_vertices, id_filename)
        # folder_dir = 'MAE_%s_h%dd%dw%d_layer%d_head%d_dm%d_channel%d_dir%d_drop%.2f_%.2e' % (model_name, num_of_hours, num_of_days, num_of_weeks, num_layers, nb_head, d_model, encoder_input_size, direction, dropout, learning_rate)

        make_model = getattr(import_module('ASTGNN.model.ASTGNN'), 'make_model')

        model = make_model(DEVICE, num_layers, encoder_input_size, decoder_input_size, d_model, adj_mx, nb_head,
                           num_of_weeks,
                           num_of_days, num_of_hours, points_per_hour, num_for_predict, dropout=dropout,
                           aware_temporal_context=aware_temporal_context,
                           ScaledSAt=ScaledSAt, SE=SE, TE=TE, kernel_size=kernel_size,
                           smooth_layer_num=smooth_layer_num,
                           residual_connection=residual_connection, use_LayerNorm=use_LayerNorm)
    elif args.model == DCRNN:
        '''
        git clone https://github.com/LeiBAI/DCRNN_Pytorch.git
        mkdir ./Model
        mv ./DCRNN_Pytorch ./Model/DCRNN1
        '''

        from exp.adj_factory_for_baseline import get_adj
        adj_mx = get_adj(args.dataset, args.num_nodes)

        asym_adj = getattr(import_module('Graph-WaveNet.util'), 'asym_adj')
        adj_mx = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
        supports = [torch.tensor(i).to(args.device) for i in adj_mx]

        from Model.DCRNN1.DCRNNModel import DCRNNModel
        model = DCRNNModel(supports, num_node=args.num_nodes, input_dim=args.input_dim,
                           hidden_dim=args.rnn_units, out_dim=args.output_dim,
                           order=args.diffusion_step, num_layers=args.num_rnn_layers)
    elif args.model == DGCRN:  # deprecated
        sys.path.append(os.path.join(sys.path[-1], 'Traffic-Benchmark'))
        sys.path.append(os.path.join(sys.path[-1], 'methods'))
        sys.path.append(os.path.join(sys.path[-1], 'DGCRN'))

        supports = None
        from exp.adj_factory_for_baseline import get_adj
        adj_mx = get_adj(args.dataset, args.num_nodes)
        asym_adj = getattr(import_module('Graph-WaveNet.util'), 'asym_adj')
        adj_mx = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
        supports = [torch.tensor(i).to(args.device) for i in adj_mx]

        model = getattr(import_module('methods.DGCRN.net'), 'DGCRN')
        model = model(args.gcn_depth,
                      args.num_nodes,
                      args.device,
                      predefined_A=supports,
                      dropout=args.dropout,
                      subgraph_size=args.subgraph_size,
                      node_dim=args.node_dim,
                      middle_dim=2,
                      seq_length=args.seq_in_len,
                      in_dim=args.in_dim,
                      out_dim=args.seq_out_len,
                      layers=args.layers,
                      list_weight=[0.05, 0.95, 0.95],
                      tanhalpha=args.tanhalpha,
                      cl_decay_steps=args.cl_decay_steps,
                      rnn_size=args.rnn_size,
                      hyperGNN_dim=args.hyperGNN_dim)
    elif args.model == DMSTGCN:
        sys.path.append(os.path.join(sys.path[-1], 'DMSTGCN'))
        model_ = getattr(import_module('model'), 'DMSTGCN')
        model = model_(args.device, args.num_nodes, args.dropout, out_dim=args.seq_length, residual_channels=args.nhid,
                       dilation_channels=args.nhid, end_channels=args.nhid * 16, days=args.days, dims=args.dims,
                       order=args.order,
                       in_dim=args.in_dim, normalization=args.normalization)
    elif args.model == GCDE:
        sys.path.append(os.path.join(sys.path[-1], 'STG-NCDE'))
        sys.path.append(os.path.join(sys.path[-1], 'model'))
        make_model = getattr(import_module('STG-NCDE.model.Make_model'), 'make_model')
        model, _, _ = make_model(args)
    elif args.model == GWN:
        gwnet = getattr(import_module('Graph-WaveNet.model'), 'gwnet')
        supports = None
        if args.dataset in ['METRLA', 'PEMSBAY']:
            load_adj = getattr(import_module('Graph-WaveNet.util'), 'load_adj')
            sensor_ids, sensor_id_to_ind, adj_mx = load_adj(args.adjdata, args.adjtype)
            supports = [torch.tensor(i).to(args.device) for i in adj_mx]
        else:
            from exp.adj_factory_for_baseline import get_adj
            adj_mx = get_adj(args.dataset, args.num_nodes)
            asym_adj = getattr(import_module('Graph-WaveNet.util'), 'asym_adj')
            adj_mx = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
            supports = [torch.tensor(i).to(args.device) for i in adj_mx]
        model = gwnet(device=args.device, num_nodes=args.num_nodes, in_dim=args.input_dim, supports=supports)
    elif args.model == SCINET:
        from SCINet.models.SCINet import SCINet
        model = SCINet(
            output_len=args.output_window,
            input_len=args.input_window,
            input_dim=args.num_nodes,
            hid_size=args.hidden_size,
            num_stacks=args.stacks,
            num_levels=args.levels,
            num_decoder_layer=args.num_decoder_layer,
            concat_len=args.concat_len,
            groups=args.groups,
            kernel=args.kernel,
            dropout=args.dropout,
            single_step_output_One=args.single_step_output_One,
            positionalE=args.positionalEcoding,
            modified=True,
            RIN=args.RIN
        )
    elif args.model == HA:
        model = _HA(args)
    elif args.model == MTGNN:  # deprecated
        sys.path.append(os.path.join(sys.path[-1], 'MTGNN'))
        from MTGNN.net import gtnet
        predefined_A = None
        if hasattr(args, 'adjdata'):
            from MTGNN.util import load_adj
            predefined_A = load_adj(args.adjdata)
            predefined_A = torch.tensor(predefined_A) - torch.eye(args.num_nodes)
            predefined_A = predefined_A.to(args.device)
        model = gtnet(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                      args.device, predefined_A=predefined_A,
                      dropout=args.dropout, subgraph_size=args.subgraph_size,
                      node_dim=args.node_dim,
                      dilation_exponential=args.dilation_exponential,
                      conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                      skip_channels=args.skip_channels, end_channels=args.end_channels,
                      seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                      layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=True)
    elif SDGDN in args.model:
        Network = getattr(import_module('model.%s' % args.model), args.model)
        model = Network(args)
    else:
        raise ValueError
    return model
