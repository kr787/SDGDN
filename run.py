import os
import sys
import argparse
import configparser
from datetime import datetime

file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(file_dir)
print('appended "', file_dir, '" to sys.path')

import torch
import numpy as np
import torch.nn as nn

from exp.basic_trainer import Trainer
from exp.data_factory import get_dataloader
from exp.args_factory import get_args
from exp.model_factory import get_model
from exp.learner_factory import get_learner
from exp.costs import get_model_parameters

args = get_args()

model = get_model(args).to(args.device)

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    else:
        nn.init.uniform_(p)

if args.mode == 'train':
    get_model_parameters(model, only_num=False)

if args.mode == 'save_data':
    _ = get_dataloader(args, normalizer='None', tod=args.tod, dow=args.dow, weather=False,
                       single=args.output_window == 1, save=True)
    exit()

# load dataset
train_loader, val_loader, test_loader, scaler = get_dataloader(args,
                                                               normalizer=args.normalizer,
                                                               tod=args.tod, dow=args.dow,
                                                               weather=False, single=args.output_window == 1)

loss, optimizer, lr_scheduler = get_learner(model, args)

# config log path
current_time = datetime.now().strftime('%Y%m%d%H%M%S')
current_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(current_dir, 'experiments', args.dataset, current_time)
args.log_dir = log_dir

# start training
trainer = Trainer(model, loss, optimizer, train_loader, val_loader, test_loader, scaler,
                  args, lr_scheduler=lr_scheduler)

if 'train' in args.mode:
    trainer.train()
    logs = trainer.test(model, trainer.args, test_loader, scaler)
    metrics = logs[-1].replace('\t', '_').replace('%', '')
    torch.save(model, 'exp/trained/%s_%s_%s.pth' % (args.model, args.dataset, metrics))
    # torch.save(model.state_dict(), 'exp/trained/%s_%s_%s.pth' % (args.model, args.dataset, metrics))
elif 'test' in args.mode:
    if 'week' in args.mode:
        model_name = args.model
        if args.woA: model_name = 'woA'
        if args.woO: model_name = 'woO'
        if args.woS: model_name = 'woS'
        if args.woT: model_name = 'woT'
        if args.woV: model_name = 'woV'
        trained_dir = os.path.join('.', 'exp', model_name)
        trained = None
        for file_name in os.listdir(trained_dir):
            if args.dataset in file_name:
                trained = file_name
        if trained is None:
            raise ValueError(f'no trained model found for dataset {args.dataset} with model {model_name}')
        args.trained = os.path.join('.', 'exp', model_name, trained)

    if args.trained:
        trained_model = torch.load(args.trained).to(args.device)
        model = trained_model.to(args.device)
        print('Load saved model', args.trained)
    elif args.trained_dict:
        trained_dict = torch.load(args.trained_dict)
        model.load_state_dict(trained_dict)
        print('Load saved model state_dict', args.trained_dict)

    if 'week' in args.mode:
        if not hasattr(args, 'index_of_saturday'):
            raise ValueError(f'index_of_saturday not specified for dataset {args.dataset}')

        if 'weekend' in args.mode:
            weekdays = [(i + args.index_of_saturday) % 7 for i in range(0, 2)]
        elif 'weekday' in args.mode:
            weekdays = [(i + args.index_of_saturday) % 7 for i in range(2, 7)]

        _, _, temp_test_loader, _ = get_dataloader(args, tod=False, dow=True)
        week_idx = []
        for batch_idx, batch_data in enumerate(temp_test_loader):
            *data, target = batch_data
            if target[-1, 0, 0, -1] in weekdays:
                week_idx.append(batch_idx)
        metrics = trainer.test(model, trainer.args, test_loader, scaler, trainer.logger, week_idx=week_idx)
        with open('./exp/week.csv', 'a') as f:
            f.write('\t'.join([args.dataset, args.mode, args.model, metrics[-1]]) + '\n')
    else:
        trainer.test(model, trainer.args, test_loader, scaler, trainer.logger)

else:
    raise ValueError
