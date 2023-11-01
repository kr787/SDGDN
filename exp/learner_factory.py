import torch

import exp.metrics
from exp.model_factory import AGCRN, ASTGNN, DCRNN, GCDE, GWN, SCINET, HA, MTGNN


def get_learner(model, args):
    if args.loss_func == 'mask_mae':
        loss = lambda x, y: exp.metrics.MAE_torch(pred=x, true=y, mask_value=args.mae_thresh)
    elif args.loss_func == 'mae':
        loss = lambda x, y: exp.metrics.MAE_torch(pred=x, true=y)
    elif args.loss_func == 'mse':
        loss = lambda x, y: exp.metrics.MSE_torch(pred=x, true=y)
    else:
        raise ValueError

    if args.model == SCINET:
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                                     weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init,
                                     weight_decay=hasattr(args, 'weight_decay') and args.weight_decay or 0,
                                     amsgrad=False)

    lr_scheduler = None
    if args.model == SCINET:
        print('Applying ExponentialLR learning rate decay.')
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.decay_rate)
    elif args.model == DCRNN:
        print('Applying MultiStepLR learning rate decay.')
        lr_milestones = [int(i) for i in list(args.lr_milestones.split(','))]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=lr_milestones,
                                                            gamma=args.lr_decay_rate)
    elif 'SDGDN' in args.model:
        if hasattr(args, 'lr_decay') and args.lr_decay:
            print('Applying StepLR learning rate decay.')
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=args.lr_decay_rate,
                                                           step_size=args.cl_step)
        else:
            print('Learning rate needn\'t decay.')
    elif hasattr(args, 'lr_decay') and args.lr_decay and hasattr(args, 'lr_decay_step'):
        print('Applying MultiStepLR learning rate decay.')
        lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                            milestones=lr_decay_steps,
                                                            gamma=args.lr_decay_rate)
    else:
        print('Learning rate needn\'t decay.')

    return loss, optimizer, lr_scheduler
