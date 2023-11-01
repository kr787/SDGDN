import datetime
import os
import time

from exp.model_factory import AGCRN, ASTGNN, DCRNN, DGCRN, DMSTGCN, GCDE, GWN, HA, MTGNN, SCINET, SDGDN
from exp.data_factory import PEMS04, PEMS07, PEMS08, NYCO, NYCD, BJO, BJD


def run_cmd(cmd):
    print(datetime.datetime.now(), cmd)
    os.system(cmd)


def run(cuda=0, data='PEMSD8', model='SDGDN', mode='train', redirect=True, background=True):
    cmd = 'python -u run.py --device cuda:%d --model %s --mode %s --dataset %s'

    # cmd += ' --woA'
    # cmd += ' --woO'
    # cmd += ' --woS'
    # cmd += ' --woT'
    # cmd += ' --woV'

    # cmd += ' --trained_dict ./exp/SDGDN/SDGDN_PEMSD4_18.38_30.63_12.16.pth'

    cmd = cmd % (cuda, model, mode, data)

    if redirect:
        now = time.strftime("%y-%m-%d_%H-%M-%S", time.localtime())
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        cmd += ' > %s/log/%s/%s.log 2>&1' % (cur_dir, data, now)

    if background:
        cmd += ' & cd .'
    else:
        cmd += ' && cd .'

    run_cmd(cmd)


if __name__ == '__main__':
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    models = [
        # AGCRN,
        # ASTGNN,
        # DCRNN,
        # DMSTGCN,
        # DGCRN,
        # GCDE,
        # GWN,
        # SCINET,
        # HA,
        # MTGNN,
        SDGDN,
        # SDGDN + 'ablation',
    ]

    datasets = [
        # PEMS04,
        # PEMS07,
        PEMS08,
        # NYCO,
        # NYCD,
        # BJO,
        # BJD
    ]

    modes = [
        # 'save_data',
        # 'train',
        'test',
        # 'test_weekday',
        # 'test_weekend',
        # 'train_cost',
        # 'test_cost',
    ]

    for m in models:
        for data in datasets:
            for mode in modes:
                run(
                    cuda=0,
                    data=data,
                    model=m,
                    redirect=False,
                    background=False,
                    mode=mode,
                )
