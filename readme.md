# Spatial-Temporal Dynamic Graph Diffusion Convolutional Network for Traffic Flow Forecasting

This repository is the original pytorch implementation for SDGDN (Spatial-Temporal Dynamic Graph Diffusion Convolutional Network
for Traffic Flow Forecasting).

# Project Structure:

* `cfg`: experiment settings

    + `cfg/{dataset}.conf`: hyperparameter settings for SDGDN

    + `cfg/common/{dataset}.conf`: common settings for all models

    + `cfg/{model}/{dataset}.conf`: hyperparameter settings for baseline {model}

  > some hyperparameter settings for baseline models are not config in `cfg/{model}/{dataset}.conf`, but hard-coded in `exp/args_factory.py`, depending on the original implementation.

* `data`: datasets for experiments

    + PEMSD4 PEMSD7 PEMSD8 from [STSGCN](https://github.com/Davidham3/STSGCN)

    + NYCTaxi from [STDN](https://github.com/tangxianfeng/STDN)

    + BJTaxi from [ST-MetaNet](https://github.com/panzheyi/ST-MetaNet)

* `exp`: some modules for experiments and the trained models

    + `exp/args_factory.py`: arrange the experiment settings and merge into args

    + `exp/basic_trainer.py`: trainer engine for experiments

    + `exp/data_factory.py`: prepare the datasets and scalers for experimets

    + `exp/learner_factory.py`: decide the loss, optimizer, lr_scheduler for experimets

    + `exp/model_factory.py`: initialize the models

    + `exp/SDGDN/SDGDN_{dataset}_{MAE}_{RMSE}_{MAPE}.pth`: state_dict of trained models

* `log`: directory for training logs

* `model`: implementation of SDGDN, including the original model and its variants for ablation experiments

# Requirements

requirements for SDGDN: Python 3.8.13, Pytorch 1.9.1, Numpy 1.23.1

> If you want to run the experiments for baseline models, some other requirements are needed depending on the original implementation.

> For some unknown reason, MTGNN and DGCRN can not achieve state-of-the-art results in this repo, so it is not recommended to run the experiments here.

# Get Started

Run the code in the form of `python run.py --device %s --model %s --mode %s --dataset %s`.

For example: `python run.py --device cuda:0 --model SDGDN --mode train --dataset PEMSD8`.

> If you are using linux shell, you may also use `python sh.py` as an alternative.



