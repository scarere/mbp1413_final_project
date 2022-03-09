# Imports
from pickletools import optimize
from select import select

from tqdm import tqdm
import config
import sys
import torch
from os import path, makedirs
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from torchinfo import summary
import numpy as np

# Custom imports
sys.path.append(config.base_dir) # allow imports from base dir
from utils.dataset_train_val import Dataset_train_val
from utils.training_validation_utils import Training_Validation
from utils.model_utils import UNet, AttU_Net, Hybrid_Net
from utils.helper_functions import load_data, select_model, scale_norm
from utils.loss_utils import DiceBCELoss, DiceBCELossModified



save = True

model_args = {
    'name': 'unet_basic_test1',
    'attn': None,
    'ishybrid': False,
    'power': None,
    'swap_coeff': False
}

train_args = {
    'batch_size': 32,
    'epochs': 1,
    'initial_lr': 0.01,
    'lr_schedule': None,
}

parser = argparse.ArgumentParser()
parser.add_argument('-sp', '--save_path', default=path.join('trained_models/', model_args['name']), help='Path to save the model to')
parser.add_argument('-tq', '--tqdm', action='store_true', help='If included a progress bar is shown during training')
parser.add_argument('-el', '--epoch_lapse', default=20, help='The period of epochs to wait between model validations')
args = parser.parse_args()

if model_args['ishybrid']:
    model_args['power'] = 2.0

use_gpu = torch.cuda.is_available()
if use_gpu:
    print('Using: ', torch.cuda.get_device_name())

x_train, y_train, x_val, y_val = load_data(path.join(config.data_dir, 'undistorted'), print_shape=True, switch_channel_dim=True)

model = select_model(model_args['ishybrid'], model_args['attn'], in_chan=3, out_chan=1, use_gpu=use_gpu)

summ = summary(model, input_size=[1].append(x_train.shape[1:]), input_data=x_train[0:2], col_names=('input_size', 'output_size',  'kernel_size', 'num_params'))

optimizer = SGD(model.parameters(), lr=train_args['initial_lr'])

if train_args['lr_schedule'] is not None:
    milestones = []
    gamma = 0.1
    scheduler = MultiStepLR(optimizer, milestones, gamma)
    train_args['lr_schedule'] = {'gamma': gamma, 'milestones': milestones}

if model_args['ishybrid']:
    loss = DiceBCELossModified()
else:
    loss = DiceBCELoss()

# Add stuff to training_args dict
dict = optimizer.state_dict()['param_groups'][0].copy()
del dict['params']
train_args['optimizer'] = {'type': type(optimizer).__name__, 'params': dict}
train_args['loss'] = type(loss).__name__

'''
Due to the weird way Sayan structured the code when prototyping, we have
two classes with overlapping functionality that are used to define
all the training loops and so on. Hopefully we have time to restructure
the code into something that makes more sense. But for now we use the 
tools as they are
'''
# DTV class defines data loading, base training loops and val loss calculations
dtv_class = Dataset_train_val(train_args['batch_size'], use_gpu)

# TV takes DTV object as a parameter during training, also defines hybrid loss function
tv_class = Training_Validation(model_args['ishybrid'], power=model_args['power'], swap_coeffs=model_args['swap_coeff'])

if use_gpu:
    model = model.cuda()

# Training
train_metrics, val_metrics = tv_class.train_valid(
    unet=model,
    train_data=(x_train, y_train),
    validation_data=(x_val, y_val),
    dtv=dtv_class,
    optimizer=optimizer,
    criterion=loss,
    batch_size=train_args['batch_size'],
    use_gpu=use_gpu,
    epochs=train_args['epochs'],
    progress_bar=args.tqdm,
    epoch_lapse=int(args.epoch_lapse),
    metrics=['DICE', 'IoU']
    )

if save:
    print(path.isdir(args.save_path))
    if not path.isdir(args.save_path):
        makedirs(args.save_path)

    torch.save(model, path.join(args.save_path, model_args['name']+'.pt'))
    pd.DataFrame(val_metrics).to_csv(path.join(args.save_path, 'val_metrics.csv'))
    pd.DataFrame(train_metrics).to_csv(path.join(args.save_path, 'train_metrics.csv'))

    with open(path.join(args.save_path, 'train_args.yaml'), 'w') as file:
        yaml.dump(train_args, file)

    with open(path.join(args.save_path, 'model_args.yaml'), 'w') as file:
        yaml.dump(model_args, file)

    with open(path.join(args.save_path, model_args['name'] + '.txt'), 'w') as file:
        file.write(summ.__repr__())
    

    

    

    


