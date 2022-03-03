# Imports
from pickletools import optimize
from select import select
import config
import sys
import torch
from os.path import join
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

# Custom imports
sys.path.append(config.base_dir) # allow imports from base dir
from utils.dataset_train_val import Dataset_train_val
from utils.training_validation_utils import Training_Validation
from utils.model_utils import UNet, AttU_Net, Hybrid_Net
from utils.helper_functions import load_data, select_model
from utils.loss_utils import DiceBCELoss, DiceBCELossModified

save = True

model_args = {
    'name': 'unet_basic_test1',
    'ishybrid': False,
    'attn': None,
    'power': None,
    'swap_coeff': False
}

train_args = {
    'batch_size': 32,
    'epochs': 200,
    'initial_lr': 0.01,
    'lr_schedule': None,
}

if model_args['ishybrid']:
    model_args['power'] = 2.0

use_gpu = torch.cuda.is_available()

x_train, y_train, x_val, y_val, in_chan, out_chan = load_data(config.data_dir, return_channel_counts=True)
model = select_model(model_args['ishybrid'], model_args['attn'], in_chan=in_chan, out_chan=out_chan, use_gpu=use_gpu)

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
train_args['optimizer'] = {'type': optimizer.__class__, 'params': dict}
train_args['loss'] = loss.__class__

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

# Training
train_losses, val_losses = tv_class.train_valid(
    model,
    x_train, 
    y_train, 
    x_val, 
    y_val,
    dtv=dtv_class,
    optimizer=optimizer,
    criterion=loss,
    batch_size=train_args['batch_size'],
    use_gpu=use_gpu,
    epochs=train_args['epochs']
    )

if save:
    if model_args['ishybrid']:
        save_path = join(config.model_dir, 'baselines/unet_hybrid')
    elif model_args['attn'] == 'cosine':
        save_path = join(config.model_dir, 'baselines/unet_cos_attn')
    elif model_args['attn'] == 'channel_attention':
        save_path = join(config.model_dir, 'baselines/unet_channel_attn')
    elif model_args['attn'] is not None:
        save_path = join(config.model_dir, 'baselines/unet_pointwise_attn')
    else:
        save_path = join(config.model_dir, 'baselines/unet_basic')

    


