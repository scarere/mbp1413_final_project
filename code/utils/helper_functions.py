import os
import sys
# Add filepath to sys so that below imports work when this file is called
# from another directory
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import torch
from os.path import join
from model_utils import UNet, AttU_Net, Hybrid_Net
import numpy as np

def scale_norm(data):
    data = data/torch.max(torch.flatten(data, 1, -1))
    return data

def load_data_old(data_dir, print_shape=False, return_channel_counts=False):
    # Load data
    x_train = scale_norm(torch.tensor(torch.load(join(data_dir, 'split/x_train_split.pt'))))
    x_val = scale_norm(torch.tensor(torch.load(join(data_dir, 'split/x_val_split.pt'))))
    y_train = scale_norm(torch.tensor(torch.load(join(data_dir, 'split/y_train_split.pt'))))
    y_val = scale_norm(torch.tensor(torch.load(join(data_dir, 'split/y_val_split.pt'))))

    if print_shape:
        print('x_train: ', x_train.shape)
        print('y_train: ', y_train.shape)
        print('x_val: ', x_val.shape)
        print('y_val: ', y_val.shape)

    if return_channel_counts:
        CHANNEL_DIM=1
        in_chan = x_train.shape[CHANNEL_DIM]
        out_chan = y_train.shape[CHANNEL_DIM]
        return x_train, y_train, x_val, y_val, in_chan, out_chan
    else:
        return x_train, y_train, x_val, y_val

def load_data(data_dir, print_shape=False, switch_channel_dim=True):
    train = torch.load(os.path.join(data_dir, 'train.pt'))
    val = torch.load(os.path.join(data_dir, 'val.pt'))
    x_train = scale_norm(torch.tensor(np.stack(train['image'].to_numpy()), dtype=torch.float32))
    y_train = torch.unsqueeze(torch.tensor(np.stack(train['mask'].to_numpy()), dtype=torch.float32), -1)
    x_val = scale_norm(torch.tensor(np.stack(val['image'].to_numpy()), dtype=torch.float32))
    y_val = torch.unsqueeze(torch.tensor(np.stack(val['mask'].to_numpy()), dtype=torch.float32), -1)

    if switch_channel_dim:
        x_train = x_train.permute([0, 3, 1, 2])
        y_train = y_train.permute([0, 3, 1, 2])
        x_val = x_val.permute([0, 3, 1, 2])
        y_val = y_val.permute([0, 3, 1, 2])

    if print_shape:
        print('x_train: ', x_train.shape)
        print('y_train: ', y_train.shape)
        print('x_val: ', x_val.shape)
        print('y_val: ', y_val.shape)
    return x_train, y_train, x_val, y_val

def select_model(ishybrid, attn_type, in_chan=3, out_chan=1, use_gpu=False):
    '''Returns a pytorch model given a few arguments
    There are 3 model types:
        - Basic Unets
        - Unets with attention
        - Hybrids between Unets with attention and basic Unets
    Args:
        ishybrid (bool): If true, return a model that is a hybrid between
            a regular u-net and a u-net with attention
        attn_type: The type of attention to use. Either cosine, regular_full, 
            regular_full_dim_added, channel_attention or None, If None, 
            returns a regular U-Net
    '''
    if attn_type is None:
        model = UNet(in_channel=in_chan, out_channel=out_chan)
    elif ishybrid:
        model = Hybrid_Net(img_ch=in_chan, output_ch=out_chan, attn_type=attn_type)
    else:
        model = AttU_Net(img_ch=in_chan, output_ch=out_chan, attn_type=attn_type)

    if use_gpu:
        model = model.float().cuda()

    return model
        


