import os
import sys
# Add filepath to sys so that below imports work when this file is called
# from another directory
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import torch
from os.path import join
from model_utils import UNet, AttU_Net, Hybrid_Net

def load_data(data_dir, print_shape=False, return_channel_counts=False):
    # Load data
    x_train = torch.load(join(data_dir, 'x_train_split.pt'))
    x_val = torch.load(join(data_dir, 'x_val_split.pt'))
    y_train = torch.load(join(data_dir, 'y_train_split.pt'))
    y_val = torch.load(join(data_dir, 'y_val_split.pt'))

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
        model = model.cuda()

    return model
