import os
#from re import I
import sys
# Add filepath to sys so that below imports work when this file is called
# from another directory
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import trange
import numpy as np
from model_utils import *
from loss_utils import *
from augmentation_utils import DataAugmentation
from segment_metrics import IOU_eval
import time

class Training_Validation():
    def __init__(self,ishybrid=False,power=400,swap_coeffs=False,is_train_data_aug=False,is_apply_color_jitter=False):
        super(Training_Validation,self).__init__()
        self.ishybrid = ishybrid
        self.power = power
        self.swap_coeffs = swap_coeffs
        self.is_train_data_aug = is_train_data_aug
        self.is_apply_color_jitter = is_apply_color_jitter

    def train_valid(self, unet, train_data, optimizer, criterion, batch_size, dtv, use_gpu, epochs, scheduler=None, metrics = None, validation_data=None, epoch_lapse = 1, progress_bar=True):
        x_train, y_train = train_data
        if not torch.is_tensor(x_train):
            x_train = torch.Tensor(x_train, dtype=torch.float32)
            y_train = torch.Tensor(y_train, dtype=torch.float32)
        epoch_iter = np.ceil(x_train.shape[0] / batch_size).astype(int)

        if progress_bar:
            t = trange(epochs, leave=True)
        else:
            t = range(epochs)

        train_metrics = {'epoch': [], 'loss': [], 'lr': []}
        val_metrics = {'epoch': [], 'val_loss': []}
        for metric in metrics:
            train_metrics[metric] = []
            val_metrics['val_' + metric] = []

        for _ in t:
            ts = time.time()
            metrics_sum = dict((metric, 0) for metric in metrics)
            metrics_sum['loss'] = 0
            for i in range(epoch_iter):
                batch_train_x = x_train[i * batch_size : (i + 1) * batch_size]
                batch_train_y = y_train[i * batch_size : (i + 1) * batch_size]
                if use_gpu:
                    batch_train_x = batch_train_x.cuda()
                    batch_train_y = batch_train_y.cuda()
                if self.ishybrid:
                    batch_metrics = dtv.train_step_hybrid(batch_train_x , batch_train_y, optimizer, criterion, unet, metrics=metrics)
                else:
                    batch_metrics = dtv.train_step(batch_train_x , batch_train_y, optimizer, criterion, unet, metrics=metrics)
                
                for key in batch_metrics.keys():
                    metrics_sum[key] += batch_metrics[key]
            te = time.time() - ts
            shuffle = torch.randperm(x_train.shape[0])
            x_train = x_train[shuffle]
            y_train = y_train[shuffle]
            epoch_metrics = {}
            train_metrics['epoch'].append(_+1)
            train_metrics['lr'].append(scheduler.get_last_lr())
            scheduler.step()
            for key in metrics_sum.keys():
                epoch_metrics[key] = metrics_sum[key] / epoch_iter
                train_metrics[key].append(epoch_metrics[key])

            if (_+1) % epoch_lapse == 0:
                
                string = 'Epoch %.0f/%.0f -- %.0fs, %.3fs/step - ' % (_+1, epochs, te, te/epoch_iter)
                string = string + ', '.join([ ' ' + metric + ': %.4f' % train_metrics[metric][-1] for metric in metrics_sum.keys()])
                if validation_data is not None:
                    x_val, y_val = validation_data
                    if not torch.is_tensor(x_val):
                        x_val = torch.Tensor(x_val, dtype=torch.float32)
                        y_val = torch.Tensor(y_val, dtype=torch.float32)
                    if self.ishybrid:
                        val = dtv.get_val_metrics_hybrid(x_val, y_val, criterion, unet, metrics)
                    else:
                        val = dtv.get_val_metrics(x_val, y_val, criterion, unet, metrics)

                    for key in val.keys():
                        val_metrics[key].append(val[key])
                    val_metrics['epoch'].append(_+1)
                    string = string + ' - ' + ', '.join([ ' val_' + metric + ': %.4f' % val_metrics['val_'+metric][-1] for metric in metrics_sum.keys()])
                print(string, flush=True)

        return train_metrics, val_metrics
        
           