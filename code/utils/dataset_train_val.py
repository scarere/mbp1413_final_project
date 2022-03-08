import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain
from sklearn.model_selection import train_test_split
from loss_utils import *
from segment_metrics import IOU_eval

class Dataset_train_val(nn.Module):
    def __init__(self,batch_size,use_gpu):
        super(Dataset_train_val,self).__init__()
        self.batch_size = batch_size
        self.use_gpu = use_gpu

    def load_train_val_test(self, x_train_name, y_train_name, x_test_name, split_ratio=0.2):
        x_train = torch.load(x_train_name) # 'x_train.pt'
        y_train = torch.load(y_train_name) # 'y_train.pt')
        x_test = torch.load(x_test_name) # 'x_test.pt')

        x_train, x_val, y_train, y_val =  train_test_split(x_train, y_train, test_size=split_ratio)
        return x_train, x_val, y_train, y_val, x_test

    def eval(self, metrics, y_true, y_pred):
        output = {}
        for metric in metrics:
            if metric == 'DICE':
                output['DICE'] = DiceBCELoss().dice_loss(y_true, y_pred).item()
            elif metric == 'BCE':
                output['BCE'] = DiceBCELoss().bce_loss(y_true, y_pred).item()
            elif metric == 'IoU':
                output['IoU'] = IOU_eval().iou_evaluate_better(y_true.int(), y_pred).item()
        return output

    def hybrid_loss(self, loss_1, loss_2):
        min_vals = torch.minimum(loss_1,loss_2)
        coeffs_1 = torch.sin((np.pi/2)*min_vals/loss_1)**self.power
        coeffs_2 = torch.sin((np.pi/2)*min_vals/loss_2)**self.power
        coeffs_1 = coeffs_1/(coeffs_1 + coeffs_2)
        coeffs_2 = coeffs_2/(coeffs_1 + coeffs_2)
        if self.swap_coeffs == True:
            coeffs_1 = coeffs_1 + coeffs_2
            coeffs_2 = coeffs_1 - coeffs_2
            coeffs_1 = coeffs_1 - coeffs_2
        return (coeffs_1.detach()*loss_1 + coeffs_2.detach()*loss_2).mean()

    def train_step(self, inputs, labels, optimizer, criterion, unet, metrics=None):
        optimizer.zero_grad()
        outputs = unet(inputs)
        outputs = outputs.permute(0, 2, 3, 1)
        labels = labels.permute(0, 2, 3, 1)
        loss = criterion(outputs, labels) # F.cross_entropy
        evals = self.eval(metrics, labels, outputs)
        evals['loss'] = loss.item()
        loss.backward()
        optimizer.step()
        return evals

    def train_step_hybrid(self, inputs, labels, optimizer, criterion, unet, hybrid_loss, metrics=None):
        optimizer.zero_grad()
        outputs, outputs2 = unet(inputs)
        outputs = outputs.permute(0, 2, 3, 1)
        outputs2 = outputs2.permute(0, 2, 3, 1)
        labels = labels.permute(0, 2, 3, 1)
        outputs_avg = 0.5*(outputs + outputs2)
        loss1 = criterion(outputs.float(), labels.float()) # F.cross_entropy
        loss2 = criterion(outputs2.float(), labels.float())
        loss = hybrid_loss(loss1, loss2)
        evals = self.eval(metrics, labels, outputs_avg)
        evals['loss'] = loss.item()
        loss.backward()
        optimizer.step()
        return evals

    def get_val_metrics(self, x_val, y_val, criterion, unet, metrics):
        batch_size=self.batch_size
        epoch_iter = np.ceil(x_val.shape[0] / batch_size).astype(int)
        total_loss = 0
        total_evals = dict(('val_' + metric, 0) for metric in metrics)
        for i in range(epoch_iter):
            batch_val_x = x_val[i * batch_size : (i + 1) * batch_size]
            batch_val_y =y_val[i * batch_size : (i + 1) * batch_size]
            if self.use_gpu:
                batch_val_x = batch_val_x.cuda()
                batch_val_y = batch_val_y.cuda()
            outputs = unet(batch_val_x)
            outputs = outputs.permute(0, 2, 3, 1)
            batch_val_y = batch_val_y.permute(0, 2, 3, 1)
            # outputs.shape =(batch_size, img_cols, img_rows, n_classes)
            evals = self.eval(metrics, batch_val_y.float(), outputs.float())
            loss = criterion(outputs.float(), batch_val_y.float()) # F.cross_entropy
            total_loss += loss.data
            for key in evals.keys():
                total_evals['val_' + key] += evals[key]
        for key in total_evals.keys():
            total_evals[key] = total_evals[key] / epoch_iter
        total_evals['val_loss'] = (total_loss / epoch_iter).item()
        return total_evals

    def get_val_metrics_hybrid(self, x_val, y_val, criterion, unet, metrics=None):
        batch_size=self.batch_size
        epoch_iter = np.ceil(x_val.shape[0] / batch_size).astype(int)
        total_loss = 0
        total_evals = {}
        for i in range(epoch_iter):
            batch_val_x = x_val[i * batch_size : (i + 1) * batch_size]
            batch_val_y = y_val[i * batch_size : (i + 1) * batch_size]
            if self.use_gpu:
                batch_val_x = batch_val_x.cuda()
                batch_val_y = batch_val_y.cuda()
            m = batch_val_x.shape[0]
            outputs, outputs2 = unet(batch_val_x)
            outputs = outputs.permute(0, 2, 3, 1)
            outputs2 = outputs2.permute(0, 2, 3, 1)
            batch_val_y = batch_val_y.permute(0, 2, 3, 1)
            # outputs.shape =(batch_size, img_cols, img_rows, n_classes)
            loss1 = criterion(outputs.float(), batch_val_y.float()) # F.cross_entropy
            loss2 = criterion(outputs2.float(), batch_val_y.float())
            total_loss += self.hybrid_loss(loss1, loss2).data
            outputs_avg = 0.5*(outputs + outputs2)
            evals = self.eval(metrics, batch_val_y.float(), outputs_avg.float())
            for key in evals.keys():
                total_evals['val_' + key] += evals[key]
        for key in total_evals.keys():
            total_evals[key] = total_evals[key] / epoch_iter
        total_evals['val_loss'] = total_loss / epoch_iter
        return total_evals