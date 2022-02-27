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

class Training_Validation():
    def __init__(self,ishybrid=False,power=400,swap_coeffs=False,is_train_data_aug=False,is_apply_color_jitter=False):
        super(Training_Validation,self).__init__()
        self.ishybrid = ishybrid
        self.power = power
        self.swap_coeffs = swap_coeffs
        self.is_train_data_aug = is_train_data_aug
        self.is_apply_color_jitter = is_apply_color_jitter

    def hybrid_loss(self, loss_1, loss_2):
        min_vals = torch.minimum(loss_1,loss_2)
        coeffs_1 = torch.sin(1.57*min_vals/loss_1)**self.power
        coeffs_2 = torch.sin(1.57*min_vals/loss_2)**self.power
        coeffs_1 = coeffs_1/(coeffs_1 + coeffs_2)
        coeffs_2 = coeffs_2/(coeffs_1 + coeffs_2)
        if self.swap_coeffs == True:
            coeffs_1 = coeffs_1 + coeffs_2
            coeffs_2 = coeffs_1 - coeffs_2
            coeffs_1 = coeffs_1 - coeffs_2
        return (coeffs_1.detach()*loss_1 + coeffs_2.detach()*loss_2).mean()

    def train_valid(self, unet, x_train , y_train, x_val, y_val, optimizer, criterion, batch_size, dtv, use_gpu, epochs = 100, epoch_lapse = 20):
        if self.ishybrid == True:
          epoch_iter = np.ceil(x_train.shape[0] / batch_size).astype(int)
          t = trange(epochs, leave=True)
          train_losses = []
          val_losses = []
          for _ in t:
              total_loss = 0
              for i in range(epoch_iter):
                  batch_train_x = torch.from_numpy(x_train[i * batch_size : (i + 1) * batch_size]).float()
                  batch_train_x = batch_train_x/torch.max(batch_train_x.flatten(1),axis=1)[0].reshape(batch_train_x.shape[0],1,1,1)
                  batch_train_y = torch.from_numpy(y_train[i * batch_size : (i + 1) * batch_size].astype(int)).long()
                  if self.is_train_data_aug == True:
                      batch_train_x, batch_train_y = DataAugmentation(apply_color_jitter=self.is_apply_color_jitter)(batch_train_x, batch_train_y)
                  if use_gpu:
                      batch_train_x = batch_train_x.cuda()
                      batch_train_y = batch_train_y.cuda()
                  batch_loss = dtv.train_step_hybrid(batch_train_x , batch_train_y, optimizer, criterion, batch_train_x.shape[0], unet, self.hybrid_loss)
                  total_loss += batch_loss
              train_losses.append(total_loss.cpu().detach().numpy() / epoch_iter)
              if (_+1) % epoch_lapse == 0:
                  print('Train Loss at Epoch '+str(_+1)+': ', train_losses[-1].item())
              # val_loss = get_val_loss(x_val, y_val)
              # val_losses.append(val_loss)
              # if (_+1) % epoch_lapse == 0:
              #     print(f"Total loss in epoch {_+1} : {total_loss / epoch_iter} and validation loss : {val_loss}")
          return train_losses, None
        else:
          epoch_iter = np.ceil(x_train.shape[0] / batch_size).astype(int)
          t = trange(epochs, leave=True)
          train_losses = []
          val_losses = []
          for _ in t:
              total_loss = 0
              for i in range(epoch_iter):
                  batch_train_x = torch.from_numpy(x_train[i * batch_size : (i + 1) * batch_size]).float()
                  batch_train_x = batch_train_x/torch.max(batch_train_x.flatten(1),axis=1)[0].reshape(batch_train_x.shape[0],1,1,1)
                  batch_train_y = torch.from_numpy(y_train[i * batch_size : (i + 1) * batch_size].astype(int)).long()
                  if self.is_train_data_aug == True:
                      batch_train_x, batch_train_y = DataAugmentation(apply_color_jitter=self.is_apply_color_jitter)(batch_train_x, batch_train_y)
                  if use_gpu:
                      batch_train_x = batch_train_x.cuda()
                      batch_train_y = batch_train_y.cuda()
                  batch_loss = dtv.train_step(batch_train_x , batch_train_y, optimizer, criterion, batch_train_x.shape[0], unet)
                  total_loss += batch_loss
              train_losses.append(total_loss.cpu().detach().numpy() / epoch_iter)
              val_loss = dtv.get_val_loss(x_val, y_val, criterion, unet)
              val_losses.append(val_loss.cpu().detach().numpy())
              if (_+1) % epoch_lapse == 0:
                  print(f"Total loss in epoch {_+1} : {total_loss / epoch_iter} and validation loss : {val_loss}") 
          return train_losses, val_losses    