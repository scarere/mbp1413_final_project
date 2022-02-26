import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain
from sklearn.model_selection import train_test_split

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

    def train_step(self, inputs, labels, optimizer, criterion, batch_size, unet):
        optimizer.zero_grad()
        # photometric transformations
        bcsh = np.random.random(4)
        #     r = np.random.random()
        #     if r < 0.33:
        #         transformed_inputs = torchvision.transforms.RandomInvert(p=1.0)(batch_train_x)
        #     elif r > 0.33 and r < 0.66:
        #         transformed_inputs = torchvision.transforms.ColorJitter(brightness=bcsh[0], contrast=bcsh[1], saturation=bcsh[2])(batch_train_x)
        #     else:
        # transformed_inputs = torchvision.transforms.ColorJitter(brightness=bcsh[0], contrast=bcsh[1], saturation=bcsh[2])(batch_train_x)
        #     transformed_inputs = torchvision.transforms.RandomInvert(p=0.5)(transformed_inputs)
        # geometric transformations
        # rotated = torchvision.transforms.functional.affine(rotated, angle=-0, translate=[0, 0], scale = 1.0, shear=[0])
        #     random_angle = -3 + np.random.random()*6
        #     transformed_inputs = torchvision.transforms.functional.rotate(transformed_inputs, angle=random_angle)
        # rotated_labels = torchvision.transforms.functional.rotate(torch.unsqueeze(labels, 1), angle=random_angle)
        outputs = unet(inputs)
        # rotated_outputs = unet(transformed_inputs)
        # equivariance_loss = torch.nn.MSELoss()(rotated_outputs, torchvision.transforms.functional.rotate(outputs, angle=random_angle))
        outputs = outputs.permute(0, 2, 3, 1)
        # outputs.shape =(batch_size, img_cols, img_rows, n_classes)
        # outputs = outputs.reshape(batch_size*width_out*height_out, 1)
        # labels = labels.reshape(batch_size*width_out*height_out)
        #     rotated_outputs = outputs.reshape(batch_size*width_out*height_out, 1)
        #     rotated_labels = labels.reshape(batch_size*width_out*height_out)
        # loss = criterion(outputs.float(), labels.float()) # + equivariance_loss
        loss = criterion(outputs.float(), labels.float()) # F.cross_entropy
        loss.backward()
        optimizer.step()
        return loss

    def train_step_hybrid(self, inputs, labels, optimizer, criterion, batch_size, unet, hybrid_loss):
        optimizer.zero_grad()
        # photometric transformations
        bcsh = np.random.random(4)
        #     r = np.random.random()
        #     if r < 0.33:
        #         transformed_inputs = torchvision.transforms.RandomInvert(p=1.0)(batch_train_x)
        #     elif r > 0.33 and r < 0.66:
        #         transformed_inputs = torchvision.transforms.ColorJitter(brightness=bcsh[0], contrast=bcsh[1], saturation=bcsh[2])(batch_train_x)
        #     else:
        # transformed_inputs = torchvision.transforms.ColorJitter(brightness=bcsh[0], contrast=bcsh[1], saturation=bcsh[2])(batch_train_x)
        #     transformed_inputs = torchvision.transforms.RandomInvert(p=0.5)(transformed_inputs)
        # geometric transformations
        # rotated = torchvision.transforms.functional.affine(rotated, angle=-0, translate=[0, 0], scale = 1.0, shear=[0])
        #     random_angle = -3 + np.random.random()*6
        #     transformed_inputs = torchvision.transforms.functional.rotate(transformed_inputs, angle=random_angle)
        # rotated_labels = torchvision.transforms.functional.rotate(torch.unsqueeze(labels, 1), angle=random_angle)
        outputs, outputs2 = unet(inputs)
        # rotated_outputs = unet(transformed_inputs)
        # equivariance_loss = torch.nn.MSELoss()(rotated_outputs, torchvision.transforms.functional.rotate(outputs, angle=random_angle))
        outputs = outputs.permute(0, 2, 3, 1)
        outputs2 = outputs2.permute(0, 2, 3, 1)
        # outputs.shape =(batch_size, img_cols, img_rows, n_classes)
        # outputs = outputs.reshape(batch_size*width_out*height_out, 1)
        # labels = labels.reshape(batch_size*width_out*height_out)
        #     rotated_outputs = outputs.reshape(batch_size*width_out*height_out, 1)
        #     rotated_labels = labels.reshape(batch_size*width_out*height_out)
        # loss = criterion(outputs.float(), labels.float()) # + equivariance_loss
        loss1 = criterion(outputs.float(), labels.float()) # F.cross_entropy
        loss2 = criterion(outputs2.float(), labels.float())
        loss = hybrid_loss(loss1, loss2)
        loss.backward()
        optimizer.step()
        return loss

    def get_val_loss(self, x_val, y_val, criterion, unet):
        batch_size=self.batch_size
        epoch_iter = np.ceil(x_val.shape[0] / batch_size).astype(int)
        for _ in range(epoch_iter):
            total_loss = 0
            for i in range(epoch_iter):
                batch_val_x = torch.from_numpy(x_val[i * batch_size : (i + 1) * batch_size]).float()
                batch_val_x = batch_val_x/torch.max(batch_val_x.flatten(1),axis=1)[0].reshape(batch_val_x.shape[0],1,1,1)
                batch_val_y = torch.from_numpy(y_val[i * batch_size : (i + 1) * batch_size].astype(int)).long()
                if self.use_gpu:
                    batch_val_x = batch_val_x.cuda()
                    batch_val_y = batch_val_y.cuda()
                m = batch_val_x.shape[0]
                outputs = unet(batch_val_x)
                outputs = outputs.permute(0, 2, 3, 1)
                # outputs.shape =(batch_size, img_cols, img_rows, n_classes)
                # outputs = outputs.reshape(m*width_out*height_out, 1)
                # outputs2 = outputs2.reshape(m*width_out*height_out, 1)
                # labels = batch_val_y.reshape(m*width_out*height_out)
                loss = criterion(outputs.float(), batch_val_y.float()) # F.cross_entropy
                total_loss += loss.data
        return total_loss / epoch_iter

    def get_val_loss_hybrid(self, x_val, y_val, criterion, unet, hybrid_loss):
        batch_size=self.batch_size
        epoch_iter = np.ceil(x_val.shape[0] / batch_size).astype(int)
        for _ in range(epoch_iter):
            total_loss = 0
            for i in range(epoch_iter):
                batch_val_x = torch.from_numpy(x_val[i * batch_size : (i + 1) * batch_size]).float()
                batch_val_x = batch_val_x/torch.max(batch_val_x.flatten(1),axis=1)[0].reshape(batch_val_x.shape[0],1,1,1)
                batch_val_y = torch.from_numpy(y_val[i * batch_size : (i + 1) * batch_size].astype(int)).long()
                if self.use_gpu:
                    batch_val_x = batch_val_x.cuda()
                    batch_val_y = batch_val_y.cuda()
                m = batch_val_x.shape[0]
                outputs, outputs2 = unet(batch_val_x)
                outputs = outputs.permute(0, 2, 3, 1)
                outputs2 = outputs2.permute(0, 2, 3, 1)
                # outputs.shape =(batch_size, img_cols, img_rows, n_classes)
                # outputs = outputs.reshape(m*width_out*height_out, 1)
                # outputs2 = outputs2.reshape(m*width_out*height_out, 1)
                # labels = batch_val_y.reshape(m*width_out*height_out)
                loss1 = criterion(outputs.float(), batch_val_y.float()) # F.cross_entropy
                loss2 = criterion(outputs2.float(), batch_val_y.float())
                total_loss += hybrid_loss(loss1, loss2).data
        return total_loss / epoch_iter