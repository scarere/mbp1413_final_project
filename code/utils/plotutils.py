# importing all the necessary packages
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F  

# importing the style package
from matplotlib import style
  
# using the style for the plot
# plt.style.use('seaborn-pastel')
  
# creating plot

class Allplots():
    def __init__(self):
        super(Allplots,self).__init__()

    def plot_losses(self, train_losses, val_losses):
        plt.plot(train_losses, linestyle="-", linewidth=5, label='train')
        plt.plot(val_losses, linestyle="-", linewidth=5, label = 'validation')
        plt.legend()
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
          
        # show plot
        plt.show()

    def plot_examples(self, unet, datax, datay, num_examples=3, ishybrid=False):
        fig, ax = plt.subplots(nrows=num_examples, ncols=3, figsize=(18,4*num_examples))
        m = datax.shape[0]
        if ishybrid == True:
          for row_num in range(num_examples):
              image_indx = np.random.randint(m)
              image_arr,_ = unet(torch.from_numpy(datax[image_indx:image_indx+1]).float().cuda())
              image_arr = image_arr.squeeze(0).detach().cpu().numpy()
              ax[row_num][0].imshow(np.transpose(datax[image_indx], (1,2,0))[:,:,:].astype(int))
              ax[row_num][0].set_title("Orignal Image")
              ax[row_num][1].imshow(np.squeeze((image_arr > 0.50)[0,:,:].astype(int)),cmap='gray')
              ax[row_num][1].set_title("Segmented Image localization")
              ax[row_num][2].imshow(np.transpose(datay[image_indx], (1,2,0))[:,:,0],cmap='gray')
              ax[row_num][2].set_title("Target image")
        else:
          for row_num in range(num_examples):
            image_indx = np.random.randint(m)
            image_arr = unet(torch.from_numpy(datax[image_indx:image_indx+1]).float().cuda())
            image_arr = image_arr.squeeze(0).detach().cpu().numpy()
            ax[row_num][0].imshow(np.transpose(datax[image_indx], (1,2,0))[:,:,:].astype(int))
            ax[row_num][0].set_title("Orignal Image")
            ax[row_num][1].imshow(np.squeeze((image_arr > 0.50)[0,:,:].astype(int)),cmap='gray')
            ax[row_num][1].set_title("Segmented Image localization")
            ax[row_num][2].imshow(np.transpose(datay[image_indx], (1,2,0))[:,:,0],cmap='gray')
            ax[row_num][2].set_title("Target image")
        plt.show()

    def plot_best(self, unet, datax, datay=None, indx=None, index_ranks=None, ishybrid=False):
        if ishybrid == True:
          if datay is not None:
            num_examples = len(indx)
            fig, ax = plt.subplots(nrows=num_examples, ncols=3, figsize=(18,4*num_examples))
            m = datax.shape[0]
            for row_num in range(num_examples):
                image_indx = indx[row_num]
                if index_ranks[row_num] == 0:
                  image_arr, _ = unet(torch.from_numpy(datax[image_indx:image_indx+1]).float().cuda())
                elif index_ranks[row_num] == 1:
                  _, image_arr = unet(torch.from_numpy(datax[image_indx:image_indx+1]).float().cuda())
                image_arr = image_arr.squeeze(0).detach().cpu().numpy()
                ax[row_num][0].imshow(np.transpose(datax[image_indx], (1,2,0))[:,:,:].astype(int))
                ax[row_num][0].set_title("Orignal Image")
                ax[row_num][1].imshow(np.squeeze((image_arr > 0.50)[0,:,:].astype(int)),cmap='gray')
                ax[row_num][1].set_title("Segmented Image localization")
                ax[row_num][2].imshow(np.transpose(datay[image_indx], (1,2,0))[:,:,0],cmap='gray')
                ax[row_num][2].set_title("Target image")
            plt.show()
          
          else:
            num_examples = len(indx)
            fig, ax = plt.subplots(nrows=num_examples, ncols=2, figsize=(12,4*num_examples))
            m = datax.shape[0]
            for row_num in range(num_examples):
                image_indx = indx[row_num]
                if index_ranks[row_num] == 0:
                  image_arr, _ = unet(torch.from_numpy(datax[image_indx:image_indx+1]).float().cuda())
                elif index_ranks[row_num] == 1:
                  _, image_arr = unet(torch.from_numpy(datax[image_indx:image_indx+1]).float().cuda())
                image_arr = image_arr.squeeze(0).detach().cpu().numpy()
                ax[row_num][0].imshow(np.transpose(datax[image_indx], (1,2,0))[:,:,:].astype(int))
                ax[row_num][0].set_title("Orignal Image")
                ax[row_num][1].imshow(np.squeeze((image_arr > 0.50)[0,:,:].astype(int)),cmap='gray')
                ax[row_num][1].set_title("Segmented Image localization")
            plt.show()

        else:
            if datay is not None:
              num_examples = len(indx)
              fig, ax = plt.subplots(nrows=num_examples, ncols=3, figsize=(18,4*num_examples))
              m = datax.shape[0]
              for row_num in range(num_examples):
                  image_indx = indx[row_num]
                  image_arr = unet(torch.from_numpy(datax[image_indx:image_indx+1]).float().cuda())
                  image_arr = image_arr.squeeze(0).detach().cpu().numpy()
                  ax[row_num][0].imshow(np.transpose(datax[image_indx], (1,2,0))[:,:,:].astype(int))
                  ax[row_num][0].set_title("Orignal Image")
                  ax[row_num][1].imshow(np.squeeze((image_arr > 0.50)[0,:,:].astype(int)),cmap='gray')
                  ax[row_num][1].set_title("Segmented Image localization")
                  ax[row_num][2].imshow(np.transpose(datay[image_indx], (1,2,0))[:,:,0],cmap='gray')
                  ax[row_num][2].set_title("Target image")
              plt.show()
            
            else:
              num_examples = len(indx)
              fig, ax = plt.subplots(nrows=num_examples, ncols=2, figsize=(12,4*num_examples))
              m = datax.shape[0]
              for row_num in range(num_examples):
                  image_indx = indx[row_num]
                  image_arr = unet(torch.from_numpy(datax[image_indx:image_indx+1]).float().cuda())
                  image_arr = image_arr.squeeze(0).detach().cpu().numpy()
                  ax[row_num][0].imshow(np.transpose(datax[image_indx], (1,2,0))[:,:,:].astype(int))
                  ax[row_num][0].set_title("Orignal Image")
                  ax[row_num][1].imshow(np.squeeze((image_arr > 0.50)[0,:,:].astype(int)),cmap='gray')
                  ax[row_num][1].set_title("Segmented Image localization")
              plt.show()
