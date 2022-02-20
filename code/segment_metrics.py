from torchmetrics import IoU
import torch
import numpy as np

class IOU_eval():
    def __init__(self, ishybrid=False, num_classes=2):
        super(IOU_eval,self).__init__()
        self.iou = IoU(num_classes=num_classes)
        self.num_classes = num_classes
        self.ishybrid = ishybrid

    def iou_evaluate(self, unet, datax, datay):
        iou_vals = []
        iou_indices = []
        m = datax.shape[0]
        if self.ishybrid == True:
          for image_indx in range(m):
              image_arr,image_arr2 = unet(torch.from_numpy(datax[image_indx:image_indx+1]).float().cuda())
              image_arr = image_arr.squeeze(0).detach().cpu()
              image_arr2 = image_arr2.squeeze(0).detach().cpu()
              iou1 = self.iou(image_arr,torch.from_numpy(datay[image_indx:image_indx+1]))
              iou2 = self.iou(image_arr2,torch.from_numpy(datay[image_indx:image_indx+1]))
              iou_vals.append(torch.max(iou1,iou2).item())
              iou_indices.append(torch.argmax(torch.from_numpy(np.asarray([iou1,iou2]))).item())
        else:
          for image_indx in range(m):
              image_arr = unet(torch.from_numpy(datax[image_indx:image_indx+1]).float().cuda())
              image_arr = image_arr.squeeze(0).detach().cpu()
              iou1 = self.iou(image_arr,torch.from_numpy(datay[image_indx:image_indx+1]))
              iou_vals.append(iou1.item())
              iou_indices.append(torch.argmax(torch.from_numpy(np.asarray([iou1,iou1]))).item())
        
        return iou_vals, iou_indices

