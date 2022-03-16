from torchmetrics.classification.jaccard import JaccardIndex
import torch
import numpy as np

class IOU_eval():
    def __init__(self, ishybrid=False, num_classes=2):
        super(IOU_eval,self).__init__()
        self.iou = JaccardIndex(num_classes=num_classes) # automatically uses a threshold of 0.5 to convert predictions to binary integers
        self.num_classes = num_classes
        self.ishybrid = ishybrid

    def iou_evaluate(self, unet, datax, datay):
        iou_vals = []
        iou_indices = []
        m = datax.shape[0]
        if self.ishybrid == True:
          for image_indx in range(m):
              image_arr,image_arr2 = unet(torch.from_numpy(datax[image_indx]).float())
              image_arr = image_arr.squeeze(0).detach().cpu()
              image_arr2 = image_arr2.squeeze(0).detach().cpu()
              iou1 = self.iou(image_arr,torch.from_numpy(datay[image_indx]))
              iou2 = self.iou(image_arr2,torch.from_numpy(datay[image_indx]))
              iou_vals.append(0.5*torch.add(iou1,iou2).item())
              iou_indices.append(torch.argmax(torch.from_numpy(np.asarray([iou1,iou2]))).item())
        else:
          for image_indx in range(m):
              image_arr = unet(torch.from_numpy(datax[image_indx]).float().unsqueeze(0))
              image_arr = image_arr.squeeze(0).detach().cpu()
              iou1 = self.iou(image_arr,torch.from_numpy(datay[image_indx]))
              iou_vals.append(iou1.item())
              iou_indices.append(torch.argmax(torch.from_numpy(np.asarray([iou1,iou1]))).item())
        
        return iou_vals, iou_indices

    def iou_evaluate_better(self, y_true, y_pred):
        iou_vals = []
        for sample in range(y_pred.shape[0]):
            iou = self.iou(y_pred[sample].detach().cpu(), y_true[sample].detach().cpu()) # can't be computed on gpu for some reason
            iou_vals.append(iou)

        return torch.mean(torch.stack(iou_vals))
        

