import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import torch
import random
import math
import numpy as np


def rotate(images, masks, max_angle=0, base_angles=[0], crop=False):
    '''Rotates image and mask by random angle between -max_angle and max_angle

    Assumes last two dimensions of image and mask tensors are H,W, and first dim is
    batch size

    crop is the heigh
    '''
    output_images = torch.clone(images)
    output_masks = torch.clone(masks)
    for i in range(images.shape[0]):
        total_angle= random.choice(base_angles) + random.randint(-max_angle, max_angle)
        num_90 = round(total_angle/90) # round to get nearest axes to final position
        angle = total_angle - num_90*90 # rotation relative to nearest axis
        # rotate by intervals of 90 first to reduce interpolation
        image = torch.rot90(images[i], k=num_90, dims=[-2, -1])
        mask = torch.rot90(masks[i], k=num_90, dims=[-2, -1])
        # rotate the remaining amount
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)
        if crop:
            r_dim = rotatedRectWithMaxArea(image.shape[-1], image.shape[-2], total_angle) # new height and width
            k = np.argmin(r_dim) # if k = 0, hr is smaller, if k = 1 wr is smaller
            s = int(r_dim[k])# size of crop
            start = int((image.shape[k - 2] - s)/2)
            image = TF.resize(image[...,start:start+s, start:start+s], size=image.shape[-2:])
            mask = TF.resize(mask[...,start:start+s, start:start+s], size=mask.shape[-2:])
        output_images[i] = image
        output_masks[i] = mask > 0.5

    return output_images, output_masks

def rotate_batch(images, masks, max_angle=0, base_angles=[0]):
    total_angle= random.choice(base_angles) + random.randint(-max_angle, max_angle)
    num_90 = round(total_angle/90) # round to get nearest axis to final position
    angle = total_angle - num_90*90 # rotation relative to nearest axis
    # rotate by intervals of 90 first to reduce interpolation
    output_images = torch.rot90(images, k=num_90, dims=[-2, -1])
    output_masks = torch.rot90(masks, k=num_90, dims=[-2, -1])
    # Rotate the remaining amount
    output_images = TF.rotate(output_images, angle)
    output_masks = TF.rotate(output_masks, angle)
    return output_images, output_masks

def rotatedRectWithMaxArea(w, h, angle):
  """
  Given a rectangle of size wxh that has been rotated by 'angle' (in
  radians), computes the width and height of the largest possible
  axis-aligned rectangle (maximal area) within the rotated rectangle.
  """
  if w <= 0 or h <= 0:
    return 0,0

  width_is_longer = w >= h
  side_long, side_short = (w,h) if width_is_longer else (h,w)

  # since the solutions for angle, -angle and 180-angle are all the same,
  # if suffices to look at the first quadrant and the absolute values of sin,cos:
  sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
  if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
    # half constrained case: two crop corners touch the longer side,
    #   the other two corners are on the mid-line parallel to the longer line
    x = 0.5*side_short
    wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
  else:
    # fully constrained case: crop touches all 4 sides
    cos_2a = cos_a*cos_a - sin_a*sin_a
    wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

  return hr, wr
