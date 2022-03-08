import pandas as pd
import numpy as np
import os
import torch

def null_fun(input):
  return input

class CreateMask():
    def __init__(self):
        super(CreateMask,self).__init__()

    def generate_masks(self, encoded_solutions_csv, iterator=null_fun):
      csv_file_name = encoded_solutions_csv
      rows = []
      df = pd.read_csv(csv_file_name)
      n_imgs = df[df.columns[0]].nunique()
      unique_IDs = df['ImageId'].unique()
      dimension_list = []
      for id in iterator(unique_IDs):
        if df.loc[df['ImageId'] == id]['Usage'].values[0] == 'Ignored':
          continue
        # get height and width of each unique image
        dimensions = (df.loc[df['ImageId'] == id]['Height'].values[0], df.loc[df['ImageId'] == id]['Width'].values[0])
        vals = df.loc[df['ImageId'] == id]['EncodedPixels'].values # list of encoded pixels
        final_vals = np.asarray([])
        for j in range(len(vals)):
          eps = vals[j].split(' ')
          for l in range(len(eps)):
            eps[l] = int(eps[l])
          final_vals = np.concatenate((final_vals, np.asarray(eps)))
        arr = np.matrix.flatten(np.zeros(dimensions))
        r = 0
        while r < len(final_vals):
          arr[int(final_vals[r]):int(final_vals[r]+final_vals[r+1])] = 1
          r = r + 2
        rows.append([id,arr.reshape(np.flip(dimensions)).transpose()])
      return pd.DataFrame(rows, columns=['ImageId', 'Mask'])
