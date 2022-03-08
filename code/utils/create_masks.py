import pandas as pd
import numpy as np
import os
import torch

class CreateMask():
    def __init__(self, csv_file_name='stage1_solution.csv'):
        super(CreateMask,self).__init__()
        self.csv_file_name = csv_file_name

    def generate_test_masks(self):
      csv_file_name = self.csv_file_name
      test_masks = []
      df = pd.read_csv(csv_file_name)
      n_imgs = df[df.columns[0]].nunique()
      unique_IDs = df['ImageId'].unique()
      dimension_list = []
      for i in range(n_imgs):
        # get height and width of each unique image
        dimension_list.append((df.loc[df['ImageId'] == unique_IDs[i]]['Height'].values[0], df.loc[df['ImageId'] == unique_IDs[i]]['Width'].values[0]))
        vals = df.loc[df['ImageId'] == unique_IDs[i]]['EncodedPixels'].values # list of encoded pixels
        final_vals = np.asarray([])
        for j in range(len(vals)):
          eps = vals[j].split(' ')
          for l in range(len(eps)):
            eps[l] = int(eps[l])
          final_vals = np.concatenate((final_vals, np.asarray(eps)))
        arr = np.matrix.flatten(np.zeros(dimension_list[i]))
        r = 0
        while r < len(final_vals):
          arr[int(final_vals[r]):int(final_vals[r]+final_vals[r+1])] = 1
          r = r + 2
        test_masks.append(arr.reshape(np.flip(dimension_list[i])).transpose())
      return test_masks

# sample run
# test_masks = CreateMask(csv_file_name='stage1_solution.csv').generate_test_masks()