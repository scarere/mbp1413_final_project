# Experiment Details

This experiment tested how the magnitude of rotation affects model performance.

At the beginning of each train step, a set of random angles, A = {a_1, ..., a_B},
where B is the batch size, is selected from the interval -r < a < r
where r is defined by the rotation_angle in train_args. Then each sample is rotated
by it's corresponding angle in A. Hence each sample is rotated by a different
random angle.
