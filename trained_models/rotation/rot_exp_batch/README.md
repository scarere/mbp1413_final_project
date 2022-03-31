# Experiment Details

This experiment tested how the magnitude of rotation affects model performance.

At the beginning of each train step, a random angle, a, is selected from the
interval -r < a < r where r is defined by the rotation_angle in train_args.
The entire batch of samples is then rotated by same angle a.
