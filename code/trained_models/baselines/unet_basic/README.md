# Unet Baseline Models

Note that unet_basic_tests 1 to 5 did not have binary masks during training
due to the interpolation of the masks. Therefore their loss and metrics can not
be relied on as they were calculated using non-binary masks.
