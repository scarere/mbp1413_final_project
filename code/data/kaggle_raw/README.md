# Raw Kaggle Dataset

This folder contains publicly available data from the 2018 Kaggle Data Science Bowl Competition.

## Download Data

The original data from the competition can be downloaded [here](https://www.kaggle.com/c/data-science-bowl-2018/data)

The solutions to the final stage 2 test set are not provided in the data from the Kaggle competiton. Both the stage 2 solutions
as well as the original competition data can be obtained from the Broad Bioimage Benchmark Collection with
accession number [BBBC038](https://bbbc.broadinstitute.org/BBBC038/).

## Dataset Info

The run-length encoding for the masks are saved in csv files in order to save disk space.

There are three sets of data:

- ***stage1_train***
  - This was the original training data from the competition.
    Both the mask png files and their encodings were provided
- ***stage1_test***
  - This was a heldout test set that users could use to validation their
    models throughout the competition.
  - The masks were not provided, however
    users could make up to 5 submissions per day to see how their model
    performed on this dataset
  - The images in this dataset were not used in the stage1_train set
- ***stage2_test_final***
  - This dataset was used to determine the winners of the competition.
  - After the comptetion was over, all final submissions were scored using
    this dataset
  - This dataset specifically contains images from unseen experimental
    conditions in order to test model robustness.
  - The images in this dataset wer not used in the stage 1 datasets.

For our project we use the datasets as follows:

- ***Training***: stage1_train
- ***Validation***: stage1_test
- ***Testing***: stage2_test_final
