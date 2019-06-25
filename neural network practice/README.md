## R Implementation of Neural Network

#### Overview
- R implementation of Neural Network. After loading and preprocessing data, set parameters and run.

- The code supports CPU multi-core processing using doSNOW package. **For Linux users please don't forget to change cluster_type to "FORK"** (default = "SOCK")

#### R Package Requirements
- rhdf5 from R package "BiocManager"
- doSNOW, parallel, caret, abind, raster, grid, and tictoc

#### Files
- image_processing_using_neural_network.R - main process codes
- neural_network.R - contains functions
- train_catvnoncat.h5 - train set
- test_catvnoncat.h5 - test set
