## R Implementation of Batch Normalization with Adam Optimizer

#### Overview
- R implementation of batch normalization, adam optimizer using mini batch gradient descent. After loading and preprocessing data, set parameters and run.

- The code supports CPU multi-core processing using doSNOW package. **For Linux users please don't forget to change cluster_type to "FORK"** (default = "SOCK")

#### R Package Requirements
- rhdf5 from R package "BiocManager"
- doSNOW, parallel, caret, abind, raster, grid, and tictoc

#### Files
- image_processing_using_BN_Adam.R - main process codes
- BN_with_Adam.R - contains functions
- train_signs.h5 - train set
-  test_signs.h5 - test set
