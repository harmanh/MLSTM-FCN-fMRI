#!/usr/bin/env Rscript

# Author: Christopher J. Urban
# Date: 6/25/2018
#
# This script creates an 80-20 train-test split 
# on the ABIDE II data using:
# (1) ROI time series means,
# (2) ROI time series Pearson correlations, and
# (3) ROI time series dynamic time warping distances.

input.file = "../data/design_matrices/design_matrices.RData"

load(input.file)

set.seed(1)
train.idx = sample(1:length(subj.ids), 0.8 * length(subj.ids), replace = FALSE)

mean.train = mean.data[train.idx, ]
mean.test = mean.data[-train.idx, ]

cor.train = na.omit(cor.data[train.idx, ])
cor.test = na.omit(cor.data[-train.idx, ])

dtw.train = dtw.data[train.idx, ]
dtw.test = dtw.data[-train.idx, ]
