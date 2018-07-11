#!/usr/bin/env Rscript

# Author: Christopher J. Urban
# Date: 6/27/2018
#
# This script fits K-nearest neighbors regressions to the ABIDE II data.
# The following functional connectivity metrics are used:
# (1) ROI times series means,
# (2) ROI time series Pearson correlations, and
# (3) ROI time series dynamic time warping distances.

if(!require(doMC)) {
  install.packages("doMC", repos = "http://cran.us.r-project.org")
  library(doMC)
}

if(!require(caret)) {
  install.packages("caret", repos = "http://cran.us.r-project.org")
  library(caret)
}

if(!require(abind)) {
  install.packages("abind", repos = "http://cran.us.r-project.org")
  library(abind)
}

if(!require(e1071)) {                                                           
  install.packages("e1071", repos = "http://cran.us.r-project.org")             
  library(e1071)                                                                
}

# Register cores for parallel computations.
registerDoMC(cores = 4)

source("train_test_split.R")

fit.knn = function(x.train,
                   y.train,
                   x.test,
                   y.test,
                   n.folds,
                   k.max) {
  
  # Convert targets to factors.
  y.train = as.factor(y.train)
  y.test = as.factor(y.test)

  # Make cross-validation folds.
  set.seed(1)
  folds = createFolds(y = y.train,
                      k = n.folds,
                      list = TRUE,
                      returnTrain = FALSE)
  
  # Maximum possible number of principal components.
  smallest.fold = which.min(lengths(folds))[[1]]
  max.pcs = min(length(folds[[smallest.fold]]), ncol(x.train))
  
  # A function for combining matrices into arrays.
  acomb = function(...) abind(..., along = 3)

  # Perform n-fold cross-validation.
  error.array = foreach(i = 1:n.folds,
                        .combine = "acomb",
                        .multicombine = TRUE) %dopar% {

    y.train.folds  = y.train[-folds[[i]]]
    y.test.fold    = y.train[folds[[i]]]
    pc.train.folds = prcomp(x.train[-folds[[i]], ], scale = FALSE)
    pc.test.fold   = prcomp(x.train[folds[[i]], ], scale = FALSE)
    
    # Make a data frame to hold cross-validation errors.
    error.df = data.frame(matrix(0, nrow = max.pcs, ncol = k.max))
    
    # Test all possible principal components.
    for(j in 1:max.pcs) {
      
      subset.pc.train.folds = pc.train.folds$x[, 1:j]
      subset.pc.test.fold   = pc.test.fold$x[, 1:j]

      # Test all values of k.
      for(k in 1:k.max) {
      
        set.seed(1)
        cv.preds = class:::knn(data.frame(subset.pc.train.folds),                                                          
                               data.frame(subset.pc.test.fold),                                                           
                               y.train.folds,
                               k = k) 
        error.df[j, k] = length(which(cv.preds != y.test.fold))/
                             length(y.test.fold)
      
      }

    }

    return(error.df)
    
  }

  col.sums = apply(error.array, 1:2, sum)
  
  # Optimal number of PCs.
  n.pcs = which.min(apply(col.sums, MARGIN = 1, min))[[1]]
  
  # Best k.
  best.k = which.min(apply(col.sums, MARGIN = 2, min))[[1]]

  # Fit KNN model to the full training data and make predictions.
  pc.train = prcomp(x.train, scale = FALSE)
  pc.test  = (scale(x.test, pc.train$center, pc.train$scale) %*%
                        pc.train$rotation)
  set.seed(1)
  preds = class:::knn(data.frame(pc.train$x[, 1:n.pcs]),
                      data.frame(pc.test[, 1:n.pcs]),
                      y.train,
                      k = best.k)

  return(list(confusion.matrix = confusionMatrix(preds, y.test)))

}

mean.knn = fit.knn(x.train = mean.train[, -c(1:2)],
                   y.train = mean.train$DX_GROUP,
                   x.test = mean.test[, -c(1:2)],
                   y.test = mean.test$DX_GROUP,
                   n.folds = 10,
                   k.max = 50)

cor.knn = fit.knn(x.train = cor.train[, -c(1:2)],
                  y.train = cor.train$DX_GROUP,
                  x.test = cor.test[, -c(1:2)],
                  y.test = cor.test$DX_GROUP,
                  n.folds = 10,
                  k.max = 50)

dtw.knn = fit.knn(x.train = dtw.train[, -c(1:2)],
                  y.train = dtw.train$DX_GROUP,
                  x.test = dtw.test[, -c(1:2)],
                  y.test = dtw.test$DX_GROUP,
                  n.folds = 10,
                  k.max = 50)

cat("KNN with mean data:\n")
mean.knn$confusion.matrix
cat("\n")

cat("KNN with correlation data:\n")
cor.knn$confusion.matrix
cat("\n")

cat("KNN with dynamic time warped data:\n")
dtw.knn$confusion.matrix
cat("\n")