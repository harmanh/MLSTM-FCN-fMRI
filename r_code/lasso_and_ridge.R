#!/usr/bin/env Rscript

# Author: Christopher J. Urban
# Date: 6/26/2018
#
# This script performs LASSO or ridge regression using principal components
# on the ABIDE II data. PCs are constructed using
# the following functional connectivity metrics:
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

if(!require(glmnet)) {
  install.packages("glmnet", repos = "http://cran.us.r-project.org")
  library(glmnet)
}

# Register cores for parallel computations.
registerDoMC(cores = 4)

source("train_test_split.R")

lasso.or.ridge = function(x.train,
                          y.train,
                          x.test,
                          y.test,
                          n.folds,
                          method) {
  
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

  # Make a grid of lambdas.
  grid = 10^seq(10, -2, length = 100)
  
  if (method == "lasso") {
    alpha = 1
  } else if (method == "ridge") {
    alpha = 0
  }

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
    error.df = data.frame(matrix(0, nrow = max.pcs - 1, ncol = length(grid)))
    
    # Test all possible principal components.
    for(j in 2:max.pcs) {
      
      subset.pc.train.folds = pc.train.folds$x[, 1:j]
      subset.pc.test.fold   = pc.test.fold$x[, 1:j]
      
      cv.fit = glmnet(subset.pc.train.folds,
                      y.train.folds,
                      alpha = alpha,
                      lambda = grid,
                      family = "binomial")
      
      # Test each lambda in the grid.
      for(k in 1:length(grid)) {
       
        cv.probs = predict(cv.fit,
                           s = grid[k],
                           newx = subset.pc.test.fold)
        cv.preds = rep(0, nrow(subset.pc.test.fold))
        cv.preds[cv.probs > 0.5] = 1
        error.df[j - 1, k] = length(which(cv.preds != y.test.fold))/
                                 length(y.test.fold)
         
      }
      
    }
    
    return(error.df)
    
  }
  
  col.sums = apply(error.array, 1:2, sum)
  
  # Optimal number of PCs.
  n.pcs = which.min(apply(col.sums, MARGIN = 1, min))[[1]] + 1
  
  # Best lambda.
  lambda.index = which.min(apply(col.sums, MARGIN = 2, min))[[1]]
  best.lam = grid[lambda.index]
  
  # Fit model with optimal number of PCs and best lambda.
  pc.train = prcomp(x.train, scale = FALSE)
  fit = glmnet(pc.train$x[, 1:n.pcs],
               y.train,
               alpha = alpha,
               lambda = best.lam,
               family = "binomial")
  
  # Evaluate performance on the test data set.
  probs = predict(fit,
                  s = best.lam,
                  newx = (scale(x.test,
                  pc.train$center, pc.train$scale) %*%
                      pc.train$rotation)[, 1:n.pcs],
                  type = "response")
  preds = rep(0, length(y.test))
  preds[probs > 0.5] = 1
  preds = as.factor(preds)
  
  return(list(confusion.matrix = confusionMatrix(preds, y.test)))
  
}

mean.lasso = lasso.or.ridge(x.train = mean.train[, -c(1:2)],
                            y.train = mean.train$DX_GROUP,
                            x.test = mean.test[, -c(1:2)],
                            y.test = mean.test$DX_GROUP,
                            n.folds = 10,
                            method = "lasso")

cor.lasso = lasso.or.ridge(x.train = cor.train[, -c(1:2)],
                           y.train = cor.train$DX_GROUP,
                           x.test = cor.test[, -c(1:2)],
                           y.test = cor.test$DX_GROUP,
                           n.folds = 10,
                           method = "lasso")

dtw.lasso = lasso.or.ridge(x.train = dtw.train[, -c(1:2)],
                           y.train = dtw.train$DX_GROUP,
                           x.test = dtw.test[, -c(1:2)],
                           y.test = dtw.test$DX_GROUP,
                           n.folds = 10,
                           method = "lasso")

mean.ridge = lasso.or.ridge(x.train = mean.train[, -c(1:2)],                              
                            y.train = mean.train$DX_GROUP,                                
                            x.test = mean.test[, -c(1:2)],                               
                            y.test = mean.test$DX_GROUP,                                 
                            n.folds = 10,
                            method = "ridge")                                                 
                                                                                
cor.ridge = lasso.or.ridge(x.train = cor.train[, -c(1:2)],                                
                           y.train = cor.train$DX_GROUP,                                  
                           x.test = cor.test[, -c(1:2)],                                 
                           y.test = cor.test$DX_GROUP,                                   
                           n.folds = 10,
                           method = "ridge")                                                  
                                                                                
dtw.ridge = lasso.or.ridge(x.train = dtw.train[, -c(1:2)],                                
                           y.train = dtw.train$DX_GROUP,                                  
                           x.test = dtw.test[, -c(1:2)],                                 
                           y.test = dtw.test$DX_GROUP,                                   
                           n.folds = 10,
                           method = "ridge")

cat("LASSO regression with mean data accuracy:\n")
mean.lasso$confusion.matrix
cat("\n")

cat("LASSO regression with correlation data accuracy:\n")
cor.lasso$confusion.matrix
cat("\n")

cat("LASSO regression with dynamic time warped data accuracy:\n")
dtw.lasso$confusion.matrix
cat("\n")

cat("Ridge regression with mean data accuracy:\n")
mean.ridge$confusion.matrix
cat("\n")                                                                      
                                                                                
cat("Ridge regression with correlation data accuracy:\n")
cor.ridge$confusion.matrix
cat("\n")                                                                      
                                                                                
cat("Ridge regression with dynamic time warped data accuracy:\n")
dtw.ridge$confusion.matrix
cat("\n") 
