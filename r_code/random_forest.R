#!/usr/bin/env Rscript

# Author: Christopher J. Urban
# Date: 7/10/2018
#
# This script fits random forest models to
# the ABIDE II data. The following
# functional connectivity metrics are used:
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

if(!require(randomForest)) {
  install.packages("randomForest", repos = "http://cran.us.r-project.org")
  library(randomForest)
}

# Register cores for parallel computations.
registerDoMC(cores = 4)

source("train_test_split.R")

fit.rf = function(x.train,
                  y.train,
                  x.test,
                  y.test,
                  n.folds,
                  mtry.max) {
 
  # Convert targets to factors.
  y.train = as.factor(y.train)
  y.test = as.factor(y.test)
  
  # Make cross-validation folds.
  set.seed(1)
  folds = createFolds(y = y.train,
                      k = n.folds,
                      list = TRUE,
                      returnTrain = FALSE)
  
  # Perform n-fold cross-validation.
  error.df = foreach(i = 1:n.folds, .combine = "rbind") %dopar% {
    
    y.train.folds  = y.train[-folds[[i]]]
    y.test.fold    = y.train[folds[[i]]]
    x.train.folds  = x.train[-folds[[i]], ]
    x.test.fold    = x.train[folds[[i]], ]
    
    # Make a vector to hold errors.
    errors = rep(NA, mtry.max)
    
    # Test all values of m at each fold.
    for(j in 1:mtry.max) {
      
      set.seed(333)
      cv.fit = randomForest(x = x.train.folds,
                         y = y.train.folds,
                         mtry = j,
                         ntree = floor(sqrt(ncol(x.train.folds))))
      cv.preds = predict(cv.fit, newdata = x.test.fold, type = "response")
      errors[j] = length(which(cv.preds != y.test.fold))/
                      length(y.test.fold)
      
    }
    
    return(errors)
    
  }
  
  # Output optimal m.
  best.mtry = which.min(colSums(error.df))[[1]]
  
  # Fit random forest with optimal m.
  set.seed(444)
  fit = randomForest(x = x.train,
                     y = y.train,
                     mtry = best.mtry,
                     ntree = floor(sqrt(ncol(x.train))))
  preds = predict(fit, newdata = x.test, type = "response")
  
  return(list(confusion.matrix = confusionMatrix(preds, y.test)))
    
}

mean.rf = fit.rf(x.train = mean.train[, -c(1:2)],
                 y.train = mean.train$DX_GROUP,
                 x.test = mean.test[, -c(1:2)],
                 y.test = mean.test$DX_GROUP,
                 n.folds = 10,
                 mtry.max = 25)

cor.rf = fit.rf(x.train = cor.train[, -c(1:2)],                             
                y.train = cor.train$DX_GROUP,                               
                x.test = cor.test[, -c(1:2)],                               
                y.test = cor.test$DX_GROUP,
                n.folds = 10,
                mtry.max = 25)

dtw.rf = fit.rf(x.train = dtw.train[, -c(1:2)],                             
                y.train = dtw.train$DX_GROUP,                               
                x.test = dtw.test[, -c(1:2)],                               
                y.test = dtw.test$DX_GROUP,
                n.folds = 10,
                mtry.max = 25)

cat("Random forest with mean data:\n")
mean.rf$confusion.matrix
cat("\n")

cat("Random forest with correlation data:\n")
cor.rf$confusion.matrix
cat("\n")

cat("Random forest with dynamic time warped data:\n")
dtw.rf$confusion.matrix
cat("\n")
