#!/usr/bin/env Rscript

# Author: Christopher J. Urban
# Date: 6/25/2018
#
# This script performs logistic regression using principal components
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

# Register cores for parallel computations.
registerDoMC(cores = 4)

source("train_test_split.R")

logistic_regression = function(x.train,
                               y.train,
                               x.test,
                               y.test,
                               n.folds) {
  
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
  
  # Perform n-fold cross-validation.
  error.df = foreach(i = 1:n.folds, .combine = "rbind") %dopar% {
    
    y.train.folds  = y.train[-folds[[i]]]
    y.test.fold    = y.train[folds[[i]]]
    pc.train.folds = prcomp(x.train[-folds[[i]], ], scale = FALSE)
    pc.test.fold   = prcomp(x.train[folds[[i]], ], scale = FALSE)
    
    # Make a vector to hold errors.
    errors = rep(NA, max.pcs)
    
    # Test all possible principal components at each fold.
    for(j in 1:max.pcs) {
      
      cv.train = cbind.data.frame(y.train.folds, pc.train.folds$x[, 1:j])
      cv.test  = cbind.data.frame(y.test.fold, pc.test.fold$x[, 1:j])
      
      # This makes the predict function work.
      colnames(cv.test) = colnames(cv.train)
      
      cv.fit = glm(y.train.folds ~ ., data = cv.train, family = binomial)
      cv.probs = predict(cv.fit, cv.test, type = "response")
      cv.preds = rep(0, nrow(cv.test))
      cv.preds[cv.probs > 0.5] = 1
      
      errors[j] = length(which(cv.preds != y.test.fold))/
                      length(y.test.fold)
      
    }
    
    return(errors)
    
  }
  
  # Output optimal number of PCs.
  n.pcs = which.min(colSums(error.df))[[1]]
    
  # Run logistic regression with optimal number of PCs.
  pc.train = prcomp(x.train, scale = FALSE)
  data.train = cbind.data.frame(y = y.train, pc.train$x[, 1:n.pcs])
  fit = glm(y ~ ., data = data.train, family = binomial)
  
  # Evaluate performance on the test data set.
  pc.test = as.data.frame((scale(x.test, pc.train$center, pc.train$scale) %*% 
                pc.train$rotation)[, 1:n.pcs])
  probs = predict(object = fit, newdata = pc.test, type = "response")
  preds = rep(0, length(y.test))
  preds[probs > 0.5] = 1
  preds = as.factor(preds)

  return(list(confusion.matrix = confusionMatrix(preds, y.test)))
  
}

mean.logistic = logistic_regression(x.train = mean.train[, -c(1:2)],
                                    y.train = mean.train$DX_GROUP,
                                    x.test = mean.test[, -c(1:2)],
                                    y.test = mean.test$DX_GROUP,
                                    n.folds = 10)

cor.logistic = logistic_regression(x.train = cor.train[, -c(1:2)],                      
                                   y.train = cor.train$DX_GROUP,                               
                                   x.test = cor.test[, -c(1:2)],                              
                                   y.test = cor.test$DX_GROUP,                                
                                   n.folds = 10)

dtw.logistic = logistic_regression(x.train = dtw.train[, -c(1:2)],                      
                                   y.train = dtw.train$DX_GROUP,                               
                                   x.test = dtw.test[, -c(1:2)],                              
                                   y.test = dtw.test$DX_GROUP,                                
                                   n.folds = 10)

cat("Logistic regression with mean data accuracy:\n")
mean.logistic$confusion.matrix
cat("\n")

cat("Logistic regression with correlation data accuracy:\n")
cor.logistic$confusion.matrix
cat("\n")

cat("Logistic regression with dynamic time warped data accuracy:\n")
dtw.logistic$confusion.matrix
cat("\n")
