#!/usr/bin/env Rscript

# Author: Christopher J. Urban
# Date: 6/26/2018
#
# This script fits support vector machines to
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

# Register cores for parallel computations.
registerDoMC(cores = 4)

source("train_test_split.R")

fit.svm = function(x.train,
                   y.train,
                   x.test,
                   y.test) {

  # Convert targets to factors.
  y.train = as.factor(y.train)
  y.test = as.factor(y.test)
  
  # Combine training predictors and target into a single data frame.
  data.train = data.frame(x = x.train, y = y.train)

  # Set seeds to ensure results are reproducible.
  set.seed(123)
  seeds <- vector(mode = "list", length = 11)
  for(i in 1:10) seeds[[i]] <- sample.int(1000, 50)
  seeds[[11]] <- sample.int(1000, 1)
  
  # Control object to perform 10-fold cross-validation.
  ctrl = trainControl(method = "cv",
                      number = 10,
                      seeds = seeds)
  
  # Fit SVM with radial basis function.
  set.seed(1)
  cv.fit = train(form = y ~ .,
                 data = data.train,
                 method = "svmRadial", 
                 trControl = ctrl,
                 tuneLength = 30,
                 allowParallel = TRUE,
                 metric = "Accuracy")
  
  # Makes predict function work.
  colnames(x.test) = colnames(data.train[, -ncol(data.train)])
  
  # Make predictions.
  preds = predict(cv.fit, x.test)
  return(list(confusion.matrix = confusionMatrix(preds, y.test)))

}

mean.svm = fit.svm(x.train = mean.train[, -c(1:2)],
                   y.train = mean.train$DX_GROUP,
                   x.test = mean.test[, -c(1:2)],
                   y.test = mean.test$DX_GROUP)

cor.svm = fit.svm(x.train = cor.train[, -c(1:2)],                             
                   y.train = cor.train$DX_GROUP,                               
                   x.test = cor.test[, -c(1:2)],                               
                   y.test = cor.test$DX_GROUP)

dtw.svm = fit.svm(x.train = dtw.train[, -c(1:2)],                             
                   y.train = dtw.train$DX_GROUP,                               
                   x.test = dtw.test[, -c(1:2)],                               
                   y.test = dtw.test$DX_GROUP)

cat("SVM with mean data:\n")
mean.svm$confusion.matrix
cat("\n")

cat("SVM with correlation data:\n")
cor.svm$confusion.matrix
cat("\n")

cat("SVM with dynamic time warped data:\n")
dtw.svm$confusion.matrix
cat("\n")
