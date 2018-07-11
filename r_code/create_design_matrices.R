#!/usr/bin/env Rscript

# Author: Christopher J. Urban
# Date: 6/25/2018
#
# This script:
# (1) reads in the ABIDE II fMRI data, which consists of
#     one matrix of ROI time series for each subject,
# (2) reads in the ABIDE II phenotypic data,
# (3) computes the mean of
#     each subject's ROI time series,
# (4) computes the Pearson correlation between
#     each subject's ROI time series,
# (5) computes the dynamic time warping distance between
#     each subject's ROI time series. 

if(!require(doMC)) {
  install.packages("doMC", repos = "http://cran.us.r-project.org")
  library(doMC)
}

if(!require(dtw)) {
  install.packages("dtw", repos = "http://cran.us.r-project.org")
  library(dtw)
}

if(!require(parallelDist)) {
  install.packages("parallelDist", repos = "http://cran.us.r-project.org")
  library(parallelDist)
}

# Register cores for parallel computations.
registerDoMC(cores = 4)

#-------------------------------------------------------# 
# (1) Read in the fMRI data.
#-------------------------------------------------------# 

# Read the fMRI data filenames.
fmri.path = "../data/fmri/"
fmri.filenames = dir(fmri.path, pattern = ".csv")

# Make a vector of subject IDs.
subj.ids = substr(fmri.filenames,
                  start = regexpr("5[0-9]{4}", fmri.filenames),
                  stop = regexpr("5[0-9]{4}", fmri.filenames) + 4)

# Read in brain ROI data (ROI = region of interest) and
# convert ROI names to unique strings.
rois = read.csv("../data/brain_region_legend.csv")
rois = as.character(rois[, 2])
rois = make.unique(rois, sep = ".")

fmri.data = foreach(i = 1:length(fmri.filenames)) %dopar% {
  
  fmri.filename.path = paste(fmri.path, fmri.filenames[i], sep = "")
  subj.df = read.csv(fmri.filename.path)
  subj.df = subj.df[, -ncol(subj.df)]
  colnames(subj.df) = rois
  
  return(subj.df)
  
}

# Save fMRI data as an RData file.
save(subj.ids,
     fmri.data,
     file = paste(fmri.path,
                  "all_fmri_data.RData",
                  sep = ""))

#-------------------------------------------------------#                       
# (2) Read in the phenotypic data.                                                        
#-------------------------------------------------------# 

pheno.path = "../data/pheno/"

pheno.data = read.csv(paste(pheno.path,
                      "Phenotypic_V1_0b_preprocessed1.csv",
                       sep = ""))

# Subset the phenotypic data, keeping only
# individuals for whom we have fMRI data.
pheno.data = pheno.data[pheno.data$SUB_ID %in% subj.ids, ]

# Select the subject ID and diagnosis group columns.
cols = c("SUB_ID", "DX_GROUP")
pheno.data = pheno.data[, colnames(pheno.data) %in% cols]

# Recode diagnosis group (1 = Autism, 0 = No autism).
pheno.data$DX_GROUP = ifelse(pheno.data$DX_GROUP == 2, 0, 1)

#-------------------------------------------------------#                       
# (3) Compute the mean of
#     each ROI time series,
# (4) Compute the Pearson correlation between
#     each ROI time series,
# (5) Compute the DTW distance between
#     each ROI time series. 
#-------------------------------------------------------#

# Compute the ROI means.
mean.roi = foreach(i = 1:length(subj.ids), .combine = "rbind") %dopar% {

  colMeans(fmri.data[[i]])

}

# Compute the ROI Pearson correlations.
cor.roi = foreach(i = 1:length(subj.ids), .combine = "rbind") %dopar% {

  cor.df = cor(fmri.data[[i]], method = "pearson")
  cor.df[lower.tri(cor.df)]
                            
}

# Compute the ROI DTW distances.
dtw.roi = foreach(i = 1:length(subj.ids), .combine = "rbind") %dopar% {

  dtw.df = parDist(as.matrix(t(fmri.data[[i]])),
                   method = "dtw",
                   step.pattern = "symmetric2",
                   window.type = "none",
                   upper = T,
                   diag = T,
                   threads = 8)
  dtw.df = as.data.frame(as.matrix(dtw.df))
  dtw.df[lower.tri(dtw.df)]
                            
}

# Create all combinations of the ROI names and rename
# correlation and DTW data frame columns.
name.combos = combn(rois, 2, simplify = FALSE)
name.combos = sapply(name.combos,
                     function(x) paste(x[1], x[2], sep = " - "))
colnames(cor.roi) = name.combos
colnames(dtw.roi) = name.combos

# Reorder the phenotypic data based on vector of subject IDs.
pheno.data = pheno.data[match(subj.ids, pheno.data$SUB_ID), ]

# Create design matrices.
mean.data = cbind.data.frame(pheno.data, mean.roi)[order(pheno.data$SUB_ID), ]
cor.data  = cbind.data.frame(pheno.data, cor.roi)[order(pheno.data$SUB_ID), ]
dtw.data  = cbind.data.frame(pheno.data, dtw.roi)[order(pheno.data$SUB_ID), ]

#-------------------------------------------------------#                       
# Save results.
#-------------------------------------------------------#

design.matrix.path = "../data/design_matrices/"

# Save as an RData file.
save(subj.ids,
     mean.data,
     cor.data,
     dtw.data,
     file = paste(design.matrix.path,
                  "design_matrices.RData",
                  sep = ""))

# Save as CSVs.
write.csv(mean.data,
          paste(design.matrix.path, "roi_means_data.csv", sep = ""))
write.csv(cor.data,
          paste(design.matrix.path, "roi_cor_data.csv", sep = ""))
write.csv(dtw.data,  
          paste(design.matrix.path, "roi_dtw_data.csv", sep = ""))
