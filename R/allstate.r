# setwd("Kaggle/AllState/R")

library("dplyr")
library("caret")
library("gbm")

source("../../../PWMisc/R/UtilitySources.r")
source("../../../PWMisc/R/Graphics.r")

FileDir <- "../data"

minDays <- 2

rCombn <- combn(7, 2)
rLetters <- c("A", "B", "C", "D", "E", "F", "G")
rCombnLetters <- apply(rCombn, 2, function(ccc) paste0(rLetters[ccc], collapse=""))
response_vars <- c("customer_ID", paste0("purchase_", c(rLetters, rCombnLetters)))
  
###############################################################################
#
# exploreDistributions -
#
###############################################################################

exploreDistributions <- function(FileDir="../data"){
  
  #
  # Truncate the data so that it looks more like the test data
  # 
  
  #
  # Get test and train distributions
  #
  
  set.seed(1417)
  
  dat <- read.csv(paste0(FileDir, "/train.csv"))
  shopping_point_dat <- dat[dat$record_type==0, ]
  test_dat <- read.csv(paste0(FileDir, "/test_v2.csv"))

  customers_test <- unique(test_dat$customer_ID)
  customers_train <- unique(shopping_point_dat$customer_ID)
  
  distrib_test <- sapply(customers_test, function(x){
    sum(test_dat$customer_ID==x)
  })  
  
  distrib_train <- sapply(customers_train, function(x){
    sum(shopping_point_dat$customer_ID==x)
  })  
  
  distrib_test_center <- distrib_test - minDays
  distrib_train_center <- distrib_train - minDays
  
  #
  # Test some distributions
  #
  
  samps <- length(distrib_test_center)
  samps_train <- length(distrib_train_center)
  
  #Geometric - number of shopping points before a purchase point
  #Poisson - number of shopping points given a rate
  
  geom_LL <- function(p, xxx) -sum(dgeom(xxx, p, log=TRUE))
  pois_LL <- function(lambda, xxx) -sum(dpois(xxx, lambda, log=TRUE))
  
  geom_prob_test <- optimize(f=geom_LL, interval=c(0, 1), xxx=distrib_test_center)$minimum
  geom_rand_test <- rgeom(samps, geom_prob_test)
  
  pois_lambda_test <- optimize(f=pois_LL, interval=c(0, 100), xxx=distrib_test_center)$minimum
  pois_rand_test <- rpois(samps, pois_lambda_test)
    
  distrib_df <- data.frame(test=sort(distrib_test_center), geom=sort(geom_rand_test),
    pois=sort(pois_rand_test))
  diff_df <- data.frame(geom=distrib_df[, 1] - distrib_df[, 2],
    pois=distrib_df[, 1] - distrib_df[, 3])
  
  
  summary(diff_df)
  plot(distrib_df$test, distrib_df$geom)
  plot(distrib_df$test, distrib_df$pois)
  
  pois_lambda_train <- optimize(f=pois_LL, interval=c(0, 100), xxx=distrib_train_center)$minimum
  pois_rand_train <- rpois(samps, pois_lambda_train)
  
  x_diff <- pois_rand_train - pois_rand_test
  x_diff[x_diff < 0] <- 0
  
  pois_lambda_diff <- optimize(f=pois_LL, interval=c(0, 100), xxx=x_diff)$minimum
  pois_rand_diff <- rpois(samps_train, pois_lambda_diff)
  
  distrib_manip <- data.frame(customer_ID=customers_train, original=distrib_train_center)
  distrib_manip$pois_diff <- pois_rand_diff
  
  distrib_manip$new <- distrib_manip$original - distrib_manip$pois_diff
  distrib_manip$new[distrib_manip$new < 0] <- 0
  
  summary(distrib_test_center)
  summary(distrib_manip$original)
  summary(distrib_manip$new)
  
  #
  # Custom distribution
  #
  
  mean_test <- mean(distrib_test_center)
  mean_train <- mean(distrib_train_center)
  
  mean_diff <- mean_train - mean_test
  
  tab_test <- table(distrib_test_center) / length(distrib_test_center)
  tab_test[length(names(tab_test)) + 1] <- 0
  names(tab_test)[length(names(tab_test))] <- "10"
  tab_train <- table(distrib_train_center) / length(distrib_train_center) 
  cdf_test <- cumsum(tab_test)
  cdf_train <- cumsum(tab_train)
  
  #
  A_mat <- sapply(c(0:10), function(x, p){
    
    all_equal <- rep(1 / (x + 1), x + 1)
    desc <- seq(from=(x+1), to=1, length.out=x + 1) / ( (x+2)*(x+1) / 2 )
    
    out <- all_equal * p + desc * (1-p) 
    
    c(out, rep(0, 10 - x)) 
  }, p=0.5)

  sol <- A_mat %*% tab_train
  outcome_A <- cbind(sol, tab_test)
  
  # A_mat 
  # columns are the input train customer length
  # rows are the output test customer length
  
  
  #
  # Manipulate train distribution
  #
  transform_mat <- apply(A_mat, 2, cumsum)
  distr_mapping <- data.frame(customer_ID=customers_train, original=distrib_train_center,
    rand=runif(length(customers_train), 0, 1))
  
  distr_mapping$new <- apply(distr_mapping, 1, function(ccc){
    
    probs <- transform_mat[, ccc["original"] + 1]
    return(min(which(probs > ccc["rand"])) - 1)
    
  }) + 2 #Add back the centered 2
  distr_mapping$original <- distr_mapping$original + 2
  
  write.csv(distr_mapping, paste0(FileDir, "/distribution_mapping.csv"), row.names=FALSE)
  
  #
  # Remove from the training set some rows to match test set
  #
  
  ndx_keep <- c()
  
  for(rrr in 1:nrow(distr_mapping)){
    row_dat <- distr_mapping[rrr, ]
    cust <- row_dat$customer_ID
    ndx_keep <- c(ndx_keep, 
      which(shopping_point_dat$customer_ID == cust)[1:row_dat$new]
    )
  }
  
  output_dat <- shopping_point_dat[ndx_keep, ]
  write.csv(output_dat, paste0(FileDir, "/shopping_train.csv"), row.names=FALSE)
}

###############################################################################
#
# createFeatures -
#
###############################################################################

createFeatures <- function(FileDir="../data", train=TRUE){
  
  if(train){
  
    dat <- read.csv(paste0(FileDir, "/train.csv"))
    
    shopping_point_dat <- dat[dat$record_type==0, ]
    purchase_point_dat <- dat[dat$record_type==1, ]
    
    purchase_point_dat$time <- as.POSIXct(paste(Sys.Date(), purchase_point_dat$time))
    
    #
    # Get the truncated training data
    #
    
    Extension <- "/shopping_train.csv"
    
  }else{ #Test
    
    Extension <- "/test_v2.csv"
  }
  
  shopping_point_dat <- read.csv(paste0(FileDir, Extension))
  shopping_point_dat$time <- as.POSIXct(paste(Sys.Date(), shopping_point_dat$time))
  
  #
  # Create cross sectional feature set
  #
  
  customer_sp_dat <- group_by(shopping_point_dat, customer_ID)
  
  cross_section_dat <- summarise(customer_sp_dat, 
    
    count=length(day), 
    
    time_spent=as.numeric(time[length(time)] - time[1]),
    
    state=paste0(unique(state), collapse=","),

    homeowner=homeowner[length(homeowner)], # ~ 90% 
    
    avg_car_age=mean(car_age, na.rm=TRUE),
    median_car_age=median(car_age, na.rm=TRUE),
    min_car_age=min(car_age, na.rm=TRUE),
    max_car_age=max(car_age, na.rm=TRUE),
    last_car_age=car_age[length(car_age)],
    
    last_car_value=car_value[length(car_value)],
    
    risk_factor=risk_factor[length(risk_factor)], # ~ 90% 
    
    last_age_youngest=age_youngest[length(age_youngest)], # ~ 80% 
    
    last_age_oldest=age_oldest[length(age_oldest)], # ~ 80% 
    
    avg_C_previous=mean(C_previous, na.rm=TRUE), # ~ 88%
    max_C_previous=max(C_previous, na.rm=TRUE), # ~ 91%
    last_C_previous=C_previous[length(C_previous)], # ~ 93%
    
    max_duration_previous=max(duration_previous, na.rm=TRUE), # ~ 80%
    last_duration_previous=duration_previous[length(duration_previous)], # ~ 87%
    
    avg_cost=mean(cost, na.rm=TRUE),
    last_cost=cost[length(cost)],
    multiple_cost=length(unique(cost)) > 1,
    
    ###
    
    avg_A=mean(A, na.rm=TRUE),
    median_A=median(A, na.rm=TRUE),
    penult_A=A[length(A) - 1],
    last_A=A[length(A)],
    num_diff_A=length(unique(A)),
    
    avg_B=mean(B, na.rm=TRUE),
    last_B=B[length(B)],
    num_diff_B=length(unique(B)),
    
    avg_C=mean(C, na.rm=TRUE),
    penult_C=C[length(C) - 1],
    last_C=C[length(C)],
    num_diff_C=length(unique(C)),
    
    avg_D=mean(D, na.rm=TRUE),
    penult_D=D[length(D) - 1],
    last_D=D[length(D)],
    num_diff_D=length(unique(D)),
    
    avg_E=mean(E, na.rm=TRUE),
    penult_E=E[length(E) - 1],
    last_E=E[length(E)],
    num_diff_E=length(unique(E)),
    
    avg_F=mean(F, na.rm=TRUE),
    median_F=median(F, na.rm=TRUE),
    penult_F=F[length(F) - 1],
    last_F=F[length(F)],
    num_diff_F=length(unique(F)),
    
    avg_G=mean(G, na.rm=TRUE),
    min_G=min(G, na.rm=TRUE),
    median_G=median(G, na.rm=TRUE),
    penult_G=G[length(G) - 1],
    last_G=G[length(G)],
    num_diff_G=length(unique(G))
    
  )
  
  for(ccc in 1:ncol(cross_section_dat)){
    if(is.logical(cross_section_dat[, ccc])) cross_section_dat[, ccc] <- as.integer(cross_section_dat[, ccc])
  }
    
  if(train){
      
    for(ccc in 1:ncol(rCombn)){
      
      pair <- rLetters[rCombn[, ccc]]
      
      purchase_point_dat[, paste0(pair, collapse="")] <-
        apply(purchase_point_dat[, pair], 1, paste0, collapse="")
    }
    
    names(purchase_point_dat)[-1] <- paste0("purchase_", names(purchase_point_dat)[-1])
    response_vars <- c("customer_ID", paste0("purchase_", c(rLetters, rCombnLetters)))
    
    all_dat <- merge(purchase_point_dat[, response_vars], cross_section_dat, by="customer_ID")
    
    all_dat$purchase_changedMind <- ifelse(
      apply(all_dat[, paste0("last_", rLetters)], 1, paste0, collapse="") == 
      apply(all_dat[, paste0("purchase_", rLetters)], 1, paste0, collapse=""),
      0, 1)    
    
    #
    # Save features data.frame
    #
    
    if(!file.exists(paste0(FileDir, "/model_changedMind.rds"))){
      write.csv(all_dat, paste0(FileDir, "/features.csv"), row.names=FALSE)
      predictMindChanges()
    }
    
    model_mindChanges <- readRDS(paste0(FileDir, "/model_changedMind.rds"))
    
    probs <- predict.gbm(model_mindChanges, type="response")
    all_dat$prob_mind_changed <- probs
    
    all_dat <- all_dat[, names(all_dat) != "purchase_changedMind"]
    
    write.csv(all_dat, paste0(FileDir, "/features.csv"), row.names=FALSE)
    
  }else{ #Test
    
    if(!file.exists(paste0(FileDir, "/model_changedMind.rds"))){
      stop("No changedMind model has been built.")
    }
    
    model_mindChanges <- readRDS(paste0(FileDir, "/model_changedMind.rds"))
    
    probs <- predict.gbm(model_mindChanges, newdata=cross_section_dat, type="response")
    cross_section_dat$prob_mind_changed <- probs
    
    write.csv(cross_section_dat, paste0(FileDir, "/test_features.csv"), row.names=FALSE)
  }
  
}

###############################################################################
#
# predictMindChanges -
#
###############################################################################

predictMindChanges <- function(FileDir="../data"){
    
  dat <- read.csv(paste0(FileDir, "/train.csv"), stringsAsFactors=TRUE)
  all_dat <- read.csv(paste0(FileDir, "/features.csv"), stringsAsFactors=TRUE)

  changed_dat <- all_dat[, -c(1:29)]
  
  model_changedMind <- gbm(purchase_changedMind ~ ., data=changed_dat, distribution="adaboost",
    n.trees=1000, interaction.depth=4, shrinkage=0.1, cv.folds=10)
      
  saveRDS(model_changedMind, paste0(FileDir, "/model_changedMind.rds"))
}

###############################################################################
#
# buildModels -
#
###############################################################################

buildModels <- function(FileDir="../data"){
  
  print(paste0(Sys.time(), ": Building models."))
  
  #
  #Read in the data and clean
  #
  all_dat <- read.csv(paste0(FileDir, "/features.csv"))
  lll <- length(response_vars)
  totalCols <- ncol(all_dat)
  all_dat <- read.csv(paste0(FileDir, "/features.csv"), 
    colClasses=c(rep("factor", lll), rep(NA, totalCols - lll)), stringsAsFactors=TRUE)
  
#     all_dat$purchase_policy <- as.factor(apply(all_dat[, response_vars][, -1], 1, paste, collapse=""))
#     response_vars <- c(response_vars, "purchase_policy")
  
  #
  # Partition
  #
  set.seed(1234)
  
  ndx_train <- createDataPartition(1:nrow(all_dat), p=0.7999, list=FALSE)
  
  train_IDs <- all_dat$customer_ID[ndx_train]
  test_IDs <- all_dat$customer_ID[-ndx_train]
  
  dat_train <- all_dat[ndx_train, ]
  features_train <- dat_train[, -c(1:length(response_vars))]
  response_train <- dat_train[,  c(1:length(response_vars))]
  
  dat_test <- all_dat[-ndx_train, ]
  features_test <- dat_test[, -c(1:length(response_vars))]
  response_test <- dat_test[,  c(1:length(response_vars))]
  
  rm(all_dat)
  rm(dat_train)
  rm(dat_test)
  
  #
  # Set up parameters, sample model
  #
  
#     tr_control <- trainControl(method="cv", number=3)
#     tune_grid <- expand.grid(.interaction.depth=c(2), 
#       .n.trees=c(20), .shrinkage=c(0.1))
#     
#     model_gbm_sample <- train(x=features_train,
#                     y=as.factor(response_train[, 2]),
#                     method="gbm",
#                     metric="Accuracy",
#                     trControl=tr_control,
#                     tuneGrid=tune_grid)
  
  cv_folds <- 5
  n_trees <- 1000
  interaction <- 2
  shrinkage <- 0.05
  
  
  for(ndxResponse in c(13:length(response_vars))){
    
    #Set up the data frame for training
    resp_nm <- names(response_train)[ndxResponse]
    response_levels <- response_train[, ndxResponse]
    
    print(paste0(Sys.time(), ": Building model - ", resp_nm))
    
    browser()
    
    gbm_train_dat <- data.frame(features_train, response_train[, ndxResponse])
    names(gbm_train_dat)[ncol(gbm_train_dat)] <- resp_nm
    
    gbm_test_dat <- data.frame(features_test, response_test[, ndxResponse])
    names(gbm_test_dat)[ncol(gbm_test_dat)] <- resp_nm
    
    #Build the model
    model_gbm <- gbm(formula=as.formula(paste0(resp_nm, " ~ .")), 
      data=gbm_train_dat, distribution="multinomial", 
      cv.folds=cv_folds, n.cores=1,
      n.trees=n_trees, interaction.depth=interaction, shrinkage=shrinkage)
    
    #Make test predicitons
    probs_gbm <- predict.gbm(model_gbm, features_test, type="response")
    
    preds_gbm <- to_predictions(probs_gbm)
    
    levels(preds_gbm) <- levels(response_test[, ndxResponse])
    
    conf_matrix_gbm <- confusionMatrix(preds_gbm, response_test[, ndxResponse])
    
#       models[[ndxResponse - 1]] <- model_gbm
#       predictions[[ndxResponse - 1]] <- data.frame(customer_ID=response_test$customer_ID,
#         actual=response_test[, ndxResponse], preds=preds_gbm)
#       confusion_matrices[[ndxResponse - 1]] <- conf_matrix_gbm
    
    saveRDS(list(Models=model_gbm, Probs=probs_gbm, Preds=preds_gbm, TestIDs=test_IDs,
      Confusion=conf_matrix_gbm), 
      paste0(FileDir, "/model-", resp_nm, ".rds"))
    
    rm(model_gbm)
    rm(probs_gbm)
    rm(preds_gbm)
    rm(conf_matrix_gbm)
  }
  
  print(paste0(Sys.time(), ": Finished building models."))
}

###############################################################################
#
# examineModelFeatures -
#
###############################################################################

examineModelFeatures <- function(FileDir="../data"){
    
  model_dat <- lapply(response_vars[-1], function(rrr){
    readRDS(paste0(FileDir, "/model-", rrr, ".rds"))
  })
  
  names(model_dat) <- response_vars[-1]
  
  #TODO: change to dplyr
  
  library("plyr")
  variable_importance <- ldply(names(model_dat), function(mNm){
    mmm <- model_dat[[mNm]]
    summ_dat <- data.frame(summary(mmm$Models))
    summ_dat$Model <- mNm
    return(summ_dat)
  })
  
  variable_importance <- ddply(variable_importance, "var", summarize, Mean=mean(rel.inf), 
    Min=min(rel.inf), Max=max(rel.inf))
  
  variable_importance <- variable_importance[order(variable_importance$Max), ]
  
}

###############################################################################
#
# examineModels -
#
###############################################################################

examineModels <- function(FileDir="../data"){
  
  #Get the model data
  model_dat <- lapply(response_vars[-1], function(rrr){
    readRDS(paste0(FileDir, "/model-", rrr, ".rds"))
  })
  names(model_dat) <- response_vars[-1]
  
  test_IDs <- model_dat[[1]]$TestIDs
  
  #Get the test set of the data
  all_dat <- read.csv(paste0(FileDir, "/features.csv"), stringsAsFactors=TRUE)
  
  dat <- all_dat[is.element(all_dat$customer_ID, test_IDs), ]
  
  #Predictions
  single_predictions <- as.data.frame(sapply(model_dat, function(x) return(x$Preds)))
  names(single_predictions) <- paste0("pred_", names(single_predictions))
  
  
  #Confusion matrices    
  conf_matrix <- ldply(model_dat, function(x) return(x$Confusion$overall))
  conf_matrix <- data.frame(conf_matrix[, c(1:4)],
    levels=apply(single_predictions, 2, function(x) length(unique(x))))
  
  #
  # Calculate the actual predictions
  #
  
  prediction_dat <- lapply(model_dat, function(x){
    ldply(as.character(x$Preds), function(y) unlist(strsplit(y, split="")))
  })
  prediction_dat <- Reduce(cbind, prediction_dat)
  prediction_dat <- as.data.frame(apply(prediction_dat, 2, as.numeric))
  names(prediction_dat) <- c(rLetters, unlist(strsplit(rCombnLetters, "")))
  
  
  browser()
  
  
#     predictions <- as.data.frame(sapply(rLetters, function(lll){
# #       round(rowMeans(prediction_dat[names(prediction_dat)==lll]))
#       round(apply(prediction_dat[names(prediction_dat)==lll], 1, median))
#     }))
  
  prediction_dat_combo <- prediction_dat[, -c(1:7)]
  names(prediction_dat_combo) <- names(prediction_dat)[-c(1:7)]
  
#     preds0 <- dat[, paste0("last_", rLetters)]
#     preds1 <- prediction_dat[, 1:7]    
#     preds2 <- as.data.frame(sapply(rLetters, function(lll){
#       round(rowMeans(prediction_dat[names(prediction_dat)==lll]))
#     }))
#     preds3 <- as.data.frame(sapply(rLetters, function(lll){
#       round(apply(prediction_dat[names(prediction_dat)==lll], 1, median))
#     }))
#     preds4 <- as.data.frame(sapply(rLetters, function(lll){
#       round(rowMeans(prediction_dat_combo[names(prediction_dat_combo)==lll]))
#     }))
#     preds5 <- as.data.frame(sapply(rLetters, function(lll){
#       round(apply(prediction_dat_combo[names(prediction_dat_combo)==lll], 1, median))
#     }))
#     
#     preds6 <- preds0
#     preds6[dat$count <= 3, ] <- preds3[dat$count <= 3, ]
#     
#     threshLast <- 5
#     preds7 <- preds3
#     preds7[rowSums(preds7 == preds0) > threshLast, ] <- 
#       preds0[rowSums(preds7 == preds0) > threshLast, ] 
#     
#     threshDiff <- 2
#     preds8 <- preds7
#     preds8[rowSums(preds1 == preds0) < threshDiff, ] <- 
#       preds1[rowSums(preds1 == preds0) < threshDiff, ] 
  
  
  preds10 <- dat[, paste0("last_", rLetters)]
  
  preds11 <- as.data.frame(sapply(rLetters, function(lll){
    apply(prediction_dat[names(prediction_dat)==lll], 1, median)
  }))
      
  preds12 <- preds10
  preds12[dat$count <= 2, ] <- preds11[dat$count <= 2, ]
  
  
  lowerProb <- 0.525
  higherProb <- 0.8
  preds13 <- preds12
  preds13[dat$prob_mind_changed < lowerProb, ] <- preds10[dat$prob_mind_changed < lowerProb, ]
  preds13[dat$prob_mind_changed > higherProb, ] <- preds11[dat$prob_mind_changed > higherProb, ]
  
  preds14 <- preds13
  preds14[dat$last_car_age < 10 & dat$last_age_youngest > 60, ] <- preds10[
    dat$last_car_age < 10 & dat$last_age_youngest > 60, ]
  preds14[dat$state=="NY" & dat$prob_mind_changed > 0.65, ] <- preds12[
    dat$state=="NY" & dat$prob_mind_changed > 0.65, ]
  
  for(predictions in list(round(preds10), round(preds11), round(preds12), 
    round(preds13), round(preds14))){
  
    names(predictions) <- paste0("pred_purchase_", rLetters)
    
    #
    #Make the full prediction
    #
    
    predictions$pred_purchase_full <- apply(predictions, 1, function(x) paste0(x, collapse=""))
    
    dat$last_full <- apply(dat[, paste0("last_", rLetters)], 1, function(x) paste0(x, collapse=""))
    dat$purchase_full <- apply(dat[, paste0("purchase_", rLetters)], 1, function(x) paste0(x, collapse=""))
    
    full_prediction_dat <- cbind(dat, predictions)
    
    nExamples <- nrow(full_prediction_dat)
    
    full_prediction_dat$pred_correct <- full_prediction_dat$purchase_full ==  
      full_prediction_dat$pred_purchase_full
    
    full_prediction_dat$last_correct <- full_prediction_dat$purchase_full ==  
      full_prediction_dat$last_full
      
    full_prediction_dat$last_num_wrong <- rowSums(
      full_prediction_dat[, paste0("purchase_", rLetters)] != 
      full_prediction_dat[, paste0("last_", rLetters)])
    
    full_prediction_dat$pred_num_wrong <- rowSums(
      full_prediction_dat[, paste0("purchase_", rLetters)] != 
      full_prediction_dat[, paste0("pred_purchase_", rLetters)])
    
    
    wrong <- full_prediction_dat[full_prediction_dat$last_correct & !full_prediction_dat$pred_correct, ]
    all_wrong <- full_prediction_dat[!full_prediction_dat$pred_correct, ]
    last_wrong <- full_prediction_dat[!full_prediction_dat$last_correct, ]
    
#       print(colSums(all_wrong[, paste0("purchase_", rLetters)] != 
#           all_wrong[, paste0("pred_purchase_", rLetters)]))
    print(table(rowSums(all_wrong[, paste0("purchase_", rLetters)] != 
        all_wrong[, paste0("pred_purchase_", rLetters)])))
    
    print(sum(full_prediction_dat$pred_correct))
    
  }
  
}

###############################################################################
#
# exploreMisses -
#
###############################################################################

exploreMisses <- function(FileDir="../data"){
  
  #Get the model data
  model_dat <- lapply(response_vars[-1], function(rrr){
    readRDS(paste0(FileDir, "/model-", rrr, ".rds"))
  })
  names(model_dat) <- response_vars[-1]
  
  test_IDs <- model_dat[[1]]$TestIDs

  #Predictions
  single_predictions <- as.data.frame(sapply(model_dat, function(x) return(x$Preds)))
  names(single_predictions) <- paste0("pred_", names(single_predictions))
  
  
  #Confusion matrices    
  conf_matrix <- ldply(model_dat, function(x) return(x$Confusion$overall))
  conf_matrix <- data.frame(conf_matrix[, c(1:4)],
    levels=apply(single_predictions, 2, function(x) length(unique(x))))
  
  #
  # Calculate the actual predictions
  #
  
  prediction_dat <- lapply(model_dat, function(x){
    ldply(as.character(x$Preds), function(y) unlist(strsplit(y, split="")))
  })
  prediction_dat <- Reduce(cbind, prediction_dat)
  prediction_dat <- as.data.frame(apply(prediction_dat, 2, as.numeric))
  names(prediction_dat) <- c(rLetters, unlist(strsplit(rCombnLetters, "")))
  
  browser()
  rm(model_dat)
  
  #Get the test set of the data
  #
  
  all_dat <- read.csv(paste0(FileDir, "/features.csv"), stringsAsFactors=TRUE)
  
  dat <- all_dat[is.element(all_dat$customer_ID, test_IDs), ]
  
  
  prediction_dat_combo <- prediction_dat[, -c(1:7)]
  names(prediction_dat_combo) <- names(prediction_dat)[-c(1:7)]
  
  preds10 <- dat[, paste0("last_", rLetters)]
  
  preds11 <- as.data.frame(sapply(rLetters, function(lll){
    apply(prediction_dat[names(prediction_dat)==lll], 1, median)
  }))
      
  preds12 <- preds10
  preds12[dat$count <= 2, ] <- preds11[dat$count <= 2, ]
  
  
  lowerProb <- 0.525
  higherProb <- 0.8
  preds13 <- preds12
  preds13[dat$prob_mind_changed < lowerProb, ] <- preds10[dat$prob_mind_changed < lowerProb, ]
  preds13[dat$prob_mind_changed > higherProb, ] <- preds11[dat$prob_mind_changed > higherProb, ]
  
  preds14 <- preds13
  preds14[dat$last_car_age < 10 & dat$last_age_youngest > 60, ] <- preds10[
    dat$last_car_age < 10 & dat$last_age_youngest > 60, ]
  preds14[dat$state=="NY" & dat$prob_mind_changed > 0.65, ] <- preds12[
    dat$state=="NY" & dat$prob_mind_changed > 0.65, ]
      
  
  predictions <- round(preds14)
  
  names(predictions) <- paste0("pred_purchase_", rLetters)
  
  #
  #Make the full prediction
  #
  
  predictions$pred_purchase_full <- apply(predictions, 1, function(x) paste0(x, collapse=""))
  
  dat$last_full <- apply(dat[, paste0("last_", rLetters)], 1, function(x) paste0(x, collapse=""))
  dat$purchase_full <- apply(dat[, paste0("purchase_", rLetters)], 1, function(x) paste0(x, collapse=""))
  
  full_prediction_dat <- cbind(dat, predictions)
  
  nExamples <- nrow(full_prediction_dat)
  
  full_prediction_dat$pred_correct <- full_prediction_dat$purchase_full ==  
    full_prediction_dat$pred_purchase_full
  
  full_prediction_dat$last_correct <- full_prediction_dat$purchase_full ==  
    full_prediction_dat$last_full
    
  full_prediction_dat$last_num_wrong <- rowSums(
    full_prediction_dat[, paste0("purchase_", rLetters)] != 
    full_prediction_dat[, paste0("last_", rLetters)])
  
  full_prediction_dat$pred_num_wrong <- rowSums(
    full_prediction_dat[, paste0("purchase_", rLetters)] != 
    full_prediction_dat[, paste0("pred_purchase_", rLetters)])
  
  
  wrong <- full_prediction_dat[full_prediction_dat$last_correct & !full_prediction_dat$pred_correct, ]
  all_wrong <- full_prediction_dat[!full_prediction_dat$pred_correct, ]
  last_wrong <- full_prediction_dat[!full_prediction_dat$last_correct, ]
  last_wrong_also <- full_prediction_dat[!full_prediction_dat$last_correct & !full_prediction_dat$pred_correct, ]
  
  print(table(rowSums(all_wrong[, paste0("purchase_", rLetters)] != 
      all_wrong[, paste0("pred_purchase_", rLetters)])))
  
  print(sum(full_prediction_dat$pred_correct))
  
  #
  #
  #
  
  wrg <- last_wrong_also[last_wrong_also$pred_num_wrong == 1, ]
  full <- full_prediction_dat
  
  nwrong <- nrow(wrg)
  nfull <- nrow(full)
  
  resp <- c(2:8)
  vars <- c(c(30:50), 82)
  
  wrg_resp <- wrg[, resp]
  wrg_vars <- wrg[, vars]
  full_resp <- full[, resp]
  full_vars <- full[, vars]
  
  wrg_vars$homeowner <- as.factor(wrg_vars$homeowner)
  wrg_vars$multiple_cost <- as.factor(wrg_vars$multiple_cost)
  full_vars$homeowner <- as.factor(full_vars$homeowner)
  full_vars$multiple_cost <- as.factor(full_vars$multiple_cost)
  
  factorNdx <- !sapply(wrg_vars, is.numeric)
  
  wrg_vars_factors <- wrg_vars[, factorNdx]
  wrg_vars_num <- wrg_vars[, !factorNdx]
  full_vars_factors <- full_vars[, factorNdx]
  full_vars_num <- full_vars[, !factorNdx]
  
  wrg_vars_factors$dist <- "Wrong"
  wrg_vars_num$dist <- "Wrong"
  full_vars_factors$dist <- "All"
  full_vars_num$dist <- "All"
  
  numCharts <- lapply(names(full_vars_num)[names(full_vars_num)!="dist"], 
    function(var_name){
    
    newdat <- rbind(wrg_vars_num[, c(var_name, "dist")], 
                    full_vars_num[, c(var_name, "dist")])
    
    names(newdat)[1] <- "variable"
      
      
    ggplot(newdat, aes(x=variable, fill=dist)) + 
      geom_histogram(alpha = 0.4, color="black",
        aes(y = ..density..), position = 'identity') +
      scale_fill_brewer(palette="Set1") +
        theme_standard + labs(x=var_name, y=NULL, 
        title=paste0("Distributions: ", var_name))
  })
  
  factorCharts <- lapply(names(full_vars_factors)[names(full_vars_factors)!="dist"],
    function(var_name){
    
    newdat <- rbind(wrg_vars_factors[, c(var_name, "dist")], 
                    full_vars_factors[, c(var_name, "dist")])
    
    names(newdat)[1] <- "factor"
    newdat <- dcast(data.frame(table(newdat)), factor ~ dist, value.var="Freq")
    
    newdat[, 2] <- newdat[, 2] / nfull
    newdat[, 3] <- newdat[, 3] / nwrong
      
    dd <- melt(newdat, "factor")
      
    ggplot(dd, aes(x=factor, y=value, fill=variable)) + 
      geom_bar(alpha = 0.4, color="black", stat='identity') +
      scale_y_continuous(labels=percent) +
      scale_fill_brewer(palette="Set1") +
      theme_standard + labs(x=var_name, y=NULL, 
        title=paste0("Distributions: ", var_name))
  })
  
  browser()
  print("Save charts to pdf?")
  
  pdf(paste0(FileDir, "/variable_distributions.pdf"), width=11, height=8.5)
  sapply(numCharts, print)
  sapply(factorCharts, print)
  dev.off()
  
  browser()
  
  important_vars <- c("state", "prob_mind_changed", "last_car_age", "last_age_youngest")    
  # NY and PA
  # newer than 10
  # older than 60
  
  wrg_vars <- wrg[, vars]
  full_vars <- full[, vars]
  
  wrg_vars$PA <- 1
  wrg_vars$NY <- 1
  wrg_vars$newer_than_10 <- 1
  wrg_vars$older_than_60 <- 1
  wrg_vars$PA[wrg_vars$state!="PA"] <- 0
  wrg_vars$NY[wrg_vars$state!="NY"] <- 0
  wrg_vars$newer_than_10[wrg_vars$last_car_age > 10] <- 0
  wrg_vars$older_than_60[wrg_vars$last_age_youngest < 60] <- 0
  
  full_vars$PA <- 1
  full_vars$NY <- 1
  full_vars$newer_than_10 <- 1
  full_vars$older_than_60 <- 1
  full_vars$PA[full_vars$state!="PA"] <- 0
  full_vars$NY[full_vars$state!="NY"] <- 0 
  full_vars$newer_than_10[full_vars$last_car_age > 10] <- 0
  full_vars$older_than_60[full_vars$last_age_youngest < 60] <- 0
  
  new_vars <- c("PA", "NY", "newer_than_10", "older_than_60")
  
  wrg_vars <- wrg_vars[, new_vars]
  full_vars <- full_vars[, new_vars]
  
  wrg_vars$PAAndNewer <- wrg_vars$PA * wrg_vars$newer_than_10
  wrg_vars$PAAndOlder <- wrg_vars$PA * wrg_vars$older_than_60
  wrg_vars$NYAndNewer <- wrg_vars$NY * wrg_vars$newer_than_10
  wrg_vars$NYAndOlder <- wrg_vars$NY * wrg_vars$older_than_60
  wrg_vars$NewerAndOlder <- wrg_vars$newer_than_10 * wrg_vars$older_than_60
  
  full_vars$PAAndNewer <- full_vars$PA * full_vars$newer_than_10
  full_vars$PAAndOlder <- full_vars$PA * full_vars$older_than_60
  full_vars$NYAndNewer <- full_vars$NY * full_vars$newer_than_10
  full_vars$NYAndOlder <- full_vars$NY * full_vars$older_than_60
  full_vars$NewerAndOlder <- full_vars$newer_than_10 * full_vars$older_than_60
  
  dist_comp <- data.frame(Wrong=t(t(colMeans(wrg_vars))), All=t(t(colMeans(full_vars))))
  dist_comp$Diff <- dist_comp$Wrong - dist_comp$All
  
  new_vars <- c("NY", "older_than_60", "NewerAndOlder")
  
  wrg_vars <- wrg_vars[, new_vars]
  full_vars <- full_vars[, new_vars]
  
  browser()
  
  #
  #
  #
  
  ages_dat <- full[full$last_car_age < 10 & full$last_age_youngest > 60, ] # last
  ny_prob_dat <- full[full$state=="NY" & full$prob_mind_changed > 0.65, ] # pred
  
  sum(ages_dat$pred_correct)
  sum(ages_dat$last_correct)
  sum(ny_prob_dat$pred_correct)
  sum(ny_prob_dat$last_correct)
  
  #
  #
  #
  
}

###############################################################################
#
# generatePredictions -
#
###############################################################################

generatePredictions <- function(CalcPredictions=TRUE, lProb=0.525,
  hProb=0.8, FileDir="../data"){
  
  if(CalcPredictions){
    
    #Get the test data
    test_dat <- read.csv(paste0(FileDir, "/test_features.csv"), stringsAsFactors=TRUE)    
    
    #Load in the models
    models <- lapply(response_vars[-1], function(rrr){
      readRDS(paste0(FileDir, "/model-", rrr, ".rds"))$Model
    })
    names(models) <- response_vars[-1]
    
    feature_names <- rownames(summary(models[[1]]))
    
    features_test <- test_dat[, feature_names]
    
    #Make base predictions
    all_probs <- lapply(models, function(mod) predict(mod, features_test))
    all_preds <- sapply(all_probs, to_predictions)
    
#     predictions_output <- all_preds
    
    rm(models)
    
    #
    # Calculate the actual predictions
    #
    
    prediction_dat <- apply(all_preds, 2, function(x){
      ldply(x, function(y) unlist(strsplit(y, split="")))
    })
    prediction_dat <- Reduce(cbind, prediction_dat)
    prediction_dat <- as.data.frame(apply(prediction_dat, 2, as.numeric))
    names(prediction_dat) <- c(rLetters, unlist(strsplit(rCombnLetters, "")))
    
    write.csv(prediction_dat, paste0(FileDir, "/model_prediction_dat.csv"), 
      row.names=FALSE)
  }else{
    
    prediction_dat <- read.csv(paste0(FileDir, "/model_prediction_dat.csv"))
    test_dat <- read.csv(paste0(FileDir, "/test_features.csv"), stringsAsFactors=TRUE)  
    features_test <- test_dat
  }
  
  predictions_output <- create_full_predictions(prediction_dat, features_test, lProb, hProb)
  
  plan <- apply(predictions_output, 1, paste, collapse="")
  submission <- data.frame(customer_ID=test_dat$customer_ID, plan=plan)
  write.csv(submission, paste0(FileDir, "/submission_combn_probs_", lProb, "_", hProb,".csv"), 
    row.names=FALSE)
  
  browser()
}

###############################################################################
#
# to_predictions -
#
###############################################################################

to_predictions <- function(probs){
 return(as.factor(colnames(probs)[apply(probs, 1, function(x) which.max(x))]))
}

###############################################################################
#
# create_full_predictions -
#
###############################################################################

create_full_predictions <- function(raw_predictions, features_dat, lProb=0.525, hProb=0.8){
  
  preds_last <- features_dat[, paste0("last_", rLetters)]
  
  preds_median <- as.data.frame(sapply(rLetters, function(lll){
    apply(raw_predictions[names(raw_predictions)==lll], 1, median)
  }))
      
  preds_count <- preds_last
  preds_count[features_dat$count <= 2, ] <- preds_median[features_dat$count <= 2, ]
  
    
  preds_output <- preds_count
  preds_output[features_dat$prob_mind_changed < lProb, ] <- 
    preds_last[features_dat$prob_mind_changed < lProb, ]
  preds_output[features_dat$prob_mind_changed > hProb, ] <- 
    preds_median[features_dat$prob_mind_changed > hProb, ]
#     preds_output[features_dat$last_car_age < 10 & features_dat$last_age_youngest > 60, ] <- preds_last[
#       features_dat$last_car_age < 10 & features_dat$last_age_youngest > 60, ]
#     preds_output[features_dat$state=="NY" & features_dat$prob_mind_changed > 0.65, ] <- preds_count[
#       features_dat$state=="NY" & features_dat$prob_mind_changed > 0.65, ]
  
  return(preds_output)
}

