library(fasttime)
library(data.table)
library(tidyverse)
library(zoo)
library(lubridate)
library(lightgbm)
library(xgboost)

train_path <- "mnt/ssd/kaggle-talkingdata2/competition_files/train.csv"
test_path <- "mnt/ssd/kaggle-talkingdata2/competition_files/test.csv"

train <-fread(train_path, nrows=1, header=TRUE)
col_train<- colnames(train)
train <- fread(train_path, col.names=col_train, skip=124903891, nrows=30000000, header=TRUE)

test <- fread(test_path)

train[, tot_ip := .N, by=.(ip)]
train[, tot_app := .N, by=.(app)]
train[, tot_channel := .N, by=.(channel)]
train[, tot_os := .N, by=.(os)]
train[, tot_device := .N, by=.(device)]

train[, ip_channel := .N, by=.(ip, channel)]
train[, ip_app := .N, by=.(ip, app)]
train[, app_channel := .N, by=.(app, channel)]
train[, app_device := .N, by=.(app, device)]
train[, app_os := .N, by=.(app, os)]
train[, ip_app_channel := .N, by=.(ip, app, channel)]
train[, ip_app_device := .N, by=.(ip, app, device)]

train[, ("attributed_time") := NULL]
train[, click_time := fastPOSIXct(click_time)]
train[, time := hour(click_time)*60 + minute(click_time)]
train[, time := as.integer(time)]
train[, day := day(click_time)]
inds <- unique(train$day)
train[, paste0("OH", inds) := lapply(inds, function(x) day == x)]

train[, vol := .N, by=.(time, day)]
train[, mnt_ip := .N, by=.(time, day, ip)]
train[, mnt_app := .N, by=.(time, day, app)]
train[, mnt_channel := .N, by=.(time, day, channel)]
train[, mnt_os := .N, by=.(time, day, os)]
train[, mnt_device := .N, by=.(time, day, device)]

train[, app_30 := rollsum(mnt_app, k=30, fill=as.integer(0)), by=.(app)]
train[, ip_30 := rollsum(mnt_ip, k=30, fill=as.integer(0)), by=.(ip)]
train[, os_30 := rollsum(mnt_os, k=30, fill=as.integer(0)), by=.(os)]
train[, channel_30 := rollsum(mnt_channel, k=30, fill=as.integer(0)), by=.(channel)]
train[, device_30 := rollsum(mnt_device, k=30, fill=as.integer(0)), by=.(device)]

train[, app_lft := rollsum(mnt_app, k=45, fill=as.integer(0), align="left"), by=.(app)]
train[, ip_lft := rollsum(mnt_ip, k=45, fill=as.integer(0), align="left"), by=.(ip)]
train[, os_lft := rollsum(mnt_os, k=45, fill=as.integer(0), align="left"), by=.(os)]
train[, channel_lft := rollsum(mnt_channel, k=45, fill=as.integer(0), align="left"), by=.(channel)]
train[, device_lft := rollsum(mnt_device, k=45, fill=as.integer(0), align="left"), by=.(device)]

train[, app_rgt := rollsum(mnt_app, k=45, fill=as.integer(0), align="right"), by=.(app)]
train[, ip_rgt := rollsum(mnt_ip, k=45, fill=as.integer(0), align="right"), by=.(ip)]
train[, os_rgt := rollsum(mnt_os, k=45, fill=as.integer(0), align="right"), by=.(os)]
train[, channel_rgt := rollsum(mnt_channel, k=45, fill=as.integer(0), align="right"), by=.(channel)]
train[, device_rgt := rollsum(mnt_device, k=45, fill=as.integer(0), align="right"), by=.(device)]
train[, app_rgt_10 := rollsum(mnt_app, k=10, fill=as.integer(0), align="right"), by=.(app)]
train[, app_rgt_90 := rollsum(mnt_app, k=90, fill=as.integer(0), align="right"), by=.(app)]
train[, mnt_ip_by_app := .N, by=.(time, day, ip, app)]
train[, mnt_app_by_device := .N, by=.(time, day, device, app)]
train[, mnt_app_by_channel := .N, by=.(time, day, channel, app)]
train[, mnt_app_by_os := .N, by=.(time, day, os, app)]

train[, ip_by_app_45 := rollsum(mnt_ip_by_app, k=45, fill=as.integer(0)), by=.(app)]
train[, app_by_ip_45 := rollsum(mnt_ip_by_app, k=45, fill=as.integer(0)), by=.(ip)]
train[, garbage := rollsum(mnt_app, k=90, fill=as.integer(0)), by=.(device)]
train[, garbage2 := rollsum(mnt_ip, k=30, fill=as.integer(0)), by=.(channel)]
train[, garbage3 := rollsum(mnt_channel, k=60, fill=as.integer(0)), by=.(app)]

###feature importance
sub_train <- train[time %in% c(1130:1350, 30:210, 270:420)]

ip_table <- table(train[runif(10000,1,nrow(train))][["ip"]]) %>% sort(decreasing = TRUE)
ip_names <- ip_table %>% names
large_ip_train <- train[ip %in% ip_names[1:17]]

app_table <- table(train[runif(100000,1,nrow(train))][["app"]]) %>% sort(decreasing = TRUE)
app_names <- app_table %>% names
large_app_train <- train[app %in% app_names[1:17]]

sub_train <- test[seq(1,nrow(test), by= 10000)]
sub_test <- test[seq(1,nrow(test), by= 100)]

temptrain <- train[, lapply(.SD, as.numeric)]
dtrain <- temptrain
dtrain <- dtrain[,.(ip_channel,
                    device_rgt,
                    os_rgt,
                    channel_rgt,
                    ip_rgt,
                    app_rgt,
                    tot_ip, 
                    ip_30,
                    mnt_ip,
                    ip_lft,
                    tot_app, 
                    app_30,
                    mnt_app,
                    app_lft,
                    tot_channel,
                    channel_30,
                    mnt_channel,
                    channel_lft,
                    tot_os, 
                    os_30,
                    mnt_os,
                    os_lft,
                    tot_device,
                    device_30,
                    mnt_device,
                    device_lft,
                    time, 
                    vol)]


dtrain <- as.matrix(dtrain)
param <- list( objective 	= "binary:logistic", 
               grow_policy = "lossguide",
               tree_method = "hist",
               eval_metric = "auc", 
               max_depth = 10,
               max_leaves 	= 100, 
               max_delta_step = 7,
               scale_pos_weight = 590,
               eta = 0.1, 
               subsample = 0.7, 
               min_child_weight = 0,
               nthread = 4,
               colsample_bytree= 0.7, 
               colsample_bylevel=0.7)

params = list( eta = 0.3,
               tree_method= "hist",
               grow_policy= "lossguide",
               max_leaves= 1400,  
               max_depth= 0, 
               subsample= 0.9, 
               colsample_bytree= 0.7, 
               colsample_bylevel=0.7,
               min_child_weight=0,
               alpha=4,
               objective= 'binary:logistic', 
               scale_pos_weight=9,
               eval_metric= 'auc', 
               nthread=4, 
               silent = TRUE)
dtrain <- xgb.DMatrix(dtrain, label = as.matrix(as.numeric(train[,is_attributed])))
bst <- xgb.train(param, dtrain, nrounds=10)
xgb.importance(model = bst)

###boosting
tr_idx <- sample(train[,.I], floor(train[,.N]*.5))

temptrain <- train[, sample.int(ncol(train)), with=FALSE][, lapply(.SD, as.numeric), .SDcols = !"click_time"]
temptrain <- as.matrix(temptrain)
ltrain <- lgb.Dataset(data = temptrain[,which(colnames(temptrain) != 'is_attributed')], 
                      label=temptrain[,which(colnames(temptrain) == 'is_attributed')])
rm(temptrain)
gc()

lgb_params = list(
  boosting_type= 'gbdt',
  objective= 'binary',
  learning_rate= 0.15,
  num_leaves= 255,
  max_depth= 6, 
  min_child_samples= 100, 
  max_bin= 150, 
  subsample= 0.7,
  subsample_freq= 1, 
  colsample_bytree= 0.7, 
  min_child_weight= 0, 
  subsample_for_bin= 200000,
  min_split_gain= 0, 
  reg_alpha= 0, 
  reg_lambda= 0, 
  nthread= 4,
  verbose= 0,
  metric='auc',
  is_unbalance=TRUE
  )

lmodel <- lgb.cv(lgb_params, ltrain, nrounds=20, nfold = 5)


