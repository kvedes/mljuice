# load packages -----------------------------------------------------------

pacman::p_load(readr, xgboost, Matrix, igraph, DiagrammeR, ggplot2, gridExtra, dplyr, Ckmeans.1d.dp)

# data and xgboost -----------------------------------------------------------------

# load the trainingdata
train <- read_csv("train.csv") %>% data.frame() %>% mutate_if(is.character, factor) 

# expanding to a sparse model matrix with one-hot encoding for factors
xg.data <- 
  sparse.model.matrix(~ . -1, 
                      model.frame(train[,-which(names(train) %in% "SalePrice")],na.action = na.pass), 
                      contrasts.arg = lapply(train[,sapply(train,is.factor)], contrasts, contrasts=FALSE))

# data in format for xgboost
xg.labelledData =xgb.DMatrix(xg.data, label=train$SalePrice)


# xgboost parameters
param <- list("booster"             = "gbtree", # specifying that the basis functions are regression trees
              "eta"                 = 0.05,     # specifying the learning rate 0<eta<1
              "gamma"               = 50,       # specifying loss for extra leaf node
              "max_depth"           = 5,        # maximum depth of tree
              "alpha"               = 0,        # L1 regularization
              "lambda"              = 1,        # L2 regularization
              "subsample"           = 0.5,      # how much of the data is used in subsampling for a new iteration(rows)
              "colsample_bytree"    = 0.8       # how much of the data is used in subsampling for a new iteration(collumns)
)

# run a 3-fold cv
cvData <- xgb.cv(params = param,
                 xg.labelledData,
                 nrounds = 2000,
                 verbose = TRUE,
                 nfold = 3,
                 metrics = c("mae")) 

# plot result of cv
ggplot(data=cvData$evaluation_log, aes(x=iter)) +
  geom_line(aes(y=train_mae_mean, color="train-error")) + 
  geom_line(aes(y=test_mae_mean, color="validation-error")) +
  labs(color="type of error",x="iterations", y="mean absolute error") 

# make a model 
ptm <- proc.time()
bst <- xgboost(params = param, 
              xg.labelledData, 
              nrounds = 2000,
              verbose = TRUE
)
proc.time() - ptm

# names of the regressors and the one-hot encoded factors
names = dimnames(xg.data)[[2]]

# plotting feature importance
importance_matrix = xgb.importance(names, model = bst)
xgb.ggplot.importance(importance_matrix, top_n = 20, n_clusters = c(1, 2))

# complexity plot
xgb.plot.deepness(model = bst)

# plotting a tree
xgb.plot.tree(model = bst, feature_names = as.character(names), n_first_tree = 2)

# partial dependence plot -------------------------------------------------

# partial dependence plot resolution
no_gridpoints = 30

# extracting most important feature
mostImpFeat <- importance_matrix$Feature[1]

# grid to evaluate features
grid <- seq(min(train[,mostImpFeat]),max(train[,mostImpFeat]), by = (max(train[,mostImpFeat])-min(train[,mostImpFeat]))/no_gridpoints)

# initiate empty vector to contain partial predictions
partialPredictions <- c()

# make partial predictions
for(grid_i in grid){
  # constructing temporary data-frame
  train_tmp <- train
  train_tmp[,mostImpFeat] <- rep(1,dim(train_tmp)[1]) * grid_i
  
  # names of the regressors and the one-hot encoded factors
  xgb.train_tmp <- sparse.model.matrix(~ . -1, 
                                       model.frame(train_tmp[,-which(names(train_tmp) %in% "SalePrice")],na.action = na.pass), 
                                       contrasts.arg = lapply(train_tmp[,sapply(train_tmp,is.factor)], contrasts, contrasts=FALSE))
  
  partialPredictions <- c(partialPredictions, mean(predict(bst, newdata = xgb.train_tmp)))
}


ggplot(data = cbind.data.frame(grid = grid, partialPredictions = partialPredictions), aes(grid)) + 
  geom_path(aes(y=partialPredictions)) + 
  labs(x= mostImpFeat,y="partial predicted price")
